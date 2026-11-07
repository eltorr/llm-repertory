"""Qwen3.8's chat format, for the server's chat endpoints: the prompt, the answer's parts and its tool calls.

  ChatTemplate      renders the model folder's chat template in Jinja's sandbox, as Hugging Face's tokenizers
                    do, so no Python from the model folder ever runs
  AnswerStream      splits an answer arriving piece by piece into reasoning and content deltas, holding back only
                    what may still turn out to be markup, so prose streams even when tools are offered
  whole_answer      the same for a finished answer
  parse_tool_calls  the model's tool calls as OpenAI tool_calls
  with_sampling     the sampling values the model is meant to run with, where a request leaves them out

The format, as the model writes it (the prompt has already opened the think block when thinking is on):
  ...reasoning...</think>

  prose
  <tool_call>
  <function=NAME>
  <parameter=KEY>
  VALUE
  </parameter>
  </function>
  </tool_call>
"""
import json
import re
import secrets
import time
from pathlib import Path

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
CALL_OPEN, CALL_CLOSE = "<tool_call>", "</tool_call>"

# The values the Qwen team recommends: one set with thinking on, one with it off
SAMPLING = {True: {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
            False: {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}}


def thinking_on(body):
    return bool((body.get("chat_template_kwargs") or {}).get("enable_thinking", True))


def with_sampling(body):
    """The request with the model's sampling values filled in wherever it gives none (or null)."""
    out = dict(body)
    for key, value in SAMPLING[thinking_on(body)].items():
        if out.get(key) is None:
            out[key] = value
    return out


# ------------------------------------------------------------------------------------------------------------ prompt
class TemplateError(ValueError):
    """The template refused the conversation (its raise_exception), e.g. a role it doesn't know."""


def _refuse(message):
    raise TemplateError(message)


def _to_json(value, indent=None, ensure_ascii=False, sort_keys=False, separators=None):
    return json.dumps(value, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys, separators=separators)


class ChatTemplate:
    """A chat template, compiled once. Jinja's immutable sandbox: the template can't call into Python or change the
    data it is given."""

    def __init__(self, source):
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        jinja = ImmutableSandboxedEnvironment(extensions=["jinja2.ext.loopcontrols"], lstrip_blocks=True,
                                              trim_blocks=True)
        jinja.filters["tojson"] = _to_json
        jinja.globals.update(raise_exception=_refuse, strftime_now=time.strftime)
        self.template = jinja.from_string(source)

    @classmethod
    def from_model(cls, model_dir):
        """chat_template.jinja in the model folder, else the template in its tokenizer_config.json."""
        folder = Path(model_dir)
        path = folder / "chat_template.jinja"
        if path.is_file():
            return cls(path.read_text(encoding="utf-8"))
        source = json.loads((folder / "tokenizer_config.json").read_text(encoding="utf-8")).get("chat_template")
        if isinstance(source, list):          # several named templates: the default one
            source = next((t.get("template") for t in source if t.get("name") == "default"), None)
        if not isinstance(source, str):
            raise FileNotFoundError(f"no chat template in {folder}")
        return cls(source)

    def render(self, messages, tools=None, thinking=True, options=None):
        """The prompt for `messages`, ending where the model's answer starts. `options`: the request's
        chat_template_kwargs (reasoning_effort, ...); the arguments set here win over them."""
        extra = {k: v for k, v in (options or {}).items()
                 if k not in ("messages", "tools", "add_generation_prompt", "enable_thinking")}
        return self.template.render(messages=messages, tools=tools or None, add_generation_prompt=True,
                                    enable_thinking=thinking, **extra)


# -------------------------------------------------------------------------------------------------------- tool calls
CALL_FUNCTION = re.compile(r"<function=(?P<name>[^>\n]*)>(?P<body>.*?)(?:</function>|$)", re.S)
CALL_PARAMETER = re.compile(r"<parameter=(?P<key>[^>\n]*)>(?P<value>.*?)(?:</parameter>|(?=<parameter=)|$)", re.S)


def _schemas(tools):
    """Each offered function's parameter schemas, by function name."""
    out = {}
    for tool in tools or ():
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        if isinstance(fn, dict) and fn.get("name"):
            out[fn["name"]] = (fn.get("parameters") or {}).get("properties") or {}
    return out


def _value(raw, schema):
    """A parameter's value: the text as written for a string parameter (the format puts it on lines of its own),
    else JSON where it parses as JSON, else the text."""
    text = raw[1:] if raw.startswith("\n") else raw
    text = text[:-1] if text.endswith("\n") else text
    if (schema or {}).get("type") == "string":
        return text
    try:
        return json.loads(text)
    except ValueError:
        return text.strip()


def _calls_in(block, schemas):
    calls = []
    for fn in CALL_FUNCTION.finditer(block):
        function, params, values = fn["name"].strip(), {}, {}
        params = schemas.get(function, params)
        for p in CALL_PARAMETER.finditer(fn["body"]):
            values[p["key"].strip()] = _value(p["value"], params.get(p["key"].strip()))
        calls.append({"id": "call_" + secrets.token_hex(12), "type": "function",
                      "function": {"name": function, "arguments": json.dumps(values, ensure_ascii=False)}})
    return calls


def parse_tool_calls(text, tools):
    """(the text without its tool calls, [OpenAI tool calls]). Only when tools were offered; a call cut off by the
    end of the answer still counts. Text without a call that parses comes back as it was."""
    if not tools or CALL_OPEN not in text:
        return text, []
    schemas, calls, prose, at = _schemas(tools), [], [], 0
    while True:
        start = text.find(CALL_OPEN, at)
        if start < 0:
            prose.append(text[at:])
            break
        prose.append(text[at:start])
        end = text.find(CALL_CLOSE, start)
        stop = len(text) if end < 0 else end
        calls += _calls_in(text[start + len(CALL_OPEN):stop], schemas)
        at = len(text) if end < 0 else end + len(CALL_CLOSE)
    if not calls:
        return text, []
    return "".join(prose).strip(), calls


# ----------------------------------------------------------------------------------------------------------- answers
def _cut_tag(text, tag):
    """How many characters at the end of `text` could be the start of `tag` (it may arrive in the next piece)."""
    for n in range(min(len(tag) - 1, len(text)), 0, -1):
        if tag.startswith(text[-n:]):
            return n
    return 0


class AnswerStream:
    """An answer as it arrives, split into what a client may see now.

    feed(piece) and finish() return events: ("reasoning", text), ("content", text) and, from finish() only,
    ("tool_calls", calls). With thinking on, the text up to </think> is reasoning. With tools offered, everything
    from <tool_call> on is kept and parsed whole at the end. A tag cut in two by a piece boundary is held until the
    next piece shows what it is, so no markup reaches the client."""

    def __init__(self, thinking=True, tools=None):
        self.part = "reasoning" if thinking else "content"
        self.tools = tools
        self.held = ""           # arrived, not sent yet: a possible partial tag
        self.calls = ""          # from <tool_call> on

    def feed(self, piece):
        return self._split(piece, final=False)

    def finish(self):
        events = self._split("", final=True)
        if self.calls:
            rest, calls = parse_tool_calls(self.calls, self.tools)
            if rest:
                events.append(("content", rest))
            if calls:
                events.append(("tool_calls", calls))
        return events

    def _split(self, piece, final):
        text, self.held, events = self.held + piece, "", []
        while text:
            if self.part == "calls":
                self.calls += text
                break
            if self.part == "reasoning":
                before, found, after = text.partition(THINK_CLOSE)
                if found:
                    events.append(("reasoning", before.replace(THINK_OPEN, "")))
                    self.part, text = "content", after
                    continue
                tag = THINK_CLOSE
            else:
                start = text.find(CALL_OPEN) if self.tools else -1
                if start >= 0:
                    events.append(("content", text[:start]))
                    self.part, text = "calls", text[start:]
                    continue
                tag = CALL_OPEN if self.tools else None
            keep = 0 if final or tag is None else _cut_tag(text, tag)
            events.append((self.part, text[:len(text) - keep].replace(THINK_OPEN, "") if self.part == "reasoning"
                           else text[:len(text) - keep]))
            self.held = text[len(text) - keep:]
            break
        return [(kind, value) for kind, value in events if value]


def whole_answer(text, thinking=True, tools=None):
    """A finished answer: (reasoning or None, content, tool calls). A think block never closed (the answer ran out
    of tokens) is all reasoning."""
    reasoning = None
    if thinking:
        before, found, after = text.partition(THINK_CLOSE)
        reasoning, text = (before, after) if found else (text, "")
        reasoning = reasoning.replace(THINK_OPEN, "").strip()
    content, calls = parse_tool_calls(text.strip(), tools)
    return reasoning, content, calls
