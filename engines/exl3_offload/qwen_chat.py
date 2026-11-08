"""Qwen3.8's chat format, for the engine's hooks (server.py sets them for a qwen4_exp model): the prompt, its tool
calls, its sampling and its reasoning levels. glm_chat.py is GLM-5.3's; both have the same names.

  ChatTemplate      renders the model folder's chat template in Jinja's sandbox, as Hugging Face's tokenizers
                    do, so no Python from the model folder ever runs
  parse_tool_calls  the model's tool calls as OpenAI tool_calls
  with_sampling     the sampling values the model is meant to run with, where a request leaves them out
  EFFORTS           OpenAI's reasoning effort as the template's levels

The engine splits the answer into reasoning, content and tool calls (glm53/stream_split.py) at the tags below.
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

NAME = "qwen"
CALL_OPEN, CALL_CLOSE = "<tool_call>", "</tool_call>"
THINK_OPEN, THINK_CLOSE, TOOL_OPEN = "<think>", "</think>", CALL_OPEN
END_OF_TURN = ("<|im_end|>",)       # with the ids config.json and generation_config.json list

# The values the Qwen team recommends: one set with thinking on, one with it off
SAMPLING = {True: {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
            False: {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}}
# OpenAI's scale as the template's levels: it takes low, medium and xhigh and refuses anything else
# (glm53/openai_compat.py maps the request; none / off / disabled turn thinking off). The engine's own default.
EFFORTS = {"minimal": "low", "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"}


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
    public = True       # the template's own words, for the client (serve.request_error returns them)


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
