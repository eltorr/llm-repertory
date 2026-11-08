"""GLM-5.3-Flash's chat format, for the engine's hooks (server.py sets them for a glm5_next model): the prompt, its tool
calls, its sampling and its reasoning levels. qwen_chat.py is Qwen3.8's; both have the same names.

  ChatTemplate      the model folder's chat_template.jinja in Jinja's sandbox (qwen_chat.ChatTemplate), with GLM's
                    thinking switch
  parse_tool_calls  the model's tool calls as OpenAI tool_calls
  with_sampling     the sampling values the model is meant to run with, where a request leaves them out
  EFFORTS           OpenAI's reasoning effort as the template's levels

The template always ends the generation prompt with "<think>" and has no enable_thinking: thinking off is an empty
think block, which this writes, as glm53-flash-offload's serve.py did for this model and as the template writes past
answers without reasoning. Its reasoning levels are Low, High and Max (the default). The format, as the model writes
it (the prompt has already opened the think block when thinking is on):
  ...reasoning...</think>prose<tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>...</tool_call>
"""
import json
import re
import secrets

import qwen_chat
from qwen_chat import TemplateError  # noqa: F401  (the template's refusal, as the engine answers it: a 400)

NAME = "glm"
THINK_OPEN, THINK_CLOSE, TOOL_OPEN = "<think>", "</think>", "<tool_call>"
CALL_CLOSE = "</tool_call>"
# the turn ends at the model's end of text or at the next turn's tag (generation_config.json lists the same ids)
END_OF_TURN = ("<|endoftext|>", "<|user|>", "<|observation|>")
# generation_config.json's values, with thinking on or off
SAMPLING = {True: {"temperature": 1.0, "top_p": 0.95}, False: {"temperature": 1.0, "top_p": 0.95}}
# OpenAI's scale (minimal, low, medium, high, and xhigh / max) as the template's reasoning levels: it takes low and
# high, and anything else is Max (glm53/openai_compat.py maps the request; none / off / disabled turn thinking off)
EFFORTS = {"minimal": "low", "low": "low", "medium": "high", "high": "high", "xhigh": "max", "max": "max"}


def thinking_on(body):
    return qwen_chat.thinking_on(body)


def with_sampling(body):
    """The request with the model's sampling values filled in wherever it gives none (or null)."""
    out = dict(body)
    for key, value in SAMPLING[thinking_on(body)].items():
        if out.get(key) is None:
            out[key] = value
    return out


# ------------------------------------------------------------------------------------------------------------ prompt
class ChatTemplate(qwen_chat.ChatTemplate):
    """GLM-5.3's template, compiled once in Jinja's immutable sandbox."""

    def render(self, messages, tools=None, thinking=True, options=None):
        """The prompt for `messages`, ending where the model's answer starts: in the open think block, or after an
        empty one with thinking off. `options`: the request's chat_template_kwargs (reasoning_effort, ...)."""
        prompt = super().render(messages, tools, thinking, options)
        if not thinking and prompt.endswith(THINK_OPEN):
            prompt += THINK_CLOSE
        return prompt


# -------------------------------------------------------------------------------------------------------- tool calls
CALL = re.compile(r"<tool_call>(?P<body>.*?)(?:</tool_call>|$)", re.S)
ARGUMENT = re.compile(r"<arg_key>(?P<key>.*?)</arg_key>\s*<arg_value>(?P<value>.*?)(?:</arg_value>|$)", re.S)


def _value(raw, schema):
    """A parameter's value: the text for a string parameter, else JSON where it parses as JSON, else the text."""
    text = raw.strip()
    if (schema or {}).get("type") == "string":
        return text
    try:
        return json.loads(text)
    except ValueError:
        return text


def _call(body, schemas):
    name = body.split("<arg_key>", 1)[0].strip()
    if not name or "<" in name:
        return None
    params = schemas.get(name, {})
    values = {a["key"].strip(): _value(a["value"], params.get(a["key"].strip())) for a in ARGUMENT.finditer(body)}
    return {"id": "call_" + secrets.token_hex(12), "type": "function",
            "function": {"name": name, "arguments": json.dumps(values, ensure_ascii=False)}}


def parse_tool_calls(text, tools):
    """(the text without its tool calls, [OpenAI tool calls]). Only when tools were offered; a call cut off by the
    end of the answer still counts. Text without a call that parses comes back as it was."""
    if not tools or TOOL_OPEN not in text:
        return text, []
    schemas = qwen_chat._schemas(tools)
    calls = [c for c in (_call(m["body"], schemas) for m in CALL.finditer(text)) if c]
    if not calls:
        return text, []
    return CALL.sub("", text).strip(), calls
