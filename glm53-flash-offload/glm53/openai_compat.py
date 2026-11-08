"""OpenAI request compatibility for the chat templates' reasoning levels.

OpenAI clients send OpenAI's reasoning scale (minimal / low / medium / high), but a model's template has levels of its
own: the Qwen3.8 template accepts only low / medium / xhigh and raises on anything else, which surfaced as HTTP 500
(e.g. Hermes Agent sends "high"). normalize_chat_request() maps the request in place before the template sees it, with
the model's map (EFFORT_MAP, Qwen3.8's, unless a model server passes its own: serve.EFFORTS):

  reasoning_effort, chat_template_kwargs.reasoning_effort, reasoning.effort (Responses-style)
    Qwen3.8: minimal -> low, low -> low, medium -> medium, high / xhigh / max -> xhigh
    none / off / disabled -> thinking off (chat_template_kwargs.enable_thinking = false), for every model
    anything else -> dropped (the model's default effort applies)
"""

EFFORT_MAP = {"minimal": "low", "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"}
THINKING_OFF = ("none", "off", "disabled")


def normalize_chat_request(body, efforts=None):
    """Normalize reasoning controls in an OpenAI chat request dict, with `efforts` (OpenAI's level -> the template's;
    EFFORT_MAP when None). Returns the same dict."""
    efforts = EFFORT_MAP if efforts is None else efforts
    if not isinstance(body, dict):
        return body
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort") is not None and body.get("reasoning_effort") is None:
        body["reasoning_effort"] = reasoning["effort"]
    kw = body.get("chat_template_kwargs")
    if not isinstance(kw, dict):
        kw = None
    for holder in (body, kw):
        if holder is None or holder.get("reasoning_effort") is None:
            continue
        value = str(holder["reasoning_effort"]).strip().lower()
        if value in THINKING_OFF:
            del holder["reasoning_effort"]
            if not isinstance(body.get("chat_template_kwargs"), dict):
                body["chat_template_kwargs"] = {}
            body["chat_template_kwargs"]["enable_thinking"] = False
        elif value in efforts:
            holder["reasoning_effort"] = efforts[value]
        else:
            del holder["reasoning_effort"]
    return body
