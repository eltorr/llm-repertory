# Repertory API

OpenAI-compatible: the **Responses** API, and **Chat Completions** for compatibility. Both run on one pipeline, so the
model, sampling, tools and reasoning behave the same. Tested with the official `openai` Python SDK.

```
Base URL:  http://127.0.0.1:30000/v1
API key:   not checked (send any non-empty string if your client insists)
Model:     qwen3.8-flash-next   (any name works)
```

| Method | Path | Notes |
|---|---|---|
| POST | `/v1/responses` | The main API: streaming, function tools, reasoning control. See [Responses API](#responses-api) |
| POST | `/v1/chat/completions` | Supported for compatibility, with the same features. See [Chat Completions](#chat-completions) |
| POST | `/v1/completions` | Legacy raw-prompt completion |
| GET | `/v1/models` | Answered without loading the model |

Not available: embeddings, images, audio, files, batches, assistants and Anthropic's `/v1/messages`. These return
404.

## Responses API

| Field | Notes |
|---|---|
| `input` | A string, or items: messages (`user`, `assistant`, `system`, `developer`), `function_call` and `function_call_output`. Text only: images, files and audio return 400 |
| `instructions` | Becomes the system message |
| `stream` | The standard Responses events, ending with `response.completed`, or `response.incomplete` if cut off |
| `tools` | `function` tools only. Built-in tools (web search, file search and so on) are left out |
| `tool_choice` | `"none"` disables tools; anything else acts as `"auto"` |
| `previous_response_id` | The last 64 responses are kept in memory. After a restart, send the whole conversation instead. `store: false` skips saving |
| `reasoning.effort` | The model has `low`, `medium` and `xhigh` (the default). OpenAI's values are mapped: `minimal`→`low`, `high` and `max`→`xhigh`, `none`→thinking off |
| `max_output_tokens` | Default: the rest of the context window |
| `temperature`, `top_p` | `top_k`, `min_p` and `stop` work too |

The output has three kinds of item:
- a `reasoning` item, with the model's thinking;
- a `message` item, with the answer (`response.output_text` in the SDK);
- one `function_call` item per tool call.

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="local")

r = client.responses.create(model="qwen3.8-flash-next", input="Explain B-trees in two sentences.",
                            reasoning={"effort": "medium"})
print(r.output_text)
follow_up = client.responses.create(model="qwen3.8-flash-next", input="Now in one.", previous_response_id=r.id)
```

## Chat Completions

Supported for compatibility: OpenAI still supports Chat Completions and has announced no end to it (unlike the
Assistants API, which it shut down on 2026-08-26), and many clients and agents use it.

| Field | Notes |
|---|---|
| `messages` | `system`, `user`, `assistant` (with `tool_calls`) and `tool`. Text only |
| `stream` | Server-sent events |
| `max_tokens` | Default: the rest of the context window |
| `temperature`, `top_p`, `top_k`, `min_p`, `stop` | `top_k` and `min_p` go in `extra_body` with the SDK |
| `tools`, `tool_choice` | OpenAI function schema; calls come back as standard `tool_calls`. `tool_choice: "none"` disables tools; anything else acts as `"auto"` |
| `reasoning_effort` | Mapped as in Responses |
| `chat_template_kwargs.enable_thinking` | `false` turns thinking off. Put it in `extra_body` |
| `n`, `response_format`, penalties, `logprobs`, `seed` | Ignored |

- **Thinking:** returned in `message.reasoning_content`, and streamed before the answer. If the output is cut off
  while the model is thinking, `content` is empty.
- **Tool calls:** in a stream, prose before a call streams as it comes, and the calls arrive whole in the last
  delta. No tool-call markup reaches `content`.

```python
for chunk in client.chat.completions.create(
        model="qwen3.8-flash-next", stream=True, reasoning_effort="medium",
        messages=[{"role": "user", "content": "Write a Python LRU cache."}]):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Good to know

- **Sampling defaults** where a request leaves them out:
  - with thinking: `temperature` 1.0, `top_p` 0.95, `top_k` 20;
  - without thinking: `temperature` 0.7, `top_p` 0.8, `top_k` 20.
- **One request runs at a time.** Up to 8 more wait in a queue, so run agents with one request at a time.
- **Cold start:** if the model is paused, the first request waits about 40 s while it loads, so set client
  timeouts to at least 120 s.
- **Prefix cache:** a repeated 94K-token prompt took 0.14 s instead of 15 s. Usage reports the reused part:
  `usage.input_tokens_details.cached_tokens` (Responses), `usage.prompt_tokens_details.cached_tokens` (Chat
  Completions).
- **Errors:** in OpenAI's shape.
  - `503`: the model couldn't start; for example, other apps hold the GPU.
  - `502`: the model server is unreachable.
  - `400`: the prompt fills the context window (`context_length_exceeded`), the body isn't a request (a JSON object
    with `input`, or with `messages`), or the model's chat template refuses the conversation.

**Client settings:**
- **Context window:** 262144.
- **Max output tokens:** none, or 32K or more, since thinking can be long.
- **Temperature:** leave it unset.
- **Reasoning effort:** `medium` for agent loops.

## Control API (the dashboard's, not OpenAI)

Under `/_switcher/api/` on any port, with the dashboard token: a Bearer header, or the private link's cookie. The
token is `dashboard_token` in `config.json`. On the network port, `setup/run`, starting or cancelling a benchmark,
`quit`, `quit_keep_model`, autostart, `new_remote_link` and `remote_host` are refused: they're only for the PC itself.

| Method | Path | What |
|---|---|---|
| GET | `status` | State, profile, pressure, GPU and RAM, counters, speed, events |
| GET | `metrics` | The last 100 requests: tokens, time to first token, speeds |
| GET / POST | `config` | POST `{"config": {"quant": "4.05"}}`; keys as in `config.json`, except the paths, programs, ports and token, which only `config.json` changes |
| POST | `action` | `{"action": "start_full"}`. Also `start_lean`, `pause`, `restart`, `auto_on`/`auto_off`, `wake_on`/`wake_off`, `new_remote_link`, `quit`, `quit_keep_model` |
| GET | `log` | Tails of the model's and the manager's logs |
| GET / POST | `setup`, `setup/run` | Setup status; `{"step": "build"}` runs one step |
| GET / POST | `bench`, `bench/cancel` | Benchmark output and results (`?since=N`; `?run=NAME` for an earlier one); POST `{"runs": ["quants"], "workload": "quick"}` starts one, `{"resume": true}` continues the last one cut short |

**What's refused:**

- **Other websites:** a browser request from another site, a page on port 30000 that isn't addressed to
  `127.0.0.1`, `localhost` or `::1`, and a page on the network port that isn't one of the PC's names. That stops
  DNS rebinding.
- **GET on the completion routes:** they take POST only.
- **Malformed requests:** bad header characters, conflicting lengths, bodies over 1 MB (dashboard) or 64 MB
  (`/v1`), and requests that stall.
- **Direct calls to the model server:** it only answers requests that carry a key, which only the manager sends.
