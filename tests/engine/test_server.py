"""The model server (engines/exl3_offload/server.py) on the engine's real app (glm53-flash-offload/glm53/serve.py): the
server wires its hooks in, and requests go through the ASGI app as a client sends them, with exllamav3 stubbed and a
fake job that streams a written answer through the engine's own run_job. Both APIs: streaming with tools, reasoning
then the answer, whole answers, the template's options and the model's sampling values, stop tokens, the prefix-cache
figures, the context limit and the 400s for requests that can't run. Needs torch (CPU) and fastapi, the engine's
environment; no model, no GPU."""
import argparse
import asyncio
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engines" / "exl3_offload"))
import qwen_chat  # noqa: E402
import server  # noqa: E402

CALL = ("<tool_call>\n<function=multiply>\n<parameter=a>\n17\n</parameter>\n<parameter=b>\n23\n</parameter>\n"
        "</function>\n</tool_call>")
PARAMETERS = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}
TOOLS = [{"type": "function", "function": {"name": "multiply", "parameters": PARAMETERS}}]
RESPONSES_TOOLS = [{"type": "function", "name": "multiply", "parameters": PARAMETERS}]
OFF = {"enable_thinking": False}
NEEDS = all(importlib.util.find_spec(m) for m in ("torch", "fastapi", "uvicorn"))


def stubs():
    """exllamav3 as serve.py imports it; nothing of it runs here (the job is Job)."""
    ex = types.ModuleType("exllamav3")
    ex.__path__ = []
    init = types.ModuleType("exllamav3.model_init")
    gen = types.ModuleType("exllamav3.generator")
    gen.__path__ = []
    gen.AsyncGenerator = gen.AsyncJob = object
    sampler = types.ModuleType("exllamav3.generator.sampler")
    sampler.ArgmaxSampler = sampler.ComboSampler = dict          # the sampling run_job picked, as given
    ex.model_init, ex.generator, gen.sampler = init, gen, sampler
    return {"exllamav3": ex, "exllamav3.model_init": init, "exllamav3.generator": gen,
            "exllamav3.generator.sampler": sampler}


class Job:
    """exllamav3's AsyncJob as the engine's run_job uses it: ANSWER four characters a step, then the end with the
    prompt tokens the prefix cache served (exllamav3's eos result counts whole pages and the rest of a partial one)."""
    ANSWER, CACHED, made = "", 0, []

    def __init__(self, generator, input_ids, max_new_tokens, sampler, stop_conditions, **kwargs):
        self.max_new_tokens, self.sampler, self.stop_conditions = max_new_tokens, sampler, stop_conditions
        Job.made.append(self)

    async def __aiter__(self):
        text = Job.ANSWER
        for i in range(0, len(text), 4):
            yield {"stage": "streaming", "text": text[i:i + 4], "eos": False}
        yield {"stage": "streaming", "text": "", "eos": True, "eos_reason": "stop_token",
               "new_tokens": -(-len(text) // 4), "cached_tokens": Job.CACHED}

    async def cancel(self):
        pass


class Tok:
    """The tokenizer as the engine and the server use it: every prompt is `length` tokens."""
    eos_token_id = None

    def __init__(self):
        self.prompts, self.length = [], 100

    def encode(self, text, encode_special_tokens=True):
        import torch
        self.prompts.append(text)
        return torch.zeros((1, self.length), dtype=torch.long)

    def single_id(self, text):
        return 7                                                     # <|im_end|>


class Template:
    """Qwen3.8's generation prompt (the think block left open with thinking on, written empty with it off), with what
    the template was given in the user turn; a role it doesn't know is refused, as the model's template does."""

    def render(self, messages, tools=None, thinking=True, options=None):
        for m in messages:
            if m["role"] not in ("system", "user", "assistant", "tool"):
                raise qwen_chat.TemplateError("Unexpected message role.")
        return (f"<|im_start|>user\ntools={bool(tools)} {options or {}}<|im_end|>\n<|im_start|>assistant\n"
                + ("<think>\n" if thinking else "<think>\n\n</think>\n\n"))


def post(app, path, body):
    """POST to the app as an HTTP client does: (status, the JSON body or the stream's text)."""
    data = body if isinstance(body, bytes) else json.dumps(body).encode()

    async def go():
        inbox, sent = [{"type": "http.request", "body": data, "more_body": False}], []

        async def receive():
            if inbox:
                return inbox.pop()
            await asyncio.Event().wait()                             # connected until the answer ends

        async def send(message):
            sent.append(message)
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}, "http_version": "1.1",
                 "method": "POST", "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
                 "root_path": "", "headers": [(b"host", b"127.0.0.1:30001"), (b"content-type", b"application/json")],
                 "client": ("127.0.0.1", 40000), "server": ("127.0.0.1", 30001)}
        await app(scope, receive, send)
        start = next(m for m in sent if m["type"] == "http.response.start")
        text = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body").decode()
        stream = dict(start["headers"]).get(b"content-type", b"").startswith(b"text/event-stream")
        return start["status"], text if stream else json.loads(text)
    return asyncio.run(go())


def chunks(stream):
    """A chat stream's chunks, and whether it ended with [DONE]."""
    blocks = stream.strip().split("\n\n")
    return [json.loads(b[len("data: "):]) for b in blocks if b.startswith("data: {")], blocks[-1] == "data: [DONE]"


def deltas(objects):
    return [c["choices"][0]["delta"] for c in objects]


def events(stream):
    """A Responses stream's events, each checked against its event: line."""
    out = []
    for block in stream.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        data = json.loads(lines["data"])
        if lines["event"] != data["type"]:
            raise AssertionError(block)
        out.append(data)
    return out


LIVE = types.SimpleNamespace()       # the engine module and its tokenizer, once setUpModule wired the server in


def setUpModule():
    """One server for every test, its hooks set once, as main() sets them."""
    if not NEEDS:
        return
    import fastapi  # noqa: F401  (imported before the stubs, so they outlive them)
    import torch  # noqa: F401
    import uvicorn  # noqa: F401
    LIVE.patches = [mock.patch.dict(sys.modules, stubs())]
    LIVE.patches[0].start()
    sys.path.insert(0, str(server.ENGINE))
    import serve
    LIVE.e, LIVE.tok = serve, Tok()
    cli = argparse.Namespace(model=str(ROOT / "no-model"), cache=1000, chunk=2048, host="127.0.0.1", port=30001,
                             served_name="qwen3.8-flash-next", mtp=0)
    loaded = (None, types.SimpleNamespace(eos_token_id_list=[]), None, LIVE.tok, None, None, None, None)
    server.Server(cli, serve, loaded, Template(), ram=None, vram=None)
    LIVE.patches.append(mock.patch.object(serve, "AsyncJob", Job))
    LIVE.patches[1].start()


def tearDownModule():
    for p in reversed(getattr(LIVE, "patches", [])):
        p.stop()
    if hasattr(LIVE, "e"):
        sys.path.remove(str(server.ENGINE))


@unittest.skipUnless(NEEDS, "needs torch and fastapi (the engine's environment)")
class EngineRoutes(unittest.TestCase):
    def setUp(self):
        self.e, self.tok = LIVE.e, LIVE.tok
        Job.ANSWER, Job.CACHED, Job.made[:] = "Hi.", 0, []
        self.tok.prompts.clear()
        self.tok.length = 100

    def chat(self, body):
        return post(self.e.app, "/v1/chat/completions", body)

    def responses(self, body):
        return post(self.e.app, "/v1/responses", body)


class WiringTests(EngineRoutes):
    def test_the_engine_answers_both_apis_and_the_hooks_are_set(self):
        routes = [r for r in self.e.app.router.routes if hasattr(r, "path")]
        paths = [r.path for r in routes]
        for path, endpoint in (("/v1/responses", self.e.responses), ("/v1/chat/completions", self.e.chat)):
            self.assertEqual(paths.count(path), 1)
            self.assertIs(next(r for r in routes if r.path == path).endpoint, endpoint)    # the engine's own
        for path in ("/v1/completions", "/v1/models", "/memory", "/expert_cache/resize", "/expert_cache/ram",
                     "/expert_cache/ram/stop"):
            self.assertIn(path, paths)
        self.assertEqual(self.e.EOS, [7])
        self.assertIs(self.e.parse_tool_calls, qwen_chat.parse_tool_calls)
        self.assertTrue(self.e.chat_prompt([{"role": "user", "content": "q"}]).endswith("<think>\n"))
        self.assertEqual((self.e.run_job.__module__, self.e.AsyncJob), ("serve", Job))   # not replaced by the server

    def test_sampling_defaults_and_stop_tokens_reach_the_job(self):
        self.chat({"messages": [{"role": "user", "content": "q"}], "chat_template_kwargs": OFF, "stop": "END"})
        job = Job.made[-1]
        self.assertEqual(job.sampler, {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0})
        self.assertEqual(job.stop_conditions, [7, "END"])
        self.assertEqual(job.max_new_tokens, 1000 - 100 - 8 - 2)       # no output cap: the rest of the context
        self.responses({"input": "q", "temperature": 0.2, "max_output_tokens": 50})
        job = Job.made[-1]
        self.assertEqual((job.sampler, job.max_new_tokens), ({"temperature": 0.2, "top_p": 0.95, "top_k": 20,
                                                              "min_p": 0.0}, 50))


class ChatStreamTests(EngineRoutes):
    def test_prose_streams_with_tools_offered_and_the_call_comes_whole(self):
        Job.ANSWER = "Let me multiply those numbers for you.\n" + CALL
        status, stream = self.chat({"messages": [{"role": "user", "content": "17 x 23?"}], "stream": True,
                                    "tools": TOOLS, "chat_template_kwargs": OFF})
        objects, done = chunks(stream)
        self.assertEqual((status, done), (200, True))
        d = deltas(objects)
        content = [x["content"] for x in d[1:] if x.get("content")]
        self.assertGreater(len(content), 3)                                    # streamed, not held to the end
        self.assertEqual("".join(content), "Let me multiply those numbers for you.\n")
        calls = [c for x in d if x.get("tool_calls") for c in x["tool_calls"]]
        self.assertEqual(len(calls), 1)
        self.assertEqual((calls[0]["index"], calls[0]["type"], calls[0]["function"]["name"]),
                         (0, "function", "multiply"))
        self.assertTrue(calls[0]["id"].startswith("call_"))
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"a": 17, "b": 23})
        self.assertEqual(objects[-1]["choices"][0]["finish_reason"], "tool_calls")
        self.assertNotIn("<tool_call>", "".join(content))

    def test_reasoning_then_the_answer(self):
        Job.ANSWER = "Short sum.</think>\n\n40"
        _, stream = self.chat({"messages": [{"role": "user", "content": "17 + 23?"}], "stream": True})
        objects, done = chunks(stream)
        d = deltas(objects)
        self.assertEqual(d[0], {"role": "assistant", "content": ""})
        self.assertEqual("".join(x.get("reasoning_content", "") for x in d), "Short sum.")
        self.assertEqual("".join(x.get("content", "") for x in d[1:]), "\n\n40")
        last = objects[-1]
        self.assertEqual((last["choices"][0]["finish_reason"], last["choices"][0]["delta"], done), ("stop", {}, True))
        for c in objects:
            self.assertEqual((c["object"], c["model"], c["choices"][0]["index"]),
                             ("chat.completion.chunk", "qwen3.8-flash-next", 0))
            self.assertTrue(c["id"].startswith("chatcmpl-"))
        self.assertTrue(all(c["choices"][0]["finish_reason"] is None for c in objects[:-1]))
        self.assertEqual(sorted(last["usage"]), ["completion_tokens", "prompt_tokens", "prompt_tokens_details",
                                                 "total_tokens"])

    def test_usage_reports_the_tokens_from_the_prefix_cache(self):
        Job.ANSWER, Job.CACHED, self.tok.length = "Done.", 39 * 256 + 16, 10_000
        with mock.patch.object(self.e, "CTX", 20_000):
            _, stream = self.chat({"messages": [{"role": "user", "content": "doc"}], "stream": True})
            _, whole = self.chat({"messages": [{"role": "user", "content": "doc"}]})
        for usage in (chunks(stream)[0][-1]["usage"], whole["usage"]):
            self.assertEqual(usage["prompt_tokens"], 10_000)
            self.assertEqual(usage["total_tokens"], 10_000 + usage["completion_tokens"])
            self.assertEqual(usage["prompt_tokens_details"], {"cached_tokens": 39 * 256 + 16})


class ChatWholeTests(EngineRoutes):
    def test_a_tool_call_without_streaming(self):
        Job.ANSWER = "Sure.\n" + CALL
        status, body = self.chat({"messages": [{"role": "user", "content": "q"}], "tools": TOOLS,
                                  "chat_template_kwargs": OFF})
        msg = body["choices"][0]["message"]
        self.assertEqual((status, body["object"], msg["role"], msg["content"], msg["reasoning_content"]),
                         (200, "chat.completion", "assistant", "Sure.", None))
        self.assertEqual(body["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(json.loads(msg["tool_calls"][0]["function"]["arguments"]), {"a": 17, "b": 23})
        self.assertEqual(body["usage"]["prompt_tokens_details"], {"cached_tokens": 0})

    def test_reasoning_and_content(self):
        Job.ANSWER = "Short sum.</think>\n\n40"
        _, body = self.chat({"messages": [{"role": "user", "content": "q"}]})
        msg = body["choices"][0]["message"]
        self.assertEqual((msg["reasoning_content"], msg["content"], body["choices"][0]["finish_reason"]),
                         ("Short sum.", "40", "stop"))
        self.assertNotIn("tool_calls", msg)

    def test_tool_choice_none_offers_no_tools(self):
        Job.ANSWER = CALL
        _, body = self.chat({"messages": [{"role": "user", "content": "q"}], "tools": TOOLS, "tool_choice": "none",
                             "chat_template_kwargs": OFF})
        self.assertIn("tools=False", self.tok.prompts[0])
        self.assertNotIn("tool_calls", body["choices"][0]["message"])
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")

    def test_reasoning_levels_reach_the_template(self):
        self.chat({"messages": [{"role": "user", "content": "q"}], "reasoning_effort": "high"})
        self.assertIn("'reasoning_effort': 'xhigh'", self.tok.prompts[0])
        self.chat({"messages": [{"role": "user", "content": "q"}], "reasoning_effort": "none"})
        self.assertTrue(self.tok.prompts[1].endswith("<think>\n\n</think>\n\n"))      # thinking off
        self.responses({"input": "q", "reasoning": {"effort": "minimal"}})
        self.assertIn("'reasoning_effort': 'low'", self.tok.prompts[2])


class ResponsesTests(EngineRoutes):
    def test_a_streamed_response_with_reasoning_and_a_call(self):
        Job.ANSWER, Job.CACHED = "Need the product.</think>\n\nLet me multiply.\n" + CALL, 64
        status, stream = self.responses({"input": "17 x 23?", "tools": RESPONSES_TOOLS, "stream": True})
        ev = events(stream)
        self.assertEqual(status, 200)
        self.assertEqual([e["type"] for e in ev[:2]], ["response.created", "response.in_progress"])
        self.assertEqual("".join(e["delta"] for e in ev if e["type"] == "response.reasoning_text.delta"),
                         "Need the product.")
        self.assertEqual("".join(e["delta"] for e in ev if e["type"] == "response.output_text.delta"),
                         "\n\nLet me multiply.\n")
        done = ev[-1]["response"]
        self.assertEqual((ev[-1]["type"], done["status"]), ("response.completed", "completed"))
        self.assertEqual([o["type"] for o in done["output"]], ["reasoning", "message", "function_call"])
        call = done["output"][2]
        self.assertEqual((call["name"], json.loads(call["arguments"])), ("multiply", {"a": 17, "b": 23}))
        self.assertEqual((done["usage"]["input_tokens"], done["usage"]["input_tokens_details"]["cached_tokens"]),
                         (100, 64))

    def test_a_whole_response(self):
        Job.ANSWER, Job.CACHED = "Short sum.</think>\n\n40", 99
        status, r = self.responses({"input": "17 + 23?", "instructions": "Be brief."})
        self.assertEqual((status, r["object"], r["status"], r["model"]), (200, "response", "completed",
                                                                          "qwen3.8-flash-next"))
        self.assertEqual([o["type"] for o in r["output"]], ["reasoning", "message"])
        self.assertEqual(r["output"][1]["content"][0]["text"], "40")
        self.assertEqual(r["usage"]["input_tokens_details"], {"cached_tokens": 99, "cache_write_tokens": 0})


class RefusalTests(EngineRoutes):
    def test_room_left_for_an_answer(self):
        with mock.patch.object(self.e, "CTX", 1000):
            self.assertEqual(self.e.answer_room(990), 0)                   # 8 for drafts, 2 spare
            with mock.patch.object(self.e.args, "num_draft_tokens", 2):
                self.assertEqual(self.e.answer_room(980), 16)

    def test_a_prompt_that_fills_the_context_is_refused_before_it_runs(self):
        with mock.patch.object(self.e, "CTX", 100):
            for stream in (True, False):
                for route, body in ((self.chat, {"messages": [{"role": "user", "content": "long"}]}),
                                    (self.responses, {"input": "long"})):
                    status, r = route(dict(body, stream=stream))
                    self.assertEqual((status, r["error"]["code"], r["error"]["type"]),
                                     (400, "context_length_exceeded", "invalid_request_error"))
                    self.assertIn("100 tokens", r["error"]["message"])
                # the engine's other routes check too (in OpenAI's shape)
                for path, body in (("/v1/completions", {"prompt": "long"}), ("/generate", {"text": "long"})):
                    status, r = post(self.e.app, path, dict(body, stream=stream))
                    self.assertEqual((status, r["error"]["code"]), (400, "context_length_exceeded"), path)
        self.assertEqual(Job.made, [])                                         # nothing ran

    def test_a_request_that_is_not_a_chat_request_is_a_400(self):
        for route, needs in ((self.chat, "messages"), (self.responses, "input")):
            for body in ({"model": "m"}, b"{not json", b"[1, 2]"):
                status, r = route(body)
                self.assertEqual((status, r["error"]["type"], r["error"]["param"], r["error"]["code"]),
                                 (400, "invalid_request_error", None, None))
                self.assertEqual(r["error"]["message"], f"expected a JSON object with {needs}")
        status, r = self.responses({"input": [{"role": "user", "content": [{"type": "input_audio"}]}]})
        self.assertEqual((status, r["error"]["type"]), (400, "invalid_request_error"))

    def test_a_conversation_the_template_refuses_is_a_400(self):
        for route, body in ((self.chat, {"messages": [{"role": "narrator", "content": "q"}]}),
                            (self.responses, {"input": [{"role": "narrator", "content": "q"}]})):
            status, r = route(body)
            self.assertEqual((status, r["error"]["message"], r["error"]["code"]),
                             (400, "Unexpected message role.", None))
        self.assertEqual(Job.made, [])


class StopTokenTests(unittest.TestCase):
    def test_stop_tokens(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "generation_config.json").write_text(json.dumps({"eos_token_id": [3, 5]}))
            tok = types.SimpleNamespace(eos_token_id=5, single_id=lambda s: 9)
            self.assertEqual(server.stop_tokens(types.SimpleNamespace(eos_token_id_list=[1]), tok, d), [1, 3, 5, 9])


if __name__ == "__main__":
    unittest.main()
