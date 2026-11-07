"""The model server's chat layer (engines/exl3_offload/server.py) on a fake engine: streaming with and without tools,
tool calls, the context limit and the prefix-cache figures. No model, no FastAPI: both are stand-ins here."""
import argparse
import asyncio
import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engines" / "exl3_offload"))
import server  # noqa: E402

CALL = ("<tool_call>\n<function=multiply>\n<parameter=a>\n17\n</parameter>\n<parameter=b>\n23\n</parameter>\n"
        "</function>\n</tool_call>")
TOOLS = [{"type": "function", "function": {"name": "multiply", "parameters": {
    "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}}}]


class Ids:
    """A token-id tensor's shape, for a prompt of n tokens."""

    def __init__(self, n):
        self.shape = (1, n)


class JSONResponse:
    def __init__(self, content, status_code=200):
        self.content, self.status_code = content, status_code


class StreamingResponse:
    def __init__(self, body, media_type=None):
        self.body = body


def fake_fastapi():
    fastapi = types.ModuleType("fastapi")
    fastapi.Request = object
    fastapi.HTTPException = type("HTTPException", (Exception,), {})
    responses = types.ModuleType("fastapi.responses")
    responses.JSONResponse, responses.StreamingResponse = JSONResponse, StreamingResponse
    exl3 = types.ModuleType("exllamav3.constants")
    exl3.PAGE_SIZE = 256                               # exllamav3's KV cache page
    return {"fastapi": fastapi, "fastapi.responses": responses, "exllamav3": types.ModuleType("exllamav3"),
            "exllamav3.constants": exl3}


class FakeApp:
    def __init__(self):
        self.router = types.SimpleNamespace(routes=[types.SimpleNamespace(path="/v1/chat/completions"),
                                                    types.SimpleNamespace(path="/v1/completions")])
        self.added = {}

    def add_api_route(self, path, fn, methods):
        self.added[path] = fn
        self.router.routes.append(types.SimpleNamespace(path=path))

    def on_event(self, name):
        return lambda fn: fn


def fake_engine(answer, prompt_tokens=100, cached=(0, 0)):
    """The engine's module as the server uses it: an answer streamed a few characters at a time."""
    e = types.SimpleNamespace(app=FakeApp(), G={"gen": None})
    e.prompts = []

    class AsyncJob:
        def __init__(self, *args, **kwargs):
            self.job = types.SimpleNamespace(cached_pages=cached[0], cached_tokens=cached[1])

    async def run_job(ids, sp, stop_strings=None):
        e.AsyncJob(e.G["gen"], input_ids=ids)          # as the engine's run_job: one job per request
        e.sampling = sp
        for end in range(4, len(answer) + 4, 4):
            yield answer[:end], end // 4, None
        yield answer, len(answer) // 4, {"type": "stop", "reason": "eos"}

    def watch_disconnect(req):
        return asyncio.Event(), types.SimpleNamespace(cancel=lambda: None)

    def encode(prompt, encode_special_tokens=True):
        e.prompts.append(prompt)
        return Ids(prompt_tokens)

    e.AsyncJob, e.run_job, e.watch_disconnect = AsyncJob, run_job, watch_disconnect
    e.tok = types.SimpleNamespace(encode=encode, eos_token_id=None, single_id=lambda s: 7)
    e.normalize_messages = lambda m: m
    e.openai_sp = lambda body: {"temperature": body.get("temperature"), "top_k": body.get("top_k")}
    e.stops_of = lambda body: []
    e.sse = lambda obj: "data: " + json.dumps(obj) + "\n\n"
    return e


class Template:
    def render(self, messages, tools=None, thinking=True, options=None):
        return f"PROMPT thinking={thinking} tools={bool(tools)} {options or {}}"


def make_server(engine, context=1000):
    cli = argparse.Namespace(model="m", cache=context, chunk=2048, host="127.0.0.1", port=30001,
                             served_name="qwen3.8-flash-next", mtp=0)
    loaded = (None, types.SimpleNamespace(eos_token_id_list=[]), None, engine.tok, None, None, None, None)
    with mock.patch.dict(sys.modules, fake_fastapi()):
        return server.Server(cli, engine, loaded, Template(), ram=None, vram=None)


class Request:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


def chat(srv, body):
    """The chat endpoint's reply: (status, JSON body or the stream's events)."""
    async def go():
        with mock.patch.dict(sys.modules, fake_fastapi()):
            r = await srv.engine.app.added["/v1/chat/completions"](Request(body))
            if isinstance(r, JSONResponse):
                return r.status_code, r.content
            out = []
            async for chunk in r.body:
                if chunk.startswith("data: {"):
                    out.append(json.loads(chunk[6:]))
            return 200, out
    return asyncio.run(go())


def deltas(events):
    return [ev["choices"][0]["delta"] for ev in events]


class WiringTests(unittest.TestCase):
    def test_the_chat_routes_replace_the_engines_and_the_hooks_are_set(self):
        e = fake_engine("Hi.")
        make_server(e)
        paths = [r.path for r in e.app.router.routes]
        self.assertEqual(paths.count("/v1/chat/completions"), 1)
        self.assertIn("/v1/completions", paths)                              # the engine's own stays
        for p in ("/v1/responses", "/memory", "/expert_cache/resize", "/expert_cache/ram", "/expert_cache/ram/stop"):
            self.assertIn(p, e.app.added)
        self.assertEqual(e.EOS, [7])
        self.assertEqual(e.parse_tool_calls("x", None), ("x", []))
        self.assertTrue(e.chat_prompt([{"role": "user", "content": "q"}]).startswith("PROMPT"))

    def test_sampling_defaults_reach_the_engine(self):
        e = fake_engine("Hi.")
        srv = make_server(e)
        chat(srv, {"messages": [{"role": "user", "content": "q"}], "chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(e.sampling, {"temperature": 0.7, "top_k": 20})


class StreamingTests(unittest.TestCase):
    def test_prose_streams_with_tools_offered_and_the_call_comes_whole(self):
        e = fake_engine("Let me multiply those numbers for you.\n" + CALL)
        status, events = chat(make_server(e), {"messages": [{"role": "user", "content": "17 x 23?"}], "stream": True,
                                               "tools": TOOLS, "chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(status, 200)
        d = deltas(events)
        content = [x["content"] for x in d[1:] if x.get("content")]
        self.assertGreater(len(content), 3)                                    # streamed, not held to the end
        self.assertEqual("".join(content), "Let me multiply those numbers for you.\n")
        calls = [c for x in d if x.get("tool_calls") for c in x["tool_calls"]]
        self.assertEqual(len(calls), 1)
        self.assertEqual((calls[0]["index"], calls[0]["function"]["name"]), (0, "multiply"))
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"a": 17, "b": 23})
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "tool_calls")
        self.assertNotIn("<tool_call>", "".join(content))

    def test_reasoning_then_the_answer(self):
        e = fake_engine("Short sum.</think>\n\n40")
        _, events = chat(make_server(e), {"messages": [{"role": "user", "content": "17 + 23?"}], "stream": True})
        d = deltas(events)
        self.assertEqual(d[0], {"role": "assistant", "content": ""})
        self.assertEqual("".join(x.get("reasoning_content", "") for x in d), "Short sum.")
        self.assertEqual("".join(x.get("content", "") for x in d[1:]), "\n\n40")
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")

    def test_usage_reports_the_tokens_from_the_prefix_cache(self):
        e = fake_engine("Done.", prompt_tokens=10_000, cached=(39, 16))
        _, events = chat(make_server(e, context=20_000), {"messages": [{"role": "user", "content": "doc"}],
                                                          "stream": True})
        usage = events[-1]["usage"]
        self.assertEqual(usage["prompt_tokens"], 10_000)
        self.assertEqual(usage["prompt_tokens_details"], {"cached_tokens": 39 * 256 + 16})


class WholeAnswerTests(unittest.TestCase):
    def test_a_tool_call_without_streaming(self):
        e = fake_engine("Sure.\n" + CALL)
        status, body = chat(make_server(e), {"messages": [{"role": "user", "content": "q"}], "tools": TOOLS,
                                             "chat_template_kwargs": {"enable_thinking": False}})
        msg = body["choices"][0]["message"]
        self.assertEqual((status, msg["content"], body["choices"][0]["finish_reason"]), (200, "Sure.", "tool_calls"))
        self.assertEqual(json.loads(msg["tool_calls"][0]["function"]["arguments"]), {"a": 17, "b": 23})

    def test_tool_choice_none_offers_no_tools(self):
        e = fake_engine(CALL)
        _, body = chat(make_server(e), {"messages": [{"role": "user", "content": "q"}], "tools": TOOLS,
                                        "tool_choice": "none", "chat_template_kwargs": {"enable_thinking": False}})
        self.assertIn("tools=False", e.prompts[0])
        self.assertNotIn("tool_calls", body["choices"][0]["message"])

    def test_reasoning_levels_reach_the_template(self):
        e = fake_engine("ok")
        chat(make_server(e), {"messages": [{"role": "user", "content": "q"}], "reasoning_effort": "high"})
        self.assertIn("'reasoning_effort': 'xhigh'", e.prompts[0])
        chat(make_server(e), {"messages": [{"role": "user", "content": "q"}], "reasoning_effort": "none"})
        self.assertIn("thinking=False", e.prompts[1])


class ContextTests(unittest.TestCase):
    def test_room_left_for_an_answer(self):
        self.assertEqual(server.answer_room(990, 1000), 0)                   # 8 for drafts, 2 spare
        self.assertEqual(server.answer_room(980, 1000, draft_tokens=2), 16)

    def test_a_prompt_that_fills_the_context_is_refused_before_it_runs(self):
        e = fake_engine("never")
        status, body = chat(make_server(e, context=100), {"messages": [{"role": "user", "content": "long"}],
                                                          "stream": True})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "context_length_exceeded")
        self.assertIn("100 tokens", body["error"]["message"])

    def test_a_request_without_messages_is_a_400(self):
        status, body = chat(make_server(fake_engine("x")), {"model": "m"})
        self.assertEqual((status, body["error"]["type"]), (400, "invalid_request_error"))


class FiguresTests(unittest.TestCase):
    def test_cached_tokens(self):
        job = types.SimpleNamespace(job=types.SimpleNamespace(cached_pages=3, cached_tokens=5))
        self.assertEqual(server.cached_tokens(job, page_size=256), 3 * 256 + 5)
        self.assertEqual(server.cached_tokens(types.SimpleNamespace(), page_size=256), 0)

    def test_stop_tokens(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "generation_config.json").write_text(json.dumps({"eos_token_id": [3, 5]}))
            tok = types.SimpleNamespace(eos_token_id=5, single_id=lambda s: 9)
            self.assertEqual(server.stop_tokens(types.SimpleNamespace(eos_token_id_list=[1]), tok, d), [1, 3, 5, 9])



if __name__ == "__main__":
    unittest.main()
