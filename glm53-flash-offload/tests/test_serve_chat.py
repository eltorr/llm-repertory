"""glm53/serve.py's /v1/responses and /v1/chat/completions without a model: exllamav3 is stubbed, the globals a wrapper
sets are set, and run_job / chat_prompt / parse_tool_calls / startup are replaced the way a wrapper replaces them.
Checks the SSE streams (reasoning, content streamed early, tool calls never as content), the non-streamed answers,
and for both APIs: the context check, the prefix-cache figures in usage (the real run_job, on a fake job) and the 400s
for a request that can't run. Then what replaying a turn as token ids uses: /prepare, /generate's output_ids and
teacher forcing (on a fake model). Needs torch (CPU) and fastapi; no GPU, no CUDA."""
import argparse
import asyncio
import importlib.util
import json
import os
import sys
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALL = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
QWEN_ON = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
QWEN_OFF = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
TOOLS = [{"type": "function", "function": {"name": "get_weather"}}]
NEEDS = ("torch", "fastapi", "uvicorn")


def stubs():
    ex = types.ModuleType("exllamav3"); ex.__path__ = []
    mi = types.ModuleType("exllamav3.model_init")
    gen = types.ModuleType("exllamav3.generator"); gen.__path__ = []
    gen.AsyncGenerator = gen.AsyncJob = object
    smp = types.ModuleType("exllamav3.generator.sampler")
    smp.ArgmaxSampler = smp.ComboSampler = dict        # the settings run_job picked, as given
    ex.model_init, ex.generator, gen.sampler = mi, gen, smp
    return {"exllamav3": ex, "exllamav3.model_init": mi, "exllamav3.generator": gen, "exllamav3.generator.sampler": smp}


class FakeReq:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return json.loads(self.body) if isinstance(self.body, bytes) else self.body

    async def is_disconnected(self):
        return False


class Tok:
    def encode(self, text, encode_special_tokens=True):
        import torch
        return torch.zeros((1, max(1, len(text) // 4)), dtype=torch.long)


class Job:
    """exllamav3's AsyncJob as run_job uses it: TEXT four characters at a time, then the end with its cache hits."""
    TEXT, CACHED = "", 0

    def __init__(self, generator, **kw):
        self.kw = kw

    async def __aiter__(self):
        for i in range(0, len(self.TEXT), 4):
            yield {"stage": "streaming", "text": self.TEXT[i:i + 4], "eos": False}
        yield {"stage": "streaming", "text": "", "eos": True, "eos_reason": "stop_token",
               "new_tokens": -(-len(self.TEXT) // 4), "cached_tokens": self.CACHED}

    async def cancel(self):
        pass


def parse(content, tools):
    if "<tool_call>" not in content:
        return content, []
    return content.split("<tool_call>")[0].strip(), [{"id": "c1", "type": "function",
                                                      "function": {"name": "get_weather", "arguments": "{}"}}]


def sse_data(chunks):
    """The JSON objects of an event stream: chat's data lines, or the Responses events."""
    return [json.loads(line[6:]) for c in chunks for line in c.split("\n") if line.startswith("data: {")]


HAVE = all(importlib.util.find_spec(m) for m in NEEDS)
ENGINE = {}


def setUpModule():
    """serve.py imported once, on the stubs, with the globals load() would set."""
    if not HAVE:
        return
    import fastapi, torch, uvicorn  # noqa: F401  (imported before the stubs: they outlive the patch)
    ENGINE["patch"] = mock.patch.dict(sys.modules, stubs())
    ENGINE["patch"].start()
    sys.path.insert(0, os.path.join(ROOT, "glm53"))
    import serve
    serve.tok, serve.args = Tok(), argparse.Namespace(served_name="m", num_draft_tokens=0)
    serve.CTX, serve.EOS = 4096, [2]
    ENGINE["serve"] = serve


def tearDownModule():
    if "patch" in ENGINE:
        ENGINE.pop("patch").stop()


class Serve(unittest.TestCase):
    def setUp(self):
        self.serve = ENGINE["serve"]

    def call(self, route, body, prompt=QWEN_ON, run_job=None):
        """(HTTP status, JSON body) of a route's answer, or (200, the stream's chunks). run_job None: the real one, on
        Job."""
        s = self.serve
        hooks = {"parse_tool_calls": parse, "chat_prompt": lambda messages, thinking, tools, kw: prompt,
                 "AsyncJob": Job}
        if run_job is not None:
            hooks["run_job"] = run_job
        with mock.patch.multiple(s, **hooks):
            resp = asyncio.run(getattr(s, route)(FakeReq(body)))
            if not hasattr(resp, "body_iterator"):
                return resp.status_code, json.loads(resp.body)

            async def collect():
                return [c if isinstance(c, str) else c.decode() async for c in resp.body_iterator]
            return 200, asyncio.run(collect())


@unittest.skipUnless(HAVE, "needs torch, fastapi, uvicorn")
class Chat(Serve):
    def answer(self, text, prompt, stream, tools=True, step=3):
        """Run the chat endpoint on a fake generation that yields `text` `step` characters at a time."""
        async def run_job(ids, sp, stop_strings=None):
            for i in range(step, len(text) + step, step):
                yield text[:i], i // step, ({"type": "stop", "reason": "eos"} if i >= len(text) else None)
        body = {"messages": [{"role": "user", "content": "hi"}], "stream": stream}
        if tools:
            body["tools"] = TOOLS
        _, out = self.call("chat", body, prompt, run_job)
        if not stream:
            return out
        self.assertEqual(out[-1], "data: [DONE]\n\n")
        return sse_data(out)

    def test_stream_tool_call(self):
        ev = self.answer("Need the weather.\n</think>\n\nLet me check." + CALL, QWEN_ON, True)
        deltas = [e["choices"][0]["delta"] for e in ev]
        reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
        content = "".join(d.get("content", "") for d in deltas)
        self.assertEqual(reasoning, "Need the weather.\n")
        self.assertEqual(content, "\n\nLet me check.")                 # as the model wrote it after the think block
        self.assertNotIn("<tool", content)
        calls = [d["tool_calls"] for d in deltas if "tool_calls" in d]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0]["function"]["name"], "get_weather")
        self.assertEqual(ev[-1]["choices"][0]["finish_reason"], "tool_calls")
        # content was streamed before the answer ended, not held to the end
        first_content = next(i for i, d in enumerate(deltas) if d.get("content"))
        self.assertLess(first_content, len(deltas) - 3)

    def test_stream_thinking_off(self):
        ev = self.answer("Hello there.", QWEN_OFF, True, tools=False)
        deltas = [e["choices"][0]["delta"] for e in ev]
        self.assertEqual("".join(d.get("content", "") for d in deltas), "Hello there.")
        self.assertFalse(any("reasoning_content" in d for d in deltas))
        self.assertEqual(ev[-1]["choices"][0]["finish_reason"], "stop")

    def test_non_stream(self):
        r = self.answer("Need the weather.\n</think>\n\nLet me check." + CALL, QWEN_ON, False)
        msg = r["choices"][0]["message"]
        self.assertEqual((msg["reasoning_content"], msg["content"]), ("Need the weather.", "Let me check."))
        self.assertEqual(r["choices"][0]["finish_reason"], "tool_calls")
        r = self.answer("never closed", QWEN_ON, False, tools=False)
        self.assertEqual((r["choices"][0]["message"]["reasoning_content"], r["choices"][0]["message"]["content"]),
                         ("never closed", ""))
        r = self.answer("plain", QWEN_OFF, False, tools=False)
        self.assertEqual((r["choices"][0]["message"]["reasoning_content"], r["choices"][0]["message"]["content"]),
                         (None, "plain"))

    def test_startup_can_be_replaced(self):
        seen = []

        async def mine():
            seen.append(1)
        with mock.patch.object(self.serve, "startup", mine):
            asyncio.run(self.serve._on_startup())
        self.assertEqual(seen, [1])


@unittest.skipUnless(HAVE, "needs torch, fastapi, uvicorn")
class BothApis(Serve):
    """What /v1/responses and /v1/chat/completions share: one turn, its checks and its usage."""
    CHAT = {"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS}
    RESPONSES = {"input": "hi", "tools": [{"type": "function", "name": "get_weather"}]}

    def setUp(self):
        super().setUp()
        Job.TEXT, Job.CACHED = "Need the weather.\n</think>\n\nLet me check." + CALL, 0

    def test_responses_stream_and_whole(self):
        Job.CACHED = 512
        _, chunks = self.call("responses", dict(self.RESPONSES, stream=True))
        ev = sse_data(chunks)
        self.assertEqual(ev[0]["type"], "response.created")
        self.assertEqual("".join(e["delta"] for e in ev if e["type"] == "response.output_text.delta"),
                         "\n\nLet me check.")
        done = ev[-1]
        self.assertEqual(done["type"], "response.completed")
        self.assertEqual([o["type"] for o in done["response"]["output"]], ["reasoning", "message", "function_call"])
        self.assertEqual(done["response"]["usage"]["input_tokens_details"]["cached_tokens"], 512)
        status, r = self.call("responses", self.RESPONSES)
        self.assertEqual((status, r["status"], r["output"][0]["content"][0]["text"]),
                         (200, "completed", "Need the weather."))
        self.assertEqual(r["usage"]["input_tokens_details"], {"cached_tokens": 512, "cache_write_tokens": 0})

    def test_chat_usage_has_the_cached_tokens(self):
        Job.CACHED = 300
        _, r = self.call("chat", self.CHAT)
        self.assertEqual(r["usage"]["prompt_tokens_details"], {"cached_tokens": 300})
        self.assertEqual(r["choices"][0]["finish_reason"], "tool_calls")
        _, chunks = self.call("chat", dict(self.CHAT, stream=True))
        usage = sse_data(chunks)[-1]["usage"]
        self.assertEqual(usage["prompt_tokens_details"], {"cached_tokens": 300})
        self.assertEqual(usage["total_tokens"], usage["prompt_tokens"] + usage["completion_tokens"])

    def test_a_prompt_that_fills_the_context_is_a_400_before_it_runs(self):
        made = []
        with mock.patch.object(self.serve, "CTX", 20), mock.patch.object(Job, "__init__",
                                                                         lambda *a, **k: made.append(1)):
            for route, body in (("chat", self.CHAT), ("responses", self.RESPONSES)):
                for stream in (False, True):
                    status, r = self.call(route, dict(body, stream=stream))
                    self.assertEqual((status, r["error"]["code"]), (400, "context_length_exceeded"), route)
                    self.assertIn("this server's context holds 20", r["error"]["message"])
        self.assertEqual(made, [])
        self.assertEqual(self.serve.answer_room(4096 - 10), 0)            # 8 for drafts, 2 spare
        with mock.patch.object(self.serve.args, "num_draft_tokens", 2):
            self.assertEqual(self.serve.answer_room(4096 - 20), 16)

    def test_run_job_checks_for_every_route(self):
        s = self.serve
        with mock.patch.object(s, "CTX", 20):
            for body in ({"prompt": "x" * 200}, {"prompt": "x" * 200, "stream": True}):
                with self.assertRaises(s.ContextFull):
                    asyncio.run(s.completions(FakeReq(body)))

            async def run():
                async for _ in s.run_job(s.tok.encode("x" * 200), {}):
                    pass
            with self.assertRaises(s.ContextFull):
                asyncio.run(run())
        r = asyncio.run(s.app.exception_handlers[s.ContextFull](None, s.ContextFull(200, 20)))   # what the app answers
        self.assertEqual((r.status_code, json.loads(r.body)["error"]["code"]), (400, "context_length_exceeded"))

    def test_a_body_that_is_not_a_request_is_a_400(self):
        for route, needs in (("chat", "messages"), ("responses", "input")):
            for body in ({"model": "m"}, b"{not json", [1, 2]):
                status, r = self.call(route, body)
                self.assertEqual((status, r["error"]["type"], r["error"]["param"], r["error"]["code"]),
                                 (400, "invalid_request_error", None, None))
                self.assertEqual(r["error"]["message"], f"expected a JSON object with {needs}")
        status, r = self.call("responses", {"input": [{"role": "user", "content": [{"type": "input_image"}]}]})
        self.assertEqual(status, 400)
        self.assertIn("text only", r["error"]["message"])

    def test_a_template_refusal_is_a_400(self):
        class Refused(ValueError):
            public = True           # a wrapper's refusal written for the client

        def refuse(messages, thinking, tools, kw):
            raise Refused("System message must be at the beginning.")
        with mock.patch.multiple(self.serve, chat_prompt=refuse):
            for route, body in (("chat", self.CHAT), ("responses", self.RESPONSES)):
                r = asyncio.run(getattr(self.serve, route)(FakeReq(body)))
                err = json.loads(r.body)["error"]
                self.assertEqual((r.status_code, err["message"]), (400, "System message must be at the beginning."))

    def test_any_other_error_while_preparing_is_a_generic_400(self):
        def broken(messages, thinking, tools, kw):
            raise ValueError("invalid literal for int() with base 10: '/srv/models/secret-path'")
        with mock.patch.multiple(self.serve, chat_prompt=broken):
            for route, body in (("chat", self.CHAT), ("responses", self.RESPONSES)):
                r = asyncio.run(getattr(self.serve, route)(FakeReq(body)))
                err = json.loads(r.body)["error"]
                self.assertEqual((r.status_code, err["message"]), (400, "invalid request"))   # its text: the log only


class Chars:
    """A tokenizer with a token per character, so a prompt's ids say what it was."""
    def encode(self, text, encode_special_tokens=True):
        import torch
        return torch.tensor([[ord(c) for c in text]], dtype=torch.long)


class TokenJob(Job):
    """As Job, with exllamav3's token ids: one a step, and at the end the stop token, held back."""
    IDS, STOP = (5, 6, 7), 2

    async def __aiter__(self):
        import torch
        for t in self.IDS:
            yield {"stage": "streaming", "text": "x", "token_ids": torch.tensor([[t]]), "eos": False}
        yield {"stage": "streaming", "eos": True, "eos_reason": "stop_token", "new_tokens": len(self.IDS) + 1,
               "held": {"token_ids": torch.tensor([[self.STOP]])}}


@unittest.skipUnless(HAVE, "needs torch, fastapi, uvicorn")
class Replay(Serve):
    """A turn replayed as token ids: /prepare gives the prompt's ids as the chat routes build them, /generate the
    answer's ids, and teacher forcing each answer token's logprob."""
    CHAT = {"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS,
            "chat_template_kwargs": {"enable_thinking": False}}

    @staticmethod
    def prompt(messages, thinking, tools, kw):
        return f"{thinking}|{bool(tools)}|{messages[-1]['content']}"

    def ids(self, r):
        self.assertEqual(r.status_code, 200)
        return json.loads(r.body)["ids"]

    def test_prepare_gives_the_ids_the_chat_routes_run(self):
        s, ran = self.serve, []

        async def run_job(ids, sp, stop_strings=None):
            ran.append(ids[0].tolist())
            yield "ok", 1, {"type": "stop", "reason": "eos"}
        with mock.patch.multiple(s, tok=Chars(), chat_prompt=self.prompt, run_job=run_job):
            for body in (self.CHAT, {"messages": [{"role": "user", "content": "hi"}]}):
                ids = self.ids(asyncio.run(s.prepare(FakeReq(body))))
                asyncio.run(s.chat(FakeReq(body)))
                self.assertEqual(ids, ran[-1])
            self.assertEqual("".join(map(chr, self.ids(asyncio.run(s.prepare(FakeReq(self.CHAT)))))), "False|True|hi")
            # a Responses body: its chat-shaped turn, as /v1/responses runs it
            body = {"input": "hi", "tools": [{"type": "function", "name": "get_weather"}]}
            ids = self.ids(asyncio.run(s.prepare(FakeReq(body))))
            asyncio.run(s.responses(FakeReq(body)))
            self.assertEqual((ids, "".join(map(chr, ids))), (ran[-1], "True|True|hi"))

    def test_prepare_refuses_what_the_chat_routes_refuse(self):
        s = self.serve
        for body in ({"model": "m"}, b"{not json", [1, 2]):
            r = asyncio.run(s.prepare(FakeReq(body)))
            self.assertEqual((r.status_code, json.loads(r.body)["error"]["message"]),
                             (400, "expected a JSON object with messages or input"))
        with mock.patch.object(s, "CTX", 20), mock.patch.object(s, "chat_prompt", lambda *a: "x" * 400):
            r = asyncio.run(s.prepare(FakeReq(self.CHAT)))
            self.assertEqual((r.status_code, json.loads(r.body)["error"]["code"]), (400, "context_length_exceeded"))

    def test_a_whole_generate_answer_has_its_token_ids(self):
        s = self.serve
        with mock.patch.object(s, "AsyncJob", TokenJob):
            r = asyncio.run(s.generate(FakeReq({"input_ids": [1, 2, 3], "sampling_params": {"temperature": 0}})))
        out = json.loads(r.body)
        self.assertEqual(out["output_ids"], [5, 6, 7, 2])                   # the stop token included
        self.assertEqual((out["text"], out["meta_info"]["completion_tokens"]), ("xxx", 4))

    def test_teacher_forcing_gives_each_tokens_own_logprob_from_the_rows_it_needs(self):
        import torch
        s = self.serve
        ids, start, top = [3, 9, 1, 4, 8, 2, 6], 4, 5
        table = torch.randn(len(ids), 50, generator=torch.Generator().manual_seed(0))   # the logits at each position
        seen = []

        class Model:
            every_row = False                                                 # a forward that ignores the option

            def forward(self, x, params):
                seen.append(dict(params))
                return (table if self.every_row else table[-params["last_tokens_only"]:]).unsqueeze(0)
        body = {"input_ids": ids, "return_logprob": True, "logprob_start_len": start, "top_logprobs_num": top}
        model = Model()
        with mock.patch.object(s, "model", model, create=True), \
                mock.patch.dict(s.G, gen=types.SimpleNamespace(jobs=[])):
            meta = json.loads(asyncio.run(s.generate(FakeReq(body))).body)["meta_info"]
            model.every_row = True
            self.assertEqual(json.loads(asyncio.run(s.generate(FakeReq(body))).body)["meta_info"], meta)
        self.assertEqual(seen[0]["last_tokens_only"], len(ids) - (start - 1))   # rows 3.. only reach the head
        lp = torch.log_softmax(table, -1)
        self.assertEqual(len(meta["input_token_logprobs"]), len(ids) - start)
        for j, t in enumerate(range(start, len(ids))):                        # token t, predicted at row t - 1
            got_lp, got_id, _ = meta["input_token_logprobs"][j]
            self.assertEqual(got_id, ids[t])
            self.assertAlmostEqual(got_lp, float(lp[t - 1, ids[t]]), places=5)
            v, ix = lp[t - 1].topk(top)
            self.assertEqual([e[1] for e in meta["input_top_logprobs"][j]], ix.tolist())
            self.assertAlmostEqual(meta["input_top_logprobs"][j][0][0], float(v[0]), places=5)
        self.assertEqual([e[1] for e in meta["output_top_logprobs"][0]], lp[-1].topk(top)[1].tolist())


if __name__ == "__main__":
    unittest.main()
