"""glm53/serve.py's /v1/chat/completions without a model: exllamav3 is stubbed, the globals a wrapper sets are set,
and run_job / chat_prompt / parse_tool_calls / startup are replaced the way a wrapper replaces them. Checks the SSE
stream (reasoning, content streamed early, tool calls never as content) and the non-streamed answer. Needs torch (CPU)
and fastapi; no GPU, no CUDA."""
import argparse
import asyncio
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


def stubs():
    ex = types.ModuleType("exllamav3"); ex.__path__ = []
    mi = types.ModuleType("exllamav3.model_init")
    gen = types.ModuleType("exllamav3.generator"); gen.__path__ = []
    gen.AsyncGenerator = gen.AsyncJob = object
    smp = types.ModuleType("exllamav3.generator.sampler")
    smp.ArgmaxSampler = smp.ComboSampler = object
    ex.model_init, ex.generator, gen.sampler = mi, gen, smp
    return {"exllamav3": ex, "exllamav3.model_init": mi, "exllamav3.generator": gen, "exllamav3.generator.sampler": smp}


class FakeReq:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body

    async def is_disconnected(self):
        return False


class Tok:
    def encode(self, text, encode_special_tokens=True):
        import torch
        return torch.zeros((1, max(1, len(text) // 4)), dtype=torch.long)


@unittest.skipUnless(all(__import__("importlib").util.find_spec(m) for m in ("torch", "fastapi", "uvicorn")),
                     "needs torch, fastapi, uvicorn")
class Chat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patch = mock.patch.dict(sys.modules, stubs())
        cls.patch.start()
        sys.path.insert(0, os.path.join(ROOT, "glm53"))
        import serve
        cls.serve = serve
        serve.tok, serve.args = Tok(), argparse.Namespace(served_name="m")

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()

    def answer(self, text, prompt, stream, tools=True, step=3):
        """Run the endpoint on a fake generation that yields `text` `step` characters at a time."""
        s = self.serve

        async def run_job(ids, sp, stop_strings=None):
            for i in range(step, len(text) + step, step):
                yield text[:i], i // step, ({"type": "stop", "reason": "eos"} if i >= len(text) else None)

        def parse(content, tools):
            if "<tool_call>" not in content:
                return content, []
            return content.split("<tool_call>")[0].strip(), [{"id": "c1", "type": "function",
                                                              "function": {"name": "get_weather", "arguments": "{}"}}]
        body = {"messages": [{"role": "user", "content": "hi"}], "stream": stream}
        if tools:
            body["tools"] = [{"type": "function", "function": {"name": "get_weather"}}]
        with mock.patch.multiple(s, run_job=run_job, parse_tool_calls=parse,
                                 chat_prompt=lambda messages, thinking, tools, kw: prompt):
            resp = asyncio.run(s.chat(FakeReq(body)))
            if not stream:
                return json.loads(resp.body)

            async def collect():
                return [c if isinstance(c, str) else c.decode() async for c in resp.body_iterator]
            chunks = asyncio.run(collect())
        events = [json.loads(c[6:]) for c in chunks if c.startswith("data: {")]
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        return events

    def test_stream_tool_call(self):
        ev = self.answer("Need the weather.\n</think>\n\nLet me check." + CALL, QWEN_ON, True)
        deltas = [e["choices"][0]["delta"] for e in ev]
        reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
        content = "".join(d.get("content", "") for d in deltas)
        self.assertEqual(reasoning, "Need the weather.\n")
        self.assertEqual(content, "Let me check.")
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


if __name__ == "__main__":
    unittest.main()
