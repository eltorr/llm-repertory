"""End to end against a running Repertory, through its OpenAI API: tool calls, streaming, the prefix cache and the
context limit. Skipped unless REPERTORY_LIVE_URL names the API, e.g. REPERTORY_LIVE_URL=http://127.0.0.1:30000/v1.
The model loads on the first request, so the first test can take a minute."""
import json
import os
import time
import unittest
import urllib.error
import urllib.request

URL = os.environ.get("REPERTORY_LIVE_URL", "").rstrip("/")
MODEL = "qwen3.8-flash-next"
QUIET = {"enable_thinking": False}
TOOLS = [{"type": "function", "function": {
    "name": "celsius_to_fahrenheit", "description": "Convert a temperature from Celsius to Fahrenheit.",
    "parameters": {"type": "object", "properties": {"celsius": {"type": "number"}}, "required": ["celsius"]}}}]


def post(path, body, timeout=600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def stream(body, timeout=600):
    """(arrival time, event) for each server-sent event of a streamed chat answer."""
    req = urllib.request.Request(URL + "/chat/completions", json.dumps(dict(body, stream=True)).encode(),
                                 {"Content-Type": "application/json"})
    out = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if line.startswith("data: {"):
                out.append((time.monotonic(), json.loads(line[6:])))
    return out


def joined(events, key):
    return "".join((ev["choices"][0].get("delta") or {}).get(key) or "" for _, ev in events if ev.get("choices"))


@unittest.skipUnless(URL, "set REPERTORY_LIVE_URL to run against a live server")
class LiveTests(unittest.TestCase):
    def test_a_tool_call_streamed_then_the_answer_from_its_result(self):
        messages = [{"role": "user", "content": "Use the tool to convert 21 degrees Celsius to Fahrenheit, then "
                                                "reply with only the number."}]
        events = stream({"model": MODEL, "messages": messages, "tools": TOOLS, "temperature": 0,
                         "chat_template_kwargs": QUIET})
        calls = [c for _, ev in events for c in (ev["choices"][0].get("delta") or {}).get("tool_calls") or []]
        self.assertEqual(len(calls), 1, events)
        self.assertEqual(events[-1][1]["choices"][0]["finish_reason"], "tool_calls")
        self.assertNotIn("<tool_call>", joined(events, "content"))
        call = {k: v for k, v in calls[0].items() if k != "index"}
        self.assertEqual(json.loads(call["function"]["arguments"]), {"celsius": 21})
        messages += [{"role": "assistant", "content": joined(events, "content"), "tool_calls": [call]},
                     {"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"fahrenheit": 69.8})}]
        answer = post("/chat/completions", {"model": MODEL, "messages": messages, "tools": TOOLS, "temperature": 0,
                                            "chat_template_kwargs": QUIET})["choices"][0]["message"]
        self.assertIn("69.8", answer["content"])
        self.assertFalse(answer.get("tool_calls"))

    def test_prose_streams_while_tools_are_offered(self):
        events = stream({"model": MODEL, "temperature": 0, "max_tokens": 400, "tools": TOOLS,
                         "chat_template_kwargs": {"reasoning_effort": "low"},
                         "messages": [{"role": "user", "content": "In about 80 words, explain what causes the tides. "
                                                                  "Don't use any tools."}]})
        content = [t for t, ev in events if (ev["choices"][0].get("delta") or {}).get("content")]
        self.assertGreater(len(content), 5)
        self.assertGreater(content[-1] - content[0], 0.2)                     # spread over the answer, not at once

    def test_the_prefix_cache_serves_a_repeated_and_a_growing_conversation(self):
        log = "".join(f"Reading {i}: the tank held {i * 7 % 90} litres at {i % 24}:00.\n" for i in range(400))
        messages = [{"role": "user", "content": log + "\nReply with only the word ready."}]
        body = {"model": MODEL, "messages": messages, "max_tokens": 8, "temperature": 0, "chat_template_kwargs": QUIET}
        first = post("/chat/completions", body)
        again = post("/chat/completions", body)
        n = again["usage"]["prompt_tokens"]
        self.assertGreater(n, 3000)
        self.assertGreaterEqual(again["usage"]["prompt_tokens_details"]["cached_tokens"], n - 256)
        body["messages"] = messages + [first["choices"][0]["message"], {"role": "user", "content": "And now: done."}]
        more = post("/chat/completions", body)
        self.assertGreaterEqual(more["usage"]["prompt_tokens_details"]["cached_tokens"], n - 512)

    def test_a_prompt_over_the_context_is_a_400(self):
        with urllib.request.urlopen(URL + "/models", timeout=60) as r:
            context = json.loads(r.read())["data"][0]["max_model_len"]
        text = "word " * (context + 1000)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            post("/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": text}],
                                       "max_tokens": 1, "chat_template_kwargs": QUIET})
        self.assertEqual(caught.exception.code, 400)
        self.assertEqual(json.loads(caught.exception.read())["error"]["code"], "context_length_exceeded")

    def test_the_responses_api(self):
        r = post("/responses", {"model": MODEL, "input": "Reply with only the word pong.",
                                "reasoning": {"effort": "none"}})
        self.assertEqual(r["status"], "completed")
        self.assertIn("pong", r["output"][-1]["content"][0]["text"].lower())


if __name__ == "__main__":
    unittest.main()
