"""Send the same greedy requests to an OpenAI-compatible server and record answers and timings.

    python engine_compare.py BASE_URL OUT.json [--key KEY_ENV]

Answers run to their natural end (no max_tokens). Thinking is off except in the reasoning case."""
import json
import os
import sys
import time
import urllib.request

BASE, OUT = "", ""
HEADERS = {"Content-Type": "application/json"}

WEATHER = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def document():
    lines = []
    for i in range(2400):
        t = 11 + (i * 7) % 23
        w = (i * 13) % 41
        lines.append(f"Station log {i:04d}: temperature {t} C, wind {w} km/h, pressure {990 + (i * 3) % 40} hPa.")
    lines[1717] = "Station log 1717: the backup generator was serviced by technician Ilsa Varga; code word PERIWINKLE."
    return "\n".join(lines)


CASES = [
    ("short", {"messages": [{"role": "user", "content": "Explain in three sentences how a heat pump moves heat."}]}),
    ("code", {"messages": [{"role": "user", "content":
              "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."}]}),
    ("tool", {"messages": [{"role": "user", "content": "What's the weather in Lisbon right now?"}], "tools": WEATHER}),
    ("document", {"messages": [{"role": "user", "content": document() +
                  "\n\nWho serviced the backup generator, and what is the code word? Answer in one sentence."}]}),
    ("reasoning", {"messages": [{"role": "user", "content": "Is 221 a prime number? Answer briefly."}],
                   "chat_template_kwargs": {"enable_thinking": True}}),
]


def run(name, body):
    body = dict(body, model="qwen3.8-flash-next", temperature=0, stream=True,
                stream_options={"include_usage": True})
    body.setdefault("chat_template_kwargs", {"enable_thinking": False})
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers=HEADERS)
    t0, first, content, reasoning, calls, usage, finish = time.time(), None, "", "", [], None, None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece and first is None:
                    first = time.time()
                content += d.get("content") or ""
                reasoning += d.get("reasoning_content") or ""
                calls += d.get("tool_calls") or []
                finish = ch.get("finish_reason") or finish
    t1 = time.time()
    n = (usage or {}).get("completion_tokens") or 0
    out = {"name": name, "content": content, "reasoning": reasoning, "tool_calls": calls, "finish": finish,
           "usage": usage, "ttft_s": round((first or t1) - t0, 3), "total_s": round(t1 - t0, 3),
           "decode_tps": round(n / (t1 - first), 1) if first and n > 8 and t1 > first else None}
    print(f"{name}: {n} tokens, first token {out['ttft_s']} s, decode {out['decode_tps']} tok/s, finish {finish}",
          flush=True)
    return out


if __name__ == "__main__":
    BASE, OUT = sys.argv[1].rstrip("/"), sys.argv[2]
    if "--key" in sys.argv:
        HEADERS["Authorization"] = "Bearer " + os.environ[sys.argv[sys.argv.index("--key") + 1]]
    results = [run(n, b) for n, b in CASES]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"base": BASE, "when": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results}, f, indent=1)
    print("saved", OUT)
