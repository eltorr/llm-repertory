"""The engine_compare.py cases through /v1/responses, checked against a Chat Completions baseline.

    python responses_compare.py BASE_URL BASELINE.json

Each case runs whole (stream false) and streamed; the answer text, reasoning and tool calls must equal the baseline's."""
import json
import sys
import time
import urllib.request

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else __file__.rsplit("/", 1)[0])
import engine_compare as ec  # noqa: E402  (its CASES only: it runs requests only as a script)

BASE = sys.argv[1].rstrip("/")
base = {r["name"]: r for r in json.load(open(sys.argv[2], encoding="utf-8"))["results"]}


def tools_flat(tools):
    return [dict(t["function"], type="function") for t in tools or []]


def post(body):
    req = urllib.request.Request(BASE + "/v1/responses", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=1800)


def parse_whole(d):
    text, reasoning, calls = "", "", []
    for item in d.get("output") or []:
        if item["type"] == "message":
            text += "".join(c.get("text", "") for c in item.get("content") or [])
        elif item["type"] == "reasoning":
            reasoning += "".join(c.get("text", "") for c in item.get("content") or [])
        elif item["type"] == "function_call":
            calls.append((item.get("name"), item.get("arguments")))
    return text, reasoning, calls, d.get("usage")


def parse_stream(r):
    text, reasoning, calls, usage = "", "", [], None
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        ev = json.loads(line[5:])
        t = ev.get("type")
        if t == "response.output_text.delta":
            text += ev["delta"]
        elif t == "response.reasoning_text.delta":
            reasoning += ev["delta"]
        elif t == "response.output_item.done" and ev["item"]["type"] == "function_call":
            calls.append((ev["item"].get("name"), ev["item"].get("arguments")))
        elif t == "response.completed":
            usage = ev["response"].get("usage")
    return text, reasoning, calls, usage


ok_all = True
for name, chat_body in ec.CASES:
    b = base[name]
    want = (b["content"].strip(), b["reasoning"].strip(),
            [((c.get("function") or {}).get("name"), (c.get("function") or {}).get("arguments")) for c in b["tool_calls"]])
    body = {"model": "qwen3.8-flash-next", "input": chat_body["messages"], "temperature": 0,
            "chat_template_kwargs": chat_body.get("chat_template_kwargs", {"enable_thinking": False})}
    if chat_body.get("tools"):
        body["tools"] = tools_flat(chat_body["tools"])
    for mode in ("whole", "stream"):
        t0 = time.time()
        if mode == "whole":
            got = parse_whole(json.load(post(body)))
        else:
            got = parse_stream(post(dict(body, stream=True)))
        same = (got[0].strip(), got[1].strip(), got[2]) == want
        ok_all &= same
        cached = ((got[3] or {}).get("input_tokens_details") or {}).get("cached_tokens")
        print(f"{name:9s} {mode:6s} {'IDENTICAL' if same else 'DIFFERENT'}  {time.time() - t0:5.1f} s  "
              f"usage {(got[3] or {}).get('input_tokens')} in / {(got[3] or {}).get('output_tokens')} out, "
              f"cached {cached}", flush=True)
        if not same:
            print("   got:", repr(got[:3])[:300]); print("  want:", repr(want)[:300])
print("ALL IDENTICAL" if ok_all else "SOME DIFFER")
