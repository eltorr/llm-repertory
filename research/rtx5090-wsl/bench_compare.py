"""Benchmark one running model server; writes a JSON summary."""
import argparse, json, statistics, subprocess, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--startup", type=float, default=None)
a = ap.parse_args()
URL = "http://127.0.0.1:30000/v1/chat/completions"

# ---- VRAM sampler (device-wide, includes the Windows desktop) ----
vram_peak = [0]; stop = [False]
def sample():
    while not stop[0]:
        try:
            v = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
            vram_peak[0] = max(vram_peak[0], v)
        except Exception:
            pass
        time.sleep(0.5)
threading.Thread(target=sample, daemon=True).start()

def run(content, max_tokens):
    body = json.dumps({"model": "qwen3.8-flash-next", "stream": True, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time(); t_first = t_last = None; usage = None; text = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            d = (ev.get("choices") or [{}])[0].get("delta") or {}
            piece = (d.get("reasoning_content") or "") + (d.get("content") or "")
            if piece:
                now = time.time(); t_first = t_first or now; t_last = now
                text.append(d.get("content") or "")
    n = (usage or {}).get("completion_tokens", 0); p = (usage or {}).get("prompt_tokens", 0)
    ttft = (t_first - t0) if t_first else None
    dec = (n - 1) / (t_last - t_first) if t_first and t_last and t_last > t_first and n > 1 else None
    return {"prompt_tokens": p, "completion_tokens": n, "ttft_s": ttft, "decode_tps": dec,
            "wall_s": time.time() - t0, "answer": "".join(text)}

PROMPTS = [
    "Write a Python asyncio TCP port scanner with a concurrency limit, timeouts, and a CLI. Include docstrings.",
    "Explain how a B+ tree handles insertions that split nodes, with a worked example.",
    "Write a Rust function that parses an RFC 3339 timestamp without external crates, with tests.",
    "Compare Raft and Paxos for a team choosing a consensus algorithm. Be concrete.",
    "Write a bash script that rotates logs older than 7 days, compresses them, and keeps 30 archives.",
    "A train leaves at 3pm at 80 km/h; another leaves the same station at 4pm at 100 km/h. When and where does the second catch up? Show reasoning.",
    "Write a TypeScript React hook for debounced search with cancellation via AbortController.",
    "Summarise the trade-offs of the CAP theorem and give one real system per trade-off.",
]
res = {"label": a.label, "startup_s": a.startup}
res["cold_first"] = run(PROMPTS[0], 600)
for p in PROMPTS[1:4]:
    run(p, 600)                                   # warm-up, not scored
dec = [run(p, 700) for p in PROMPTS]
res["decode_runs"] = [{k: v for k, v in d.items() if k != "answer"} for d in dec]
rates = [d["decode_tps"] for d in dec if d["decode_tps"]]
if not rates:
    json.dump(res, open(a.out, "w"), indent=1); raise SystemExit(f"{a.label}: no successful decode runs (server error?)")
res["decode_tps_median"] = statistics.median(rates); res["decode_tps_min"] = min(rates); res["decode_tps_max"] = max(rates)
res["short_ttft_median_s"] = statistics.median(d["ttft_s"] for d in dec if d["ttft_s"])

filler = "".join(f"def helper_{i}(x):\n    return x * {i} + {i % 7}\n\n" for i in range(4200))
doc = filler[: len(filler)//2] + "\n# NOTE: the deployment password is 'violet-otter-42'.\n" + filler[len(filler)//2:]
q = doc + "\n\nWhat is the deployment password mentioned in the code? Answer in one line."
for key in ("long_cold", "long_cached"):
    r = run(q, 1500)
    r["prefill_tps"] = r["prompt_tokens"] / r["ttft_s"] if r["ttft_s"] else None
    r["needle_found"] = "violet-otter-42" in r["answer"]
    r.pop("answer"); res[key] = r
res["cold_first"].pop("answer")
stop[0] = True
res["vram_peak_mib"] = vram_peak[0]
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("label", "startup_s", "decode_tps_median", "vram_peak_mib")}))
