"""Benchmarks: start the engine with a given configuration, run a fixed workload against it, and record the results.

One tool for the dashboard's Benchmark tab, `repertory bench`, and the runs behind the README's tables (research/).
It takes the GPU: the manager stops the model first, refuses requests meanwhile, and puts the model back afterwards.

Runs are QUANT:VRAM_GB|card:CONTEXT[:nostage][:ram=GIB][:mtp=N][:cpu[=CPUS]][:prefetch], or a preset:
  current  the current settings on the whole card, as the full profile runs them
  lean     the current settings in the lean profile's VRAM and RAM
  quants   each quant of the configured model (Qwen3.8-Flash-Next: 3.05 and 4.05 bpw) at the current context, on
           the whole card
  matrix   8-32 GB cards emulated on this one: the README's measured table, Qwen's two quants (needs a 32 GB card and
           64 GB of RAM; every run gets a 38 GiB RAM cache, so runs differ only in VRAM)
VRAM_GB holds the engine to that much VRAM, as on a smaller card after the desktop's share: QWEN_VRAM_LIMIT_BYTES sizes
the expert cache, and QWEN_VRAM_GUARD_BYTES stops the server past it, as running out of memory would on that card. A
smaller card's bandwidth can't be emulated, so real smaller cards are slower. ":nostage" forces staging-free prefill
(sizing.nostage_chunk: 1,536-token chunks for Qwen, 1,792 for GLM); ":ram=GIB" sets the RAM cache (default: what the
full profile would get now). QUANT is a catalog model's (3.05, 4.05, glm-3.05).
Engine features, off unless named: ":mtp=N" lets the MTP layer draft N tokens per step; ":cpu" computes RAM-resident
experts on the CPU in decode (3-bit experts only: the engine refuses others), on the CPUs in ":cpu=2-15" (default the
engine's); ":prefetch" starts NVMe reads for the next layer's predicted experts early.

The workload (PROTOCOL "repertory-1"): thinking off, greedy decoding, and every answer to its natural end (output is
never capped). A warm-up first (a one-word answer, then one unscored trial, so cold caches don't count), then trials
of one answer (how a heat pump works, in three short paragraphs: a few hundred tokens) and documents (a weather
station log, read and answered in one sentence). Each trial's text differs from the start, so nothing comes from the
prefix cache.
  quick     1 trial: the answer, documents of 1K and 8K tokens                                   ~2 min a run
  standard  3 trials: the answer, documents of 1K, 8K, 32K and 64K tokens; then the retrieval
            test: three locker combinations at 15%, 50% and 85% of a log of up to 100K tokens    ~3-6 min a run
  reference curation's regression requests (repertory/curate.py): after the warm-up, a short answer, code, a tool
            call, a 73K-token document with one fact to find, and a short answer with thinking on, each answer
            recorded whole (text, reasoning, tool calls, finish reason, token counts); then each short answer
            again as token ids with its logprobs, and the model's reference answers replayed     ~2 min a run
Documents that don't fit the run's context are skipped; the retrieval document is sized to it. Runs before Repertory
used the prompts and documents of the benchmark.py in MaxDevv/lean-qwenflash, which this project began as a fork of;
results don't compare 1:1 with those (or with his), and runs are only compared with runs of the same protocol.

The PC comes first. When a game or another app needs the GPU (a watched app, another app's GPU load, or its VRAM
crowding the card) or RAM nearly runs out, the run steps aside: its server stops, the benchmark waits until the PC has
been free for `recover_after_s`, then does that run again from the start. Finished runs are kept. A benchmark cut short
(cancelled, the manager restarted, the PC rebooted) continues with `repertory bench --resume` or the tab's Resume
button: the runs left go into the same folder.

Results: one JSON per run in the settings folder (bench/<date-time>/), saved after every step, with the machine and
its stack (GPU, driver, CUDA, PyTorch, commit) and the change since the last run of the same configuration. The
folder's _plan.json lists the benchmark's runs, for resuming.
"""
import asyncio
import collections
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import catalog, config
from .config import GB, GIB, log
from .httpio import BACKEND_KEY_HEADER, http_ok

from engines.exl3_offload import sizing  # noqa: E402

BENCH_DIR = config.STATE / "bench"
RUN_DIR = re.compile(r"^\d{8}-\d{6}(-\d+)?$")
MAX_RUNS = 32
PRESETS = {"current": "This machine's settings, whole card", "lean": "The lean profile",
           "quants": "Each quant of the configured model, whole card",
           "matrix": "8-32 GB cards emulated on this one (Qwen's quants)"}
# the README's measured table: card sizes after the desktop's share, each at the largest context that keeps staging.
# Written for Qwen's two quants (it reproduces research/vram-scaling-torch-2.15), so hard-coded; it stays Qwen's.
# Every run gets the same RAM cache, as the earlier matrix (research/vram-scaling): sized from the RAM free when a run
# starts, it varied with how much of the previous run's memory WSL had handed back yet (8-25 GiB a run).
MATRIX_RAM = 38
MATRIX = tuple(f"{s}:ram={MATRIX_RAM}" for s in (
    "3.05:30:262144", "3.05:22:262144", "3.05:14.5:262144", "3.05:10.5:65536", "3.05:10.5:131072:nostage",
    "3.05:8:16384:nostage", "3.05:7:16384:nostage", "4.05:30:262144", "4.05:22:262144", "4.05:14.5:262144",
    "4.05:10.5:32768:nostage"))
WORKLOADS = {
    "quick": {"trials": 1, "documents": (1024, 8192), "retrieval": False},
    "standard": {"trials": 3, "documents": (1024, 8192, 32768, 65536), "retrieval": True},
    # full -> lean -> full without a restart, then quick cycles (Bench.resize_steps); whole-card runs only. Then the
    # same with the RAM cache shrunk to lean's and grown back, and Windows' free RAM this many seconds after the shrink
    "resize": {"cycles": 3, "document": 32768, "ram_cycles": 2, "ram_timeline_s": (0, 5, 15, 30, 60)},
    # the RAM cache's background work where it could go wrong (Bench.stress_steps); whole-card runs only. A request
    # while it grows or frees memory must have its first token within latency_max_s (one from a cold cache: ~2 s)
    "resize-stress": {"document": 32768, "toggles": 4, "cancel_after_s": 1.0, "stop_after_s": 1.0,
                      "latency_max_s": 4.0},
    # curation's regression requests (REFERENCE_CASES), which curate.py compares with the model's reference
    "reference": {"protocol": "reference-1"},
}
VERIFY_BAD = ("vram_bad", "ram_bad", "table_bad", "home_bad_resident", "home_bad_nonresident", "dev_bad_rows")
# The workload's text. Changing any of it is a new protocol: runs are compared only with runs of the same one.
PROTOCOL = "repertory-1"            # recorded with each run
WARMUP = ("Which planet is closest to the Sun? Answer with one word.",
          "Warm-up: Explain how a heat pump moves heat from a cold place to a warm one, in three short paragraphs.")
ANSWER = "Trial {t}: Explain how a heat pump moves heat from a cold place to a warm one, in three short paragraphs."
DOC_HEAD = "Log {t}-{size}: the weather station records below are for reference only."
DOC_TAIL = "In one sentence, what does a weather station measure?"
STATIONS = ("Northfield", "Lakeside", "Ridge", "Harbor Point", "Old Mill", "Pine Hollow", "Saltmarsh")
WINDS = ("north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west")
SKIES = ("clear", "overcast", "light rain", "fog", "scattered cloud", "drizzle", "haze")
LOCKERS = ((0.15, "HARBOR", "402917"), (0.5, "MEADOW", "538064"), (0.85, "SUMMIT", "176285"))
RETRIEVAL_HEAD = "Read this station log closely; three of its entries give locker combinations."
RETRIEVAL_TAIL = ("Which combinations open the lockers HARBOR, MEADOW and SUMMIT? Reply with only the three "
                  "combinations, in that order.")
RETRIEVAL_MAX = 100000              # the longest prompt the README's tables report, well short of 256K
SETTLE_S = 3                        # after stopping a server, before reading free VRAM
WATCH_S = 1.0                       # how often a run checks whether the PC needs the GPU or RAM back
YIELD_AFTER_S = 10                  # how long the PC must need them before the run steps aside
MAX_YIELDS = 3                      # a run interrupted more often is recorded as interrupted (resume redoes it)
REPLACED = "another model server replaced this run's on its port: what was measured may be that server's"
RAM_FLOOR_GB = 1.0                  # free RAM below this: the PC is about to run out
PLAN = "_plan.json"                 # the benchmark's runs and workload, for resuming (not a result)

# Curation's regression requests: the cases of research/engine-parity/engine_compare.py, byte for byte (so its results
# compare with these). Greedy, streamed, each answer to its natural end (never max_tokens); thinking off except in the
# last. Changing any of them is a new protocol: every model's reference has to be recorded again.
REFERENCE_PROTOCOL = WORKLOADS["reference"]["protocol"]
REFERENCE_TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def reference_document():
    """2,400 station log lines (about 73K tokens) with one fact to find, at line 1717."""
    lines = []
    for i in range(2400):
        t = 11 + (i * 7) % 23
        w = (i * 13) % 41
        lines.append(f"Station log {i:04d}: temperature {t} C, wind {w} km/h, pressure {990 + (i * 3) % 40} hPa.")
    lines[1717] = "Station log 1717: the backup generator was serviced by technician Ilsa Varga; code word PERIWINKLE."
    return "\n".join(lines)


REFERENCE_CASES = (
    ("short", {"messages": [{"role": "user", "content": "Explain in three sentences how a heat pump moves heat."}]}),
    ("code", {"messages": [{"role": "user", "content":
              "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."}]}),
    ("tool", {"messages": [{"role": "user", "content": "What's the weather in Lisbon right now?"}],
              "tools": REFERENCE_TOOLS}),
    ("document", {"messages": [{"role": "user", "content": reference_document() +
                  "\n\nWho serviced the backup generator, and what is the code word? Answer in one sentence."}]}),
    ("reasoning", {"messages": [{"role": "user", "content": "Is 221 a prime number? Answer briefly."}],
                   "chat_template_kwargs": {"enable_thinking": True}}),
)
# what an answer is, for the comparison: everything but its timing
REFERENCE_ANSWER = ("content", "reasoning", "tool_calls", "finish", "prompt_tokens", "completion_tokens")
# The cases checked by what the answer must name, not token by token: teacher-forcing the 73K-token document is one
# forward pass over all of it, with logits for every position, far more memory than a run has spare.
REFERENCE_FACTS = {"document": ("Ilsa Varga", "PERIWINKLE")}
# The other cases' answers are also recorded as token ids, with the logprobs teacher forcing gives each position: the
# answer's own token's and the REFERENCE_TOP likeliest tokens', rounded to LOGPROB_DIGITS (natural log).
REFERENCE_TOP = 20
LOGPROB_DIGITS = 3


def missing_facts(facts, text):
    """The facts an answer doesn't name (case aside)."""
    return [f for f in facts if f.lower() not in (text or "").lower()]


def reference_tokens(model_id):
    """The token data of the model's regression reference (catalog/reference/MODEL.json, which curate.py records),
    by case: what a run replays. Empty when there is no reference, or it has none."""
    try:
        ref = json.loads(catalog.reference_path(model_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(ref, dict) or ref.get("protocol") != REFERENCE_PROTOCOL:
        return {}
    return {c["name"]: c["tokens"] for c in ref.get("cases") or [] if (c.get("tokens") or {}).get("answer")}


# ---------------------------------------------------------------------------------------------------------------- runs
def parse_run(spec):
    """'4.05:22:262144:nostage:ram=16' -> a run; ValueError when malformed."""
    parts = str(spec).strip().split(":")
    if len(parts) < 3:
        raise ValueError(f"bad run {spec!r}: use QUANT:VRAM_GB|card:CONTEXT[:nostage][:ram=GIB][:mtp=N][:cpu[=CPUS]]"
                         f"[:prefetch] or a preset ({', '.join(PRESETS)})")
    quant, vram, ctx, *flags = parts
    if quant not in sizing.QUANTS:
        raise ValueError(f"unknown quant {quant!r} ({' or '.join(sizing.QUANTS)})")
    vram_gb = None if vram == "card" else float(vram)
    if vram_gb is not None and not 4 <= vram_gb <= 512:
        raise ValueError(f"VRAM {vram!r} out of range (4-512 GB, or card)")
    if not ctx.isdigit() or not config.MINIMUM["context"] <= int(ctx) <= config.MAXIMUM["context"]:
        raise ValueError(f"context {ctx!r} out of range ({config.MINIMUM['context']}-{config.MAXIMUM['context']})")
    context, nostage, ram, mtp, cpu, prefetch = int(ctx), False, None, 0, None, False
    for f in flags:
        if f == "nostage":
            nostage = True
        elif f.startswith("ram=") and f[4:].isdigit() and 1 <= int(f[4:]) <= 1024:
            ram = int(f[4:])
        elif f.startswith("mtp=") and f[4:].isdigit() and 1 <= int(f[4:]) <= 8:
            mtp = int(f[4:])
        elif f == "cpu" or (f.startswith("cpu=") and re.fullmatch(r"\d+(-\d+)?(,\d+(-\d+)?)*", f[4:])):
            cpu = f[4:]
        elif f == "prefetch":
            prefetch = True
        else:
            raise ValueError(f"unknown option {f!r} in {spec!r} (nostage, ram=GIB, mtp=1-8, cpu[=CPUS], prefetch)")
    label = (f"{quant}-{'card' if vram_gb is None else f'{vram_gb:g}gb'}-{context // 1024}k"
             + ("-nostage" if nostage else "") + (f"-ram{ram}" if ram else "") + (f"-mtp{mtp}" if mtp else "")
             + ("" if cpu is None else "-cpu" + (cpu.replace(",", "_") if cpu else ""))
             + ("-prefetch" if prefetch else ""))
    return {"quant": quant, "vram_gb": vram_gb, "context": context, "nostage": nostage, "ram": ram, "mtp": mtp,
            "cpu": cpu, "prefetch": prefetch, "label": label, "spec": ":".join(parts)}


def family_quants(quant):
    """The quants of the model `quant` belongs to (its catalog family), in catalog order."""
    cat = catalog.load()
    mine = cat.find(quant)
    return [m["quant"] for m in cat.models_for(sizing.ENGINE) if mine is None or m["family"] == mine["family"]]


def nostage_chunk(quant):
    """A staging-free run's prefill chunk (sizing.nostage_chunk); 1,536 for a quant the catalog no longer has."""
    return sizing.nostage_chunk(quant) if quant in sizing.QUANTS else 1536


def expand(specs, cfg, lean_gb=None):
    """Presets and run specs -> runs."""
    if isinstance(specs, str):
        specs = [s for s in re.split(r"[,\s]+", specs) if s]
    runs = []
    for s in specs:
        s = str(s).strip()
        if s == "current":
            runs.append(parse_run(f"{cfg['quant']}:card:{cfg['context']}"))
        elif s == "lean":
            if not lean_gb:
                raise ValueError("the lean profile's VRAM isn't known yet (no GPU sample)")
            runs.append(parse_run(f"{cfg['quant']}:{lean_gb:.1f}:{cfg['context']}:ram={int(cfg['lean_ram_gb'])}"))
        elif s == "quants":
            runs += [parse_run(f"{q}:card:{cfg['context']}") for q in family_quants(cfg["quant"])]
        elif s == "matrix":
            runs += [parse_run(x) for x in MATRIX]
        else:
            runs.append(parse_run(s))
    if not runs:
        raise ValueError("no runs given")
    if len(runs) > MAX_RUNS:
        raise ValueError(f"{len(runs)} runs; at most {MAX_RUNS} at a time")
    return runs


# -------------------------------------------------------------------------------------------------------------- engine
class Client:
    """HTTP to the engine under test, with the manager's key on every request."""

    def __init__(self, port, key, model="qwen3.8-flash-next"):
        self.base = f"http://127.0.0.1:{port}"
        self.headers = {"Content-Type": "application/json", **({BACKEND_KEY_HEADER: key} if key else {})}
        self.model = model                     # the name requests carry (the catalog's served name)

    def get(self, path, timeout=10):
        with urllib.request.urlopen(urllib.request.Request(self.base + path, headers=self.headers),
                                    timeout=timeout) as r:
            return json.loads(r.read())

    def post(self, path, payload, timeout=60):
        req = urllib.request.Request(self.base + path, json.dumps(payload).encode(), self.headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def tokens(self, text):
        return len(self.post("/tokenize", {"prompt": text})["tokens"])

    def document(self, head, size, tail, codes=False):
        """A document of about `size` tokens: head, a weather station log, tail. codes: the retrieval test, a locker
        combination in place of the log entries at 15%, 50% and 85%."""
        target = size - 48 - self.tokens(head + "\n" + tail)
        if not getattr(self, "per_entry", None):
            self.per_entry = self.tokens("".join(log_entries(0, 560))) / 560
        n = max(10, int(target / self.per_entry))
        off = self.tokens("".join(log_entries(0, n))) - target        # the estimate, corrected once
        if abs(off) > self.per_entry:
            n = max(10, n - round(off / self.per_entry))
        body = log_entries(0, n)
        for frac, name, value in codes and LOCKERS or ():
            body[int(n * frac)] = f"Note: locker {name} opens with the combination {value}.\n"
        return head + "\n" + "".join(body) + tail

    def events(self, path, payload, timeout=7200):
        """A streamed request's server-sent events, as (arrival time, event)."""
        req = urllib.request.Request(self.base + path, json.dumps(payload).encode(), self.headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    return
                yield time.perf_counter(), json.loads(body)

    def answer(self, content):
        """One chat answer: thinking off, greedy, streamed to its natural end (no max_tokens)."""
        body = {"model": self.model, "stream": True, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": content}]}
        t0 = time.perf_counter()
        first = last = usage = None
        text = []
        for t, ev in self.events("/v1/chat/completions", body):
            if ev.get("usage"):
                usage = ev["usage"]
            for choice in ev.get("choices") or ():
                delta = choice.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    first, last = first or t, t
                    text.append(delta.get("content") or "")
        n, p = int((usage or {}).get("completion_tokens") or 0), int((usage or {}).get("prompt_tokens") or 0)
        ttft = first - t0 if first else None
        return {"prompt_tokens": p, "completion_tokens": n, "ttft_s": round(ttft, 3) if ttft is not None else None,
                # from the unrounded time: a fast first token rounds to 0.000 s
                "prefill_tps": round(p / max(ttft, 1e-6), 1) if ttft is not None and p else None,
                "decode_tps": round((n - 1) / (last - first), 2) if first and last and last > first and n > 1 else None,
                "wall_s": round(time.perf_counter() - t0, 1), "answer": "".join(text)}

    def reference_body(self, body):
        """One of curation's requests (REFERENCE_CASES) as it is sent: greedy, thinking off unless the case turns it
        on, never max_tokens."""
        body = dict(body, model=self.model, temperature=0)
        body.setdefault("chat_template_kwargs", {"enable_thinking": False})
        return body

    def reference(self, name, body):
        """One of curation's requests, streamed to its natural end. The whole answer (text, reasoning, tool calls
        merged from their pieces, finish reason, token counts) and its timing."""
        body = dict(self.reference_body(body), stream=True, stream_options={"include_usage": True})
        t0 = time.perf_counter()
        first = last = usage = finish = None
        content, reasoning, calls = [], [], {}
        for t, ev in self.events("/v1/chat/completions", body):
            usage = ev.get("usage") or usage
            for choice in ev.get("choices") or ():
                delta = choice.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
                    first, last = first or t, t
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or "")
                for tc in delta.get("tool_calls") or ():
                    call = calls.setdefault(tc.get("index", len(calls)), {"name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    call["name"] += fn.get("name") or ""
                    call["arguments"] += fn.get("arguments") or ""
                finish = choice.get("finish_reason") or finish
        u = usage or {}
        n, p = int(u.get("completion_tokens") or 0), int(u.get("prompt_tokens") or 0)
        ttft = first - t0 if first else None
        decode_s = last - first if first and last and last > first else None
        return {"name": name, "content": "".join(content), "reasoning": "".join(reasoning),
                "tool_calls": [calls[i] for i in sorted(calls)], "finish": finish, "prompt_tokens": p,
                "completion_tokens": n, "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                "ttft_s": round(ttft, 3) if ttft is not None else None,
                "prefill_tps": round(p / max(ttft, 1e-6), 1) if ttft is not None and p else None,
                "decode_tps": round((n - 1) / decode_s, 2) if decode_s and n > 1 else None,
                "decode_s": round(decode_s, 4) if decode_s else None, "total_s": round(time.perf_counter() - t0, 3)}

    def prompt_ids(self, body):
        """A curation request's prompt as token ids, as the engine's chat routes build it (POST /prepare)."""
        try:
            return self.post("/prepare", self.reference_body(body))["ids"]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise RuntimeError("the engine has no /prepare: it predates the replay check") from e
            raise

    def greedy_ids(self, ids):
        """The greedy answer to a prompt's token ids, to its natural end, as token ids (the stop token included)."""
        out = self.post("/generate", {"input_ids": ids, "sampling_params": {"temperature": 0}}, timeout=1800)
        if "output_ids" not in out:
            raise RuntimeError("the engine's /generate doesn't return output_ids: it predates the replay check")
        return out["output_ids"]

    def logprobs(self, prompt, answer):
        """Teacher forcing: the answer's tokens fed after the prompt's in one forward pass (POST /generate with
        return_logprob). At each of the answer's positions, the logprob of its token there and the REFERENCE_TOP
        likeliest tokens with theirs, likeliest first."""
        m = self.post("/generate", {"input_ids": prompt + answer, "return_logprob": True,
                                    "logprob_start_len": len(prompt), "top_logprobs_num": REFERENCE_TOP},
                      timeout=600)["meta_info"]
        own, top = m.get("input_token_logprobs") or [], m.get("input_top_logprobs") or []
        if len(own) != len(answer) or len(top) != len(answer):
            raise RuntimeError(f"teacher forcing gave {len(top)} positions for an answer of {len(answer)} tokens")

        def r(x):
            return round(x, LOGPROB_DIGITS) + 0.0           # + 0.0: no -0.0
        return {"logprobs": [r(e[0]) for e in own], "top_ids": [[e[1] for e in row] for row in top],
                "top_logprobs": [[r(e[0]) for e in row] for row in top]}

    def engine_stats(self):
        """Where the expert picks came from, and the cache's size."""
        st = self.get("/stats")
        ec = (st.get("expert_cache") or [{}])[0]
        h = (st.get("nv2") or {}).get("host", {})
        picks = max(1, h.get("vram", 0) + h.get("ram", 0) + h.get("nvme", 0))
        return {"slots": ec.get("slots"), "arena_gb": ec.get("arena_gb"), "mem_peak_gb": ec.get("mem_peak_gb"),
                "hit_rate": ec.get("hit_rate"), "stage_overflow": (ec.get("stage") or {}).get("overflow"),
                "ram_slots": (st.get("nv2") or {}).get("ram_slots"),
                "picks_vram_pct": round(100 * h.get("vram", 0) / picks, 1),
                "picks_ram_pct": round(100 * h.get("ram", 0) / picks, 1),
                "picks_nvme_pct": round(100 * h.get("nvme", 0) / picks, 1)}


def log_entries(start, n):
    """n entries of a made-up weather station log, each different, from entry `start` on."""
    out = []
    for i in range(start, start + n):
        station, wind, sky = STATIONS[i % len(STATIONS)], WINDS[(i * 3) % len(WINDS)], SKIES[(i * 5) % len(SKIES)]
        out.append(f"Entry {i + 1:06d}: {station} logged {(i * 37) % 41 - 8} C, wind {(i * 13) % 60} km/h from the "
                   f"{wind}, {sky}, pressure {960 + (i * 29) % 80} hPa.\n")
    return out


def machine_facts(hw, kind, display):
    """The machine and stack a run was measured on, from the manager's hardware check."""
    g = (hw.get("gpus") or [{}])[0]
    e, repo = hw.get("engine") or {}, hw.get("repo") or {}
    pcie = f"{g['pcie_gen_max']}.0 x{g['pcie_width_max']}" if g.get("pcie_gen_max") else None
    return {"host": {"gpu": g.get("name"), "pcie": pcie, "display": display, "platform": kind,
                     "ram_total_gb": (hw.get("ram") or {}).get("total_gb")},
            "stack": {"torch": e.get("torch"), "cuda": e.get("cuda_version"),
                      "driver": (hw.get("driver") or {}).get("version"),
                      "commit": (repo.get("head") or "")[:7] or None}}


def reference_speed(cases):
    """A reference run's speeds. Decode over every answer of 16+ tokens together (the tokens after the first, over the
    time from the first to the last), which steadies the short ones; prefill from the long document (its prompt
    tokens over the time to its first token)."""
    timed = [c for c in cases if c.get("decode_s") and c["completion_tokens"] >= 16]
    tokens, seconds = sum(c["completion_tokens"] - 1 for c in timed), sum(c["decode_s"] for c in timed)
    doc = next((c for c in cases if c["name"] == "document"), {})
    return {"decode_tps": round(tokens / seconds, 2) if seconds else None, "prefill_tps": doc.get("prefill_tps")}


# ------------------------------------------------------------------------------------------------------------- results
def record_files(folder):
    """The runs' JSON files in a results folder (not its _plan.json)."""
    return [p for p in sorted(Path(folder).glob("*.json")) if not p.name.startswith("_")]


def load_records(folder):
    """Every run's JSON in a results folder."""
    out = []
    for p in record_files(folder):
        try:
            rec = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(rec, dict) and rec.get("label"):
            out.append(rec)
    return out


def counts(rec):
    """A run that needs no redoing: it finished, and wasn't cancelled or interrupted (a failure counts: it would fail
    again)."""
    return bool(rec.get("finished")) and not rec.get("interrupted") and rec.get("error") != "cancelled"


def unfinished():
    """The latest benchmark, if it has runs left: (folder, plan, runs left), else None."""
    for h in history(50):
        d = BENCH_DIR / h["name"]
        try:
            plan = json.loads((d / PLAN).read_text())
        except (OSError, ValueError):
            continue                                   # from before resuming existed
        if plan.get("protocol") != PROTOCOL:
            return None                                # measured another way: finishing it would mix the two
        done = {r["label"] for r in load_records(d) if counts(r)}
        left = [r for r in plan.get("runs") or [] if isinstance(r, dict) and r.get("label") not in done]
        return (d, plan, left) if left else None
    return None


def new_run_dir():
    """A results folder named after the date and time (with a suffix if one started in the same second)."""
    base, n = time.strftime("%Y%m%d-%H%M%S"), 1
    d = BENCH_DIR / base
    while d.exists():
        n += 1
        d = BENCH_DIR / f"{base}-{n}"
    d.mkdir(parents=True)
    return d


def history(limit=12):
    """Earlier result folders, newest first."""
    if not BENCH_DIR.is_dir():
        return []
    dirs = sorted((d for d in BENCH_DIR.iterdir() if d.is_dir() and RUN_DIR.match(d.name)), reverse=True)
    return [{"name": d.name, "runs": len(record_files(d))} for d in dirs[:limit]]


def previous_record(rec, skip=None):
    """The latest earlier run of the same configuration and workload on the same GPU."""
    if not BENCH_DIR.is_dir():
        return None
    for d in sorted((d for d in BENCH_DIR.iterdir() if d.is_dir() and RUN_DIR.match(d.name)), reverse=True):
        p = d / f"{rec['label']}.json"
        if d == skip or not p.is_file():
            continue
        try:
            old = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if (not old.get("error") and old.get("workload") == rec.get("workload")
                and old.get("protocol") == rec.get("protocol")
                and (old.get("host") or {}).get("gpu") == (rec.get("host") or {}).get("gpu")):
            return dict(old, _dir=d.name)
    return None


def compare(rec, old):
    """Percent change against an earlier run."""
    if not old:
        return None

    def pct(a, b):
        return round(100 * (a - b) / b, 1) if a and b else None
    prefill = {n: pct((v or {}).get("median"), ((old.get("prefill") or {}).get(n) or {}).get("median"))
               for n, v in (rec.get("prefill") or {}).items()}
    return {"run": old.get("_dir"), "date": old.get("date"), "stack": old.get("stack"),
            "decode_pct": pct(rec.get("decode_tps_median"), old.get("decode_tps_median")),
            "document_pct": pct((rec.get("long") or {}).get("prefill_tps"), (old.get("long") or {}).get("prefill_tps")),
            "prefill_pct": {n: v for n, v in prefill.items() if v is not None}}


def summary(r):
    """A run in a few figures, for the dashboard and the CLI."""
    e, long = r.get("engine") or {}, r.get("long") or {}
    return {"label": r.get("label"), "quant": r.get("quant"), "context": r.get("context"),
            "vram": "whole card" if uncapped(r) else f"{r.get('usable_gb'):g} GB", "ram_gib": r.get("ram_gib"),
            "decode": r.get("decode_tps_median"), "decode_min": r.get("decode_tps_min"),
            "ttft": r.get("short_ttft_median_s"), "startup_s": r.get("startup_s"),
            "model_vram_gb": r.get("model_vram_gb"),
            "prefill": {n: v.get("median") for n, v in (r.get("prefill") or {}).items() if v.get("median")},
            "document": ({"tps": round(long["prefill_tps"]), "tokens": long.get("prompt_tokens"),
                          "found": long.get("codes_found", long.get("needle_found"))}
                         if long.get("prefill_tps") else None),
            "picks": (f"{e['picks_vram_pct']:.0f} / {e['picks_ram_pct']:.0f} / {e['picks_nvme_pct']:.0f}"
                      if e.get("picks_vram_pct") is not None else None),
            "error": r.get("error"), "previous": r.get("previous"), "stack": r.get("stack"), "date": r.get("date")}


def uncapped(r):
    return r.get("method") == "uncapped card"


def card_class(usable):
    return next(name for floor, name in ((26, "32 GB"), (19, "24 GB"), (12.5, "16 GB"), (8.5, "12 GB"), (0, "8 GB"))
                if usable >= floor)


def fmt(v, spec="{:.1f}", none="-"):
    return none if v is None else spec.format(v)


def table(records):
    """The report table (markdown) of a results folder; also reads results from before this tool."""
    def order(r):
        vram = r["card_total_gb"] if uncapped(r) else r["usable_gb"]
        return r["quant"], uncapped(r), -vram, r["context"], r.get("ram_gib") or 0
    rows = sorted(records, key=order)
    out = ["| Card | Quant | VRAM for the model | Context | RAM cache | Prefill chunk | Cached experts "
           "| Expert picks VRAM / RAM / NVMe | Decode, median (min) | Prefill | First token | Startup |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        e, plan, long = r.get("engine") or {}, r.get("plan") or {}, r.get("long") or {}
        nostage = r.get("nostage") or plan.get("staging") is False
        chunk = f"{nostage_chunk(r['quant']):,}, no staging" if nostage else f"{plan.get('chunk') or 0:,}"
        card = f"{round(r['card_total_gb'] * 1e9 / 2**30)} GB, uncapped" if uncapped(r) else card_class(r["usable_gb"])
        vram = "whole card" if uncapped(r) else f"{r['usable_gb']:g} GB"
        head = (f"| {card} | {r['quant']} | {vram} | {r['context'] // 1024}K | {fmt(r.get('ram_gib'), '{} GiB')} | "
                f"{chunk if plan.get('ok') or r.get('nostage') else '-'} |")
        if not r.get("decode_tps_median") or r.get("error"):
            tail = r.get("log_tail") or ""
            when = " on the long prompt" if r.get("decode_tps_median") else " on the first request"
            oom = "GUARD vram" in tail or "VRAM LIMIT EXCEEDED" in tail     # the latter in records before Repertory
            why = "ran out of VRAM" + when if oom else str(r.get("error") or "no result").splitlines()[0][:100]
            dec = f" (decode {fmt(r['decode_tps_median'])} tok/s before that)" if r.get("decode_tps_median") else ""
            out.append(f"{head} - | - | **fails**: {why}{dec} | - | - | {fmt(r.get('startup_s'), '{:.0f} s')} |")
            continue
        picks = (f"{e['picks_vram_pct']:.0f} / {e['picks_ram_pct']:.0f} / {e['picks_nvme_pct']:.0f} %"
                 if e.get("picks_vram_pct") is not None else "-")
        prefill = (f"{long['prefill_tps']:,.0f} tok/s ({long['prompt_tokens'] // 1000}K prompt"
                   f"{'' if long.get('codes_found', long.get('needle_found')) else ', missed what it had to find'})"
                   if long.get("prefill_tps") else "-")
        out.append(f"{head} {e.get('slots') or 0:,} | {picks} | **{fmt(r['decode_tps_median'])}** "
                   f"({fmt(r.get('decode_tps_min'))}) tok/s | {prefill} | "
                   f"{fmt(r.get('short_ttft_median_s'), '{:.2f} s')} | {fmt(r.get('startup_s'), '{:.0f} s')} |")
    return "\n".join(out)


def prefill_table(records):
    """Document reading by size, when the runs measured it."""
    sizes = sorted({int(n) for r in records for n, v in (r.get("prefill") or {}).items() if v.get("median")})
    if not sizes:
        return ""
    out = ["| Run | " + " | ".join(f"{n // 1024}K" for n in sizes) + " |", "|---|" + "---|" * len(sizes)]
    for r in records:
        p = r.get("prefill") or {}
        if any((p.get(str(n)) or {}).get("median") for n in sizes):
            out.append(f"| {r['label']} | " + " | ".join(fmt((p.get(str(n)) or {}).get("median"), "{:,.0f} tok/s")
                                                         for n in sizes) + " |")
    return "\n".join(out)


def measured_rows(records, source):
    """sizing.MEASURED entries for the completed runs (check gpu and ram describe the machine)."""
    out = []
    for r in records:
        long = r.get("long") or {}
        if r.get("error") or not r.get("decode_tps_median") or not long.get("prefill_tps"):
            continue
        h, plan = r.get("host") or {}, r.get("plan") or {}
        card = round(r["card_total_gb"] * 1e9 / 2**30) if uncapped(r) else int(card_class(r["usable_gb"]).split()[0])
        display = h.get("display") if uncapped(r) else card > 8 or r["usable_gb"] < 7.5
        ram = round(h["ram_total_gb"] * 1e9 / 2**30) if h.get("ram_total_gb") else None
        gpu = (h.get("gpu") or "?").replace("NVIDIA GeForce ", "")
        extra = ("" if uncapped(r) else "capped=True, ") + ("" if display is not False else "display=False, ") + \
                ("" if plan.get("staging", True) and not r.get("nostage") else "staging=False, ")
        out.append(f'    dict(card={card}, {extra}quant="{r["quant"]}", context={r["context"]}, '
                   f'ram_cache_gib={r.get("ram_gib")}, decode={r["decode_tps_median"]:.1f}, '
                   f'prefill={long["prefill_tps"]:.0f}, ram={ram}, '
                   f'platform="{h.get("platform", "linux")}", gpu="{gpu}", source="{source}"),')
    return "\n".join(out)


# -------------------------------------------------------------------------------------------------------------- runner
class Bench:
    """A benchmark inside the manager: its state, output lines and results."""

    def __init__(self, svc):
        self.svc = svc
        self.running = self.cancelled = self.waiting_for_last = False
        self.out = collections.deque(maxlen=3000)
        self.seq = 0
        self.runs, self.records = [], []
        self.workload = self.current = self.dir = self.task = None
        self.started = self.finished = None
        self.planned = 0
        self.resume_dir = None                 # the folder of the benchmark being continued
        self.base = None                       # the GPU just before this run's server started: other apps' baseline
        self.yielded = self.waiting = None     # why this run stepped aside; what the PC still needs
        self.echo = None                       # also print each line (`repertory bench` without the service)
        last = history(1)
        if last:                               # show the latest results after a restart
            self.dir = BENCH_DIR / last[0]["name"]
            self.records = load_records(self.dir)

    def emit(self, line):
        self.seq += 1
        self.out.append((self.seq, line))
        log.info("bench: %s", line)
        if self.echo:
            self.echo(line)

    def report(self, since=0, name=None):
        records = self.records
        if name and name != (self.dir.name if self.dir else None):    # an earlier run, by its folder name
            records = load_records(BENCH_DIR / name) if RUN_DIR.match(name) and (BENCH_DIR / name).is_dir() else []
        left = None if self.running else unfinished()
        return {"running": self.running, "workload": self.workload, "current": self.current,
                "runs": [r["label"] for r in self.runs], "dir": self.dir.name if self.dir else None,
                "waiting": self.waiting,
                "resumable": {"dir": left[0].name, "left": len(left[2]), "workload": left[1].get("workload")}
                if left else None,
                "results": [summary(r) for r in records], "records": records,
                "started": self.started, "finished": self.finished, "seq": self.seq,
                "lines": [[n, t] for n, t in self.out if n > since][-800:],
                "presets": PRESETS, "workloads": list(WORKLOADS), "history": history()}

    def start(self, specs, workload="standard", resume=False):
        """Check the request and start the benchmark in the background: (ok, message). resume: continue the latest
        unfinished benchmark instead (its runs left, into its folder, with its workload)."""
        svc = self.svc
        if self.running:
            return False, "a benchmark is already running"
        if svc.observe:
            return False, "observe mode: this manager does not control the model"
        if svc.setup.running:
            return False, f"setup step '{svc.setup.running}' is running; try again when it finishes"
        m = svc.mon.get()
        if not m:
            return False, "no GPU reading yet; try again in a moment"
        folder, done = None, []
        if resume:
            left = unfinished()
            if not left:
                return False, "no unfinished benchmark to resume"
            folder, plan, todo = left
            workload = plan.get("workload")
            try:
                runs = [parse_run(r["spec"]) for r in todo]
            except (KeyError, TypeError, ValueError) as e:
                return False, f"can't resume {folder.name}: {e}"
            labels = {r["label"] for r in runs}
            done = [r for r in load_records(folder) if r["label"] not in labels]
        if workload not in WORKLOADS:
            return False, f"unknown workload {workload!r} ({', '.join(WORKLOADS)})"
        if not resume:
            try:
                runs = expand(specs, svc.cfg, lean_gb=min(svc.lean_budget(m), svc.full_budget(m)) / GB)
            except ValueError as e:
                return False, str(e)
        self.running, self.cancelled = True, False
        self.runs, self.records, self.workload, self.resume_dir = runs, done, workload, folder
        self.planned = len(done) + len(runs)
        self.started, self.finished, self.current = time.time(), None, None
        self.out.clear()
        self.task = asyncio.create_task(self._run(self.task))
        n = f"{len(runs)} run{'s' if len(runs) > 1 else ''}"
        if folder:
            return True, f"resuming the benchmark of {folder.name}: {n} left, {workload} workload"
        return True, f"benchmark started: {n}, {workload} workload"

    async def cancel(self):
        if not self.running:
            return False, "no benchmark is running"
        self.cancelled = True
        if self.waiting_for_last:                   # the server running is the model's, starting again: not ours
            self.emit("cancelling")
            return True, "cancelling"
        self.emit("cancelling: stopping the benchmark's model server")
        await asyncio.to_thread(self.svc.host.backend_stop)
        return True, "cancelling"

    async def _run(self, last=None):
        svc = self.svc
        if last and not last.done():
            # The last benchmark is still starting the model again (after a stop, the start first waits for WSL to
            # hand RAM back). A run's server started now would be taken for a stray by that start, stopped, and
            # replaced by the model on the same port: wait for it, then stop the model as any benchmark does.
            self.waiting_for_last = True
            self.emit("waiting for the model to finish starting again after the last benchmark")
            await asyncio.wait([last])
            self.waiting_for_last = False
        before = svc.be.tier if svc.be.state in ("running", "starting") else None
        try:
            self.dir = self.resume_dir or new_run_dir()
            if not self.resume_dir:
                (self.dir / PLAN).write_text(json.dumps({"workload": self.workload, "protocol": PROTOCOL,
                                                            "runs": self.runs}, indent=1))
            await svc.setup.check()                       # GPU, driver, CUDA, PyTorch and commit for the records
            facts = machine_facts(svc.setup.hw or {}, svc.host.kind, svc.gpu_display)
            st = facts["stack"]
            self.emit((f"resuming {self.dir.name} ({self.planned - len(self.runs)} of {self.planned} done): "
                       if self.resume_dir else "")
                      + f"{len(self.runs)} run(s), {self.workload} workload, on {facts['host']['gpu']} "
                        f"(driver {st['driver']}, CUDA {st['cuda']}, PyTorch {st['torch']}, commit {st['commit']})")
            if svc.be.state != "stopped":
                self.emit("stopping the model for the benchmark; requests get an error until it's done")
                await svc.stop("benchmark", "benchmark")
            for r in self.runs:
                if self.cancelled:
                    break
                self.current = r["label"]
                for tries in range(1, MAX_YIELDS + 2):
                    rec = await self._one(r, facts)
                    if not rec.get("interrupted") or self.cancelled:
                        break
                    if tries > MAX_YIELDS:
                        self.emit(f"  interrupted {tries} times; moving on (resuming the benchmark redoes it)")
                        break
                    await self.wait_for_pc()
                self.records.append(rec)
        except Exception as e:  # noqa: BLE001 - report it, then restore the service
            log.exception("benchmark failed")
            self.emit(f"benchmark failed: {e}")
        finally:
            self.current = self.base = self.waiting = None
            await asyncio.to_thread(svc.host.backend_stop)
            self.running, self.finished = False, time.time()
            done = sum(1 for r in self.records if not r.get("error"))
            left = unfinished()
            self.emit(f"{'cancelled' if self.cancelled else 'finished'}: {done} of {self.planned} run(s) completed; "
                      f"results in {self.dir}"
                      + (f"; {len(left[2])} left: `repertory bench --resume` or the Resume button runs them"
                         if left and left[0] == self.dir else ""))
            if before:
                self.emit(f"starting the model again ({before})")
                await svc.start(before, "benchmark finished")

    def why_stopped(self, default):
        if self.cancelled:
            return "cancelled"
        if self.yielded:
            return "interrupted: " + "; ".join(self.yielded)
        return default

    def disturbed(self, m):
        """Why the PC needs the GPU or RAM back now ([] when it doesn't). The benchmark's own server never counts: on
        Windows the per-app counters can't see WSL, and on Linux its processes are left out."""
        cfg = self.svc.cfg
        why = []
        hits = sorted({w.lower() for w in cfg["watchlist"]} & set(m.get("running") or ()))
        if hits:
            why.append("watched app running: " + ", ".join(hits))
        if m.get("ext_util", 0) >= cfg["ext_util_pct"]:
            top = m.get("ext_util_top") or []
            why.append(f"GPU busy: {top[0][0] if top else 'another app'} at {m['ext_util']:.0f}%")
        if self.base is not None:
            grew = self.svc.ext_vram(m) - self.base["ext"]
            free = m["vram_total"] - m["vram_used"]
            if grew > 0.75 * GB and free < cfg["vram_min_free_gb"] * GB:
                why.append(f"other apps need VRAM (+{grew / GB:.1f} GB, {free / GB:.1f} GB free)")
        if m["ram_avail"] < RAM_FLOOR_GB * GB:
            why.append(f"RAM nearly out ({m['ram_avail'] / GB:.1f} GB free)")
        return why

    async def own_server(self, pid):
        """Whether the model server running is still the one this run started (the launcher's pid), not another one
        started on the same port since: the manager's own start takes a server it didn't start for a stray."""
        st, now = await asyncio.to_thread(self.svc.host.backend_status)
        return st == "running" and now == pid

    async def watch(self):
        """While a run's server is up: once the PC has needed the GPU or RAM for YIELD_AFTER_S, stop the server. The
        run is done again when the PC is free."""
        since = None
        while True:
            await asyncio.sleep(WATCH_S)
            m = self.svc.mon.get()
            why = self.disturbed(m) if m else []
            if not why:
                since = None
                continue
            since = since or time.time()
            if time.time() - since >= YIELD_AFTER_S:
                self.yielded = why
                self.emit("  stepping aside: " + "; ".join(why))
                await asyncio.to_thread(self.svc.host.backend_stop)
                return

    async def wait_for_pc(self):
        """After stepping aside: wait until the PC has been free for recover_after_s, or the benchmark is cancelled."""
        need = self.svc.cfg["recover_after_s"]
        self.waiting = self.yielded
        self.emit(f"  waiting until the PC has been free for {need:g} s, then running {self.current} again")
        clear = None
        while not self.cancelled:
            await asyncio.sleep(WATCH_S)
            m = self.svc.mon.get()
            why = self.disturbed(m) if m else ["no GPU reading"]
            if why:
                clear, self.waiting = None, why
                continue
            clear = clear or time.time()
            if time.time() - clear >= need:
                self.emit("  the PC is free again")
                break
        self.waiting = None

    def engine_env(self, r, plan, m, budget):
        """What the manager would pass the engine for this run (a capped run also gets the emulated card's guard)."""
        svc, cfg = self.svc, self.svc.cfg
        target = plan["ram_target"] if plan["ok"] else sizing.RAM_COVERAGE * sizing.expert_bytes(r["quant"])
        if r["ram"]:
            ram_gib = r["ram"]
        else:
            caps = [c for c in (cfg["max_ram_cache_gib"],
                                svc.learned.get(svc.learned_key(r["quant"], r["context"]), {}).get("ram_cap")) if c]
            ram_gib = sizing.ram_cache_gib(target, svc.ram_for_start(m), cfg, svc.ram_overhead(r["quant"]),
                                           min(caps) if caps else None)
        paths = catalog.load().model(r["quant"])["paths"]
        env = {"QUANT": r["quant"], "CONTEXT": str(r["context"]), "PORT": str(cfg["backend_port"]),
               "MODEL_DIR": "./" + paths["model_dir"], "STORE": "./" + paths["store"],
               "HOST": "127.0.0.1", "RAM_GB": str(ram_gib)}
        if r["vram_gb"] is None:
            env["QWEN_VRAM_LIMIT_BYTES"] = str(int(m["vram_total"] - svc.headroom(m)))
            env["QWEN_VRAM_GUARD_BYTES"] = str(int(m["vram_total"] + 2 * GB))
        else:
            env["QWEN_VRAM_LIMIT_BYTES"] = str(int(m["vram_used"] + budget))
            # past this the server exits, as running out of memory would on the emulated card (slack for the desktop)
            env["QWEN_VRAM_GUARD_BYTES"] = str(int(m["vram_used"] + budget + 0.25 * GB))
        if r["nostage"]:
            c = nostage_chunk(r["quant"])
            env.update({"QWEN_CHUNK": str(c), "GLM53_EC_STAGE_GB": "0", "GLM53_EC_ELASTIC_GB": "0",
                        "GLM53_EC_RESERVE_GB": f"{sizing.nostage_reserve(r['quant'], c) / GIB:.2f}"})
        else:
            env.update(sizing.engine_env(plan))
        if r.get("mtp"):
            env["QWEN_MTP"] = str(r["mtp"])
        if r.get("cpu") is not None:
            env["QWEN_CPU_LANE"] = "1"
            if r["cpu"]:
                env["GLM53_NV_CPU_CPUS"] = r["cpu"]
        if r.get("prefetch"):
            env["GLM53_NV_PREFETCH"] = "1"
        return env, ram_gib

    async def _one(self, r, facts):
        svc, cfg = self.svc, self.svc.cfg
        self.emit(f"=== {r['label']}")
        self.yielded = None
        await asyncio.to_thread(svc.host.backend_stop)
        await asyncio.sleep(SETTLE_S)
        m = await svc.fresh_sample(time.time())
        self.base = {"vram_used": m["vram_used"], "perproc_ded": m.get("perproc_ded", 0.0)}
        self.base["ext"] = svc.ext_vram(m)
        card = r["vram_gb"] is None
        budget = m["vram_total"] - m["vram_used"] - svc.headroom(m) if card else r["vram_gb"] * GB
        plan = sizing.plan(r["quant"], budget, r["context"])
        rec = {"label": r["label"], "quant": r["quant"], "usable_gb": round(budget / GB, 2), "context": r["context"],
               "nostage": r["nostage"], "mtp": r.get("mtp", 0), "cpu_lane": r.get("cpu"),
               "prefetch": r.get("prefetch", False), "workload": self.workload, "protocol": PROTOCOL,
               "date": time.strftime("%Y-%m-%d %H:%M"),
               "device_used_before_gb": round(m["vram_used"] / GB, 2), "card_total_gb": round(m["vram_total"] / GB, 2),
               "method": "uncapped card" if card else "engine cap (limit + guard)", **facts,
               "plan": {k: v for k, v in plan.items() if isinstance(v, (int, float, str, bool)) or v is None}}
        path = self.dir / f"{r['label']}.json"

        def save():
            path.write_text(catalog.flat_numbers(json.dumps(rec, indent=1)))      # token data: a line a position
        have = (svc.setup.hw or {}).get("engine") or {}
        model = catalog.load().model(r["quant"])
        if have.get("driver") and have.get(catalog.setup_id(model)) != "1":
            rec["error"] = f"the {r['quant']} bpw model isn't downloaded (Setup tab)"
        elif not plan["ok"] and not r["nostage"]:
            rec["error"] = f"does not fit: {plan.get('reason')}"
        if rec.get("error"):
            self.emit(f"  skipped: {rec['error']}")
            rec["finished"] = True
            save()
            return rec
        env, rec["ram_gib"] = self.engine_env(r, plan, m, budget)
        t0 = time.time()
        ok, pid = await asyncio.to_thread(svc.host.backend_start, {**env, "REPERTORY_BACKEND_KEY": svc.backend_key},
                                          ["server"])
        watcher = asyncio.create_task(self.watch()) if ok else None
        try:
            up = replaced = False
            if ok:
                url = f"http://127.0.0.1:{cfg['backend_port']}/v1/models"
                while time.time() - t0 < cfg["start_timeout_s"] and not self.cancelled and not self.yielded:
                    if await asyncio.to_thread(http_ok, url, 2.0, svc.backend_auth):
                        up = await self.own_server(pid)        # what answers on the port must be this run's server
                        replaced = not up
                        break
                    st, now = await asyncio.to_thread(svc.host.backend_status)
                    if st != "running" or now != pid:
                        replaced = st == "running"
                        break
                    await asyncio.sleep(2)
            rec["startup_s"] = round(time.time() - t0, 1)
            if up:
                m2 = await svc.fresh_sample(time.time())
                rec["model_vram_gb"] = round((m2["vram_used"] - m["vram_used"]) / GB, 2)
                self.emit(f"  ready in {rec['startup_s']:.0f} s: {rec['model_vram_gb']:.1f} GB VRAM, "
                          f"{rec['ram_gib']} GiB RAM cache")
                save()
                client = Client(cfg["backend_port"], svc.backend_key, catalog.served_name(model))
                try:
                    await self.workload_steps(client, r, rec, save)
                    rec["engine"] = await asyncio.to_thread(client.engine_stats)
                except Exception as e:  # noqa: BLE001 - one failed run must not end the others
                    rec["error"] = self.why_stopped(f"workload failed: {e}")
                st, now = await asyncio.to_thread(svc.host.backend_status)
                if st == "running" and now != pid:
                    rec["error"] = REPLACED                     # whatever it measured may be the other server's
                elif st != "running" and not rec.get("error"):
                    rec["error"] = "the server exited during the workload"
            else:
                rec["error"] = self.why_stopped(REPLACED if replaced else
                                                f"the server did not come up{'' if ok else f' (launcher: {pid})'}")
        finally:
            if watcher:
                watcher.cancel()
        if self.cancelled or self.yielded:                # whatever it managed, a run cut short isn't a result
            rec["error"] = self.why_stopped("")
            rec["interrupted"] = not self.cancelled
        elif rec.get("error") and svc.host.follower:
            rec["log_tail"] = svc.host.follower.tail(8)
        await asyncio.to_thread(svc.host.backend_stop)
        # the next run, or the model afterwards, can reuse this server's RAM before WSL has handed it back
        svc.last_stop_t = time.time()
        svc.freed = (svc.last_stop_t, svc.ram_for_start(m), 0.0)
        if not rec.get("error"):
            rec["previous"] = compare(rec, previous_record(rec, skip=self.dir))
        rec["finished"] = True
        save()
        s = summary(rec)
        if rec.get("interrupted"):
            pass                                          # stepping aside was reported; _run decides what's next
        elif rec.get("error"):
            self.emit(f"  failed: {rec['error']}")
        else:
            doc = s["document"]
            self.emit(f"  answers {fmt(s['decode'])} tok/s (min {fmt(s['decode_min'])})"
                      + (f", long document {doc['tps']:,} tok/s ({doc['tokens'] // 1000}K, codes "
                         f"{'found' if doc['found'] else 'MISSED'})" if doc else "")
                      + "".join(f", {int(n) // 1024}K {v:,.0f}" for n, v in s["prefill"].items())
                      + (f"; vs {rec['previous']['date']}: answers {rec['previous']['decode_pct']:+.1f}%"
                         if rec.get("previous") and rec["previous"].get("decode_pct") is not None else ""))
        return rec

    async def workload_steps(self, client, r, rec, save):
        """The workload over HTTP: the warm-up, then each trial's answer and documents, then the retrieval test."""
        if self.workload == "resize":
            return await self.resize_steps(client, r, rec, save)
        if self.workload == "resize-stress":
            return await self.stress_steps(client, r, rec, save)
        if self.workload == "reference":
            return await self.reference_steps(client, rec, save)
        wl = WORKLOADS[self.workload]

        def check():
            if self.cancelled or self.yielded:
                raise RuntimeError(self.why_stopped("stopped"))

        async def ask(text):
            check()
            return await asyncio.to_thread(client.answer, text)
        for text in WARMUP:                                           # not scored
            await ask(text)
        answers, docs = [], {}
        for trial in range(wl["trials"]):
            a = await ask(ANSWER.format(t=trial))
            a.pop("answer")
            answers.append(a)
            rec["decode_runs"] = answers
            save()
            read = []
            for n in wl["documents"]:
                if n + 1024 > r["context"]:
                    continue
                text = await asyncio.to_thread(client.document, DOC_HEAD.format(t=trial, size=n), n, DOC_TAIL)
                d = await ask(text)
                docs.setdefault(str(n), []).append(d["prefill_tps"])
                rec["prefill"] = {k: {"median": statistics.median(v), "runs": v} for k, v in docs.items() if all(v)}
                save()
                read.append(f"{n // 1024}K {fmt(d['prefill_tps'], '{:,.0f}')}")
            self.emit(f"  trial {trial + 1}: answer of {a['completion_tokens']:,} tokens at "
                      f"{fmt(a['decode_tps'])} tok/s"
                      + (f"; documents read at {', '.join(read)} tok/s" if read else ""))
        rates = [a["decode_tps"] for a in answers if a["decode_tps"]]
        rec["decode_tps_median"] = round(statistics.median(rates), 2) if rates else None
        rec["decode_tps_min"] = min(rates) if rates else None
        ttfts = [a["ttft_s"] for a in answers if a["ttft_s"] is not None]
        rec["short_ttft_median_s"] = round(statistics.median(ttfts), 3) if ttfts else None
        save()
        if wl["retrieval"]:
            n = min(RETRIEVAL_MAX, r["context"] - 2048)
            text = await asyncio.to_thread(client.document, RETRIEVAL_HEAD, n, RETRIEVAL_TAIL, True)
            a = await ask(text)
            said = a.pop("answer")
            a["codes_found"] = all(code in said for _, _, code in LOCKERS)
            rec["long"] = a
            self.emit(f"  long document: {a['prompt_tokens']:,} tokens read at {fmt(a['prefill_tps'], '{:,.0f}')} "
                      f"tok/s, {'all three combinations found' if a['codes_found'] else 'combinations MISSED'}")
            save()

    async def reference_steps(self, client, rec, save):
        """Curation's regression requests (REFERENCE_CASES) after the warm-up, each answer recorded whole; curate.py
        compares them with the model's reference. The figures the tables show come along: decode over the answers,
        prefill and the fact found in the long document. Then, untimed, the token data (replay_steps)."""
        def check():
            if self.cancelled or self.yielded:
                raise RuntimeError(self.why_stopped("stopped"))
        for text in WARMUP:                                           # not scored
            check()
            await asyncio.to_thread(client.answer, text)
        hw = self.svc.setup.hw or {}
        cases = []
        rec["engine_commit"] = (hw.get("engine") or {}).get("engine_head") or (hw.get("repo") or {}).get("head")
        rec["reference"] = {"protocol": REFERENCE_PROTOCOL, "cases": cases}
        for name, body in REFERENCE_CASES:
            check()
            a = await asyncio.to_thread(client.reference, name, body)
            cases.append(a)
            save()
            calls = ", ".join(f"{c['name']}({c['arguments']})" for c in a["tool_calls"])
            self.emit(f"  {name}: {a['completion_tokens']:,} tokens ({a['finish']}"
                      + (f", calls {calls}" if calls else "") + f"), first token after {fmt(a['ttft_s'], '{:.2f}')} s"
                      + (f", {fmt(a['decode_tps'])} tok/s" if a["decode_tps"] else ""))
        speed = reference_speed(cases)
        rec["reference"].update(speed)
        rates = [c["decode_tps"] for c in cases if c.get("decode_tps") and c["completion_tokens"] >= 16]
        rec["decode_tps_median"], rec["decode_tps_min"] = speed["decode_tps"], (min(rates) if rates else None)
        rec["short_ttft_median_s"] = cases[0]["ttft_s"]
        doc = next(c for c in cases if c["name"] == "document")
        rec["long"] = {"prompt_tokens": doc["prompt_tokens"], "prefill_tps": doc["prefill_tps"],
                       "needle_found": not missing_facts(REFERENCE_FACTS["document"], doc["content"])}
        self.emit(f"  reference: decode {fmt(speed['decode_tps'])} tok/s over the answers, the "
                  f"{doc['prompt_tokens']:,}-token document read at {fmt(speed['prefill_tps'], '{:,.0f}')} tok/s")
        save()
        await self.replay_steps(client, rec, cases, check, save)

    async def replay_steps(self, client, rec, cases, check, save):
        """Each answer but the document's (REFERENCE_FACTS) as token ids, for curate.py: the prompt's ids as the chat
        route built them (/prepare), the greedy answer to them (/generate) and its logprobs, teacher-forced
        ("tokens", what a new reference records); and when the model's reference has them, its own answer's tokens
        teacher-forced on this run ("replay", what the comparison reads)."""
        model = catalog.load().model(rec["quant"])
        old = reference_tokens(model["id"]) if model else {}
        positions = replayed = 0
        for (name, body), case in zip(REFERENCE_CASES, cases, strict=True):
            if name in REFERENCE_FACTS:
                continue
            check()
            prompt = await asyncio.to_thread(client.prompt_ids, body)
            answer = await asyncio.to_thread(client.greedy_ids, prompt)
            case["tokens"] = {"prompt": prompt, "answer": answer,
                              **await asyncio.to_thread(client.logprobs, prompt, answer)}
            positions += len(answer)
            if name in old:
                case["replay"] = await asyncio.to_thread(client.logprobs, old[name]["prompt"], old[name]["answer"])
                replayed += 1
            save()
        self.emit(f"  tokens: {positions:,} answer positions teacher-forced"
                  + (f"; the reference's {replayed} answers replayed" if replayed else "; no reference to replay"))

    def resize_kit(self, client, r, wl):
        """What the resize workloads share: answers, the per-size check, the manager's own switch requests, and the
        engine's RAM cache state (it grows in the background)."""
        svc, bench = self.svc, self
        if r["vram_gb"] is not None or r["nostage"]:
            raise RuntimeError("the resize workloads need a whole-card run with staging (they start as full does)")

        class Kit:
            docs = 0

            async def ask(self, text):
                if bench.cancelled or bench.yielded:
                    raise RuntimeError(bench.why_stopped("stopped"))
                return await asyncio.to_thread(client.answer, text)

            async def vram(self):
                return (await svc.fresh_sample(time.time()))["vram_used"]

            async def document(self):
                # a document of its own each time, or it comes from the prompt cache and prefill is never exercised;
                # several chunks long at either chunk size
                Kit.docs += 1
                return await self.ask(await asyncio.to_thread(
                    client.document, DOC_HEAD.format(t=Kit.docs, size=wl["document"]), wl["document"], DOC_TAIL))

            async def verify(self):
                check = await asyncio.to_thread(client.get, "/nv_verify?n=64", 300)
                return check, not any(check.get(k) for k in VERIFY_BAD)

            async def at(self, name):
                first, again = await self.ask(ANSWER.format(t=0)), await self.ask(ANSWER.format(t=0))
                doc = await self.document()
                check, ok = await self.verify()
                stats = await asyncio.to_thread(client.engine_stats)
                ph = {"at": name, "vram_used_gb": round(await self.vram() / GB, 2), "engine": stats,
                      "text": first.pop("answer"), "again_text": again.pop("answer"), "answer": first, "again": again,
                      "document": {k: v for k, v in doc.items() if k != "answer"}, "verify": check, "verify_ok": ok}
                bench.emit(f"  {name}: {stats['slots']} experts in VRAM, {ph['vram_used_gb']:.1f} GB in use; "
                           f"answers {fmt(first['decode_tps'])} then {fmt(again['decode_tps'])} tok/s, "
                           f"{wl['document'] // 1024}K document {fmt(doc['prefill_tps'], '{:,.0f}')} tok/s, cache "
                           f"check {'ok' if ok else 'FAILED ' + json.dumps(check)}")
                return ph

            async def shift(self, tier, current, ram_gib=None):
                m = await svc.fresh_sample(time.time())
                body, plan = svc.shift_request(tier, m, current=current, ram_gib=ram_gib)
                if body is None:
                    raise RuntimeError(f"{tier} can't be reached in place here "
                                       f"({plan.get('reason') or 'staging-free'})")
                t0 = time.perf_counter()
                reply = await asyncio.to_thread(client.post, "/expert_cache/resize", body, 300)
                s = {"to": tier, "request": body, "reply": reply, "seconds": round(time.perf_counter() - t0, 2),
                     "vram_before_gb": round(m["vram_used"] / GB, 2), "vram_after_gb": round(await self.vram() / GB, 2),
                     "ram_avail_before_gb": round(m["ram_avail"] / GB, 2)}
                p, ram = (reply.get("pools") or [{}])[0], reply.get("ram")
                reused = (f", {ram['reused_slots']} experts' memory taken back from the last shrink"
                          if ram and ram.get("reused_slots") else "")
                if ram and ram.get("growing"):
                    what = f"; RAM cache growing to {ram['target_bytes'] / GIB:.1f} GiB in the background{reused}"
                elif ram and reused:
                    what = f"; RAM cache {ram['ram_slots_before']} -> {ram['ram_slots']} experts{reused}"
                elif ram:
                    what = (f"; RAM cache {ram['ram_slots_before']} -> {ram['ram_slots']} experts "
                            f"({ram['ram_bytes'] / GIB:.1f} GiB, {ram['dropped']} dropped, {ram['moved']} moved, "
                            f"requests held {ram['ms'] / 1e3:.1f} s)")
                else:
                    what = ""
                bench.emit(f"  -> {tier} in {s['seconds']:.1f} s: {p.get('slots_before')} -> {p.get('slots')} experts "
                           f"in VRAM, {p.get('written_back', 0)} written back to RAM; VRAM in use "
                           f"{s['vram_before_gb']:.1f} -> {s['vram_after_gb']:.1f} GB{what}")
                return plan, s

            async def ram(self):
                return await asyncio.to_thread(client.get, "/expert_cache/ram", 10)

            async def wait_ram(self, timeout=300):
                """Until the RAM cache is done growing or freeing; its state then, and how long that took."""
                t0 = time.perf_counter()
                while True:
                    info = await self.ram()
                    if not info.get("growing") and not info.get("freeing"):
                        info["waited_s"] = round(time.perf_counter() - t0, 2)
                        return info
                    if time.perf_counter() - t0 > timeout:
                        raise RuntimeError(f"the RAM cache was still busy after {timeout} s: {info}")
                    await asyncio.sleep(0.5)

            async def memory(self):
                """The engine's RAM against its guard, and Windows' free RAM."""
                ram = (await asyncio.to_thread(client.get, "/memory", 10))["ram"]
                info = await self.ram()
                m = await svc.fresh_sample(time.time())
                if ram.get("limit") is None or info.get("limit_bytes") is None:
                    raise RuntimeError("the engine's RAM guard has no limit (QWEN_RAM_LIMIT_BYTES), so the RAM cache "
                                       "can't be checked against it")
                return {"ram_current_gb": round(ram["used"] / GB, 2),
                        "ram_limit_gb": round(ram["limit"] / GB, 2),
                        "ram_cache_gib": round(info["ram_bytes"] / GIB, 2),
                        "limit_minus_cache_gib": round((info["limit_bytes"] - info["ram_bytes"]) / GIB, 2),
                        "windows_free_gb": round(m["ram_avail"] / GB, 2)}

            async def ram_timeline(self, before_gb):
                """Windows' free RAM after a shrink: WSL hands freed memory back in its own time."""
                pts, t0 = [], time.perf_counter()
                for at_s in wl.get("ram_timeline_s", ()):
                    await asyncio.sleep(max(0.0, at_s - (time.perf_counter() - t0)))
                    m = await svc.fresh_sample(time.time())
                    pts.append({"s": at_s, "ram_avail_gb": round(m["ram_avail"] / GB, 2),
                                "gained_gb": round(m["ram_avail"] / GB - before_gb, 2)})
                if pts:
                    bench.emit("  free RAM (Windows) after the shrink: " + ", ".join(
                        f"{p['gained_gb']:+.1f} GB at {p['s']} s" for p in pts))
                return pts

            async def ram_shift_or_none(self, out, tier, current, ram_gib):
                """A switch that sets the RAM cache, or None (recorded in out) where the engine can't."""
                try:
                    return await self.shift(tier, current, ram_gib=ram_gib)
                except urllib.error.HTTPError as e:
                    with e:
                        err = str((json.loads(e.read() or b"{}") or {}).get("error", ""))
                    if e.code != 400 or not err.startswith("ram_bytes:"):
                        raise
                    out["ram"] = {"unsupported": err[10:].strip()}
                    bench.emit(f"  RAM cache resize not tested: {out['ram']['unsupported']}")
                    return None
        return Kit()

    async def resize_steps(self, client, r, rec, save):
        """Switching full -> lean -> full without a restart, as the manager does when a game needs VRAM (POST
        /expert_cache/resize, with the manager's own request): at each size an answer twice (the second once the
        cache has refilled), a document, and the engine's consistency check (/nv_verify); then quick cycles. Checks
        that every answer matches the first (greedy) and that VRAM is freed and taken back. Then the same with the RAM
        cache (ram_resize_phases)."""
        wl = WORKLOADS["resize"]
        k = self.resize_kit(client, r, wl)
        for text in WARMUP:                                           # not scored
            await k.ask(text)
        out = {"phases": [], "shifts": []}
        out["phases"].append(await k.at("full"))
        rec["resize"] = out
        save()
        full = rec["plan"]
        lean, s = await k.shift("lean", full)
        out["shifts"].append(s)
        out["phases"].append(await k.at("lean"))
        save()
        full, s = await k.shift("full", lean)
        out["shifts"].append(s)
        out["phases"].append(await k.at("full again"))
        save()
        for _ in range(wl["cycles"]):                                 # no requests in between
            lean, s = await k.shift("lean", full)
            out["shifts"].append(s)
            full, s = await k.shift("full", lean)
            out["shifts"].append(s)
        save()
        out["phases"].append(await k.at("full after cycles"))
        save()
        await self.ram_resize_phases(k, out, rec, full, save)
        ref = out["phases"][0]
        grown = [s["vram_after_gb"] for s in out["shifts"] if s["to"] == "full"]
        # greedy answers match only if the engine is deterministic: the first size's two answers show whether it is
        deterministic = ref["text"] == ref["again_text"]
        during = [x for x in (out.get("ram") or {}).get("during_growth", [])]
        out["checks"] = {
            "deterministic": deterministic,
            "same_answers": all(p["text"] == ref["text"] and p["again_text"] == ref["text"] for p in out["phases"])
            and all(d["text"] == ref["text"] for d in during),
            "caches_consistent": all(p["verify_ok"] for p in out["phases"]),
            "vram_freed_gb": round(ref["vram_used_gb"] - out["phases"][1]["vram_used_gb"], 2),
            # what full holds after each regrow, against the first: the arena comes back to its size at load
            "regrown_vs_first_gb": [round(g - out["shifts"][0]["vram_before_gb"], 2) for g in grown],
            "seconds_max": max(s["seconds"] for s in out["shifts"] if not s["reply"].get("ram"))}
        out["checks"].update((out.get("ram") or {}).get("checks") or {})
        rec["decode_runs"] = [ref["answer"]]
        rec["decode_tps_median"] = rec["decode_tps_min"] = ref["answer"]["decode_tps"]
        rec["short_ttft_median_s"] = ref["answer"]["ttft_s"]
        rec["prefill"] = {str(wl["document"]): {"median": ref["document"]["prefill_tps"],
                                                "runs": [ref["document"]["prefill_tps"]]}}
        c = out["checks"]
        bad = [k for k in (("same_answers",) if deterministic else ()) + ("caches_consistent",) if not c[k]]
        if bad:
            rec["error"] = "resize check failed: " + ", ".join(bad)
        same = ("answers identical" if c["same_answers"] else "ANSWERS DIFFER" if deterministic
                else "answers vary run to run here, so not compared")
        self.emit(f"  resize: {same}, caches "
                  f"{'consistent' if c['caches_consistent'] else 'INCONSISTENT'}, {c['vram_freed_gb']:.1f} GB freed "
                  f"for lean, full regrown to within {max(map(abs, c['regrown_vs_first_gb'])):.2f} GB of its first "
                  f"size, slowest switch {c['seconds_max']:.1f} s")
        if c.get("ram_slots_regrown") is not None:
            self.emit(f"  RAM cache: {c['ram_shrunk_gib']:.1f} GiB freed for lean (requests held up to "
                      f"{c['ram_held_s_max']:.1f} s), Windows had {c['ram_windows_gained_gb']:+.1f} GB more free "
                      f"{wl['ram_timeline_s'][-1]} s later; regrown in the background in up to "
                      f"{c['ram_grow_s_max']:.0f} s, "
                      f"{'to its first size' if c['ram_slots_regrown'] else 'SHORT of its first size'}; answers "
                      f"while it grew {', '.join(fmt(d['decode_tps']) for d in during)} tok/s")
        save()

    async def ram_resize_phases(self, k, out, rec, full, save):
        """The RAM cache resized in place as well (live_ram_resize): lean's when RAM runs short, full's again after
        (grown in the background, with an answer while it grows), then quick cycles. Skipped, and recorded, where the
        engine can't (older compiled kernels)."""
        wl, lean_ram, full_ram = WORKLOADS["resize"], self.svc.cfg["lean_ram_gb"], rec["ram_gib"]
        if not full_ram or full_ram <= lean_ram:
            out["ram"] = {"skipped": f"the RAM cache ({full_ram} GiB) is not above lean's ({lean_ram} GiB)"}
            self.emit(f"  RAM cache resize skipped: {out['ram']['skipped']}")
            return
        got = await k.ram_shift_or_none(out, "lean", full, lean_ram)
        if got is None:
            return
        lean, s = got
        first = s["reply"]["ram"]
        out["shifts"].append(s)
        s["ram_timeline"] = await k.ram_timeline(s["ram_avail_before_gb"])
        out["phases"].append(await k.at("lean with lean's RAM cache"))
        save()
        during, grows = [], []

        async def grow_back(current, with_answer):
            plan, s = await k.shift("full", current, ram_gib=full_ram)
            out["shifts"].append(s)
            if with_answer:                          # a request while it grows: it must answer as before
                a = await k.ask(ANSWER.format(t=0))
                a["text"] = a.pop("answer")
                a["growing_after"] = (await k.ram()).get("growing")
                during.append(a)
                self.emit(f"  answer while the RAM cache grows: {fmt(a['decode_tps'])} tok/s, first token after "
                          f"{fmt(a['ttft_s'], '{:.1f}')} s")
            info = await k.wait_ram()
            s["ram_after"] = info
            grows.append(info.get("grow_s") or 0.0)
            self.emit(f"  RAM cache grown to {info['ram_bytes'] / GIB:.1f} GiB in {info.get('grow_s', 0):.1f} s"
                      + (f" (stopped: {info['stopped']})" if info.get("stopped") else ""))
            return plan, info
        full, info = await grow_back(lean, True)
        out["phases"].append(await k.at("full, RAM cache regrown"))
        save()
        for _ in range(wl["ram_cycles"]):                             # no requests in between
            lean, s = await k.shift("lean", full, ram_gib=lean_ram)
            out["shifts"].append(s)
            await k.wait_ram()
            full, info = await grow_back(lean, False)
        out["phases"].append(await k.at("full after RAM cycles"))
        ram_shifts = [x for x in out["shifts"] if x["reply"].get("ram")]
        out["ram"] = {"during_growth": during, "checks": {
            "ram_shrunk_gib": round((first["ram_bytes_before"] - first["ram_bytes"]) / GIB, 2),
            "ram_windows_gained_gb": ram_shifts[0]["ram_timeline"][-1]["gained_gb"] if ram_shifts[0]["ram_timeline"]
            else None,
            "ram_slots_regrown": info["ram_slots"] == first["ram_slots_before"],
            "ram_held_s_max": max(x["seconds"] for x in ram_shifts),
            "ram_grow_s_max": max(grows),
            "ram_ttft_while_growing_s_max": max((d["ttft_s"] for d in during), default=None)}}
        save()

    async def stress_steps(self, client, r, rec, save):
        """The RAM cache's background work under the conditions that could go wrong: requests and a long document
        while it grows, a shrink that cancels a grow, a grow stopped partway (the manager's brake), rapid switching,
        and requests arriving around a shrink and while its memory goes back. After each: the answers against the
        first, the engine's cache check, its RAM against its guard, the limits against the cache, and Windows' free
        RAM; while it grows or frees, how soon a request gets its first token."""
        wl = WORKLOADS["resize-stress"]
        k = self.resize_kit(client, r, wl)
        lean_ram, full_ram = self.svc.cfg["lean_ram_gb"], rec["ram_gib"]
        if not full_ram or full_ram <= lean_ram:
            raise RuntimeError(f"the RAM cache ({full_ram} GiB) is not above lean's ({lean_ram} GiB)")
        for text in WARMUP:                                           # not scored
            await k.ask(text)
        out = {"variations": []}
        rec["stress"] = out
        ref = await k.at("full")
        out["ref"] = ref
        base = await k.memory()
        out["memory_at_start"] = base
        save()
        full = rec["plan"]
        state = {"full": full, "lean": None}

        async def answer():
            a = await k.ask(ANSWER.format(t=0))
            a["text"] = a.pop("answer")
            return a

        async def finish(name, answers, extra=None, ok_extra=True):
            """Close a variation: the checks every one gets, then its own."""
            info = await k.wait_ram()
            check, cache_ok = await k.verify()
            answers = answers + [await answer()]
            mem = await k.memory()
            same = all(a["text"] == ref["text"] for a in answers)
            under_guard = mem["ram_current_gb"] <= mem["ram_limit_gb"]
            # the limits move with the cache: guard minus cache stays where it started (lowering stops 2 GiB above
            # what is in use, and growth is planned in whole chunks)
            limits_follow = abs(mem["limit_minus_cache_gib"] - base["limit_minus_cache_gib"]) <= 2.5
            v = {"name": name, "answers": [{k2: a.get(k2) for k2 in ("decode_tps", "ttft_s")} for a in answers],
                 "same_answers": same, "cache_check": check, "cache_ok": cache_ok, "ram": info, "memory": mem,
                 "under_guard": under_guard, "limits_follow": limits_follow, **(extra or {})}
            v["ok"] = bool(same and cache_ok and under_guard and limits_follow and ok_extra)
            out["variations"].append(v)
            save()
            self.emit(f"  {'ok  ' if v['ok'] else 'FAIL'} {name}: answers {'identical' if same else 'DIFFER'}, cache "
                      f"check {'ok' if cache_ok else 'FAILED'}, RAM cache {info['ram_bytes'] / GIB:.1f} GiB, engine "
                      f"RAM {mem['ram_current_gb']:.1f} of {mem['ram_limit_gb']:.1f} GB allowed, limits "
                      f"{'follow' if limits_follow else 'DRIFTED'} ({mem['limit_minus_cache_gib']:+.1f} vs "
                      f"{base['limit_minus_cache_gib']:+.1f} GiB), Windows {mem['windows_free_gb']:.1f} GB free"
                      + (f"; {extra['note']}" if extra and extra.get("note") else ""))

        async def to(tier, ram_gib):
            plan, s = await k.shift(tier, state["full"] if tier == "lean" else state["lean"], ram_gib=ram_gib)
            state[tier] = plan
            return s

        # 1. requests and a long document while the RAM cache grows
        got = await k.ram_shift_or_none(out, "lean", full, lean_ram)
        if got is None:
            rec["error"] = "the engine can't resize its RAM cache: " + out["ram"]["unsupported"]
            return
        state["lean"] = got[0]
        await k.wait_ram()
        s = await to("full", full_ram)
        a1 = await answer()
        growing = (await k.ram()).get("growing")
        d = await k.document()
        growing_doc = (await k.ram()).get("growing")
        await finish("answers and a 32K document while the RAM cache grows", [a1], {
            "decode_while_growing": a1["decode_tps"], "decode_ref": ref["answer"]["decode_tps"],
            "ttft_while_growing_s": a1["ttft_s"], "ttft_ref_s": ref["answer"]["ttft_s"],
            "still_growing_after_answer": growing, "still_growing_after_document": growing_doc,
            "document_tps_while_growing": d["prefill_tps"],
            "note": f"answer {fmt(a1['decode_tps'])} tok/s, first token after {fmt(a1['ttft_s'], '{:.1f}')} s "
                    f"while growing (vs {fmt(ref['answer']['decode_tps'])}, {fmt(ref['answer']['ttft_s'], '{:.1f}')} "
                    f"s), document {fmt(d['prefill_tps'], '{:,.0f}')} tok/s"
                    f"{' while growing' if growing_doc else ''}"},
            ok_extra=a1["ttft_s"] <= wl["latency_max_s"])

        # 2. a shrink that comes while it grows: the grow stops, the shrink goes through
        await to("lean", lean_ram)
        await k.wait_ram()
        await to("full", full_ram)
        await asyncio.sleep(wl["cancel_after_s"])
        mid = await k.ram()
        s = await to("lean", lean_ram)
        info = await k.wait_ram()
        lean_slots = info["ram_slots"]
        await finish("a shrink while the RAM cache grows", [], {
            "growing_when_cancelled": mid.get("growing"), "slots_when_cancelled": mid.get("ram_slots"),
            "shrink_s": s["seconds"], "note": f"cancelled at {mid.get('ram_slots')} experts, shrink took "
                                             f"{s['seconds']:.1f} s"},
            ok_extra=info["ram_slots"] == lean_slots and not info.get("growing"))

        # 3. a grow stopped partway, as the manager's brake does when RAM gets short; then grown the rest of the way
        await to("full", full_ram)
        await asyncio.sleep(wl["stop_after_s"])
        stopped = await asyncio.to_thread(client.post, "/expert_cache/ram/stop", {}, 60)
        partial = lean_slots < stopped["ram_slots"] < stopped["target_slots"]
        a3 = await answer()
        await k.shift("full", state["full"], ram_gib=full_ram)        # same profile: the RAM cache only
        info = await k.wait_ram()
        await finish("a grow stopped partway, then finished", [a3], {
            "stopped_at": stopped["ram_slots"], "target": stopped["target_slots"],
            "stop_reason": stopped.get("stopped"),
            "partial": partial, "note": f"stopped at {stopped['ram_slots']} of {stopped['target_slots']} experts "
                                        f"({stopped.get('stopped')}), then grew to {info['ram_slots']}"},
            ok_extra=not stopped.get("growing") and info["ram_slots"] == stopped["target_slots"])

        # 4. rapid switching, not waiting for anything: back to full right after a shrink takes back the memory it
        # has not freed yet
        backs = []
        for _ in range(wl["toggles"]):
            await to("lean", lean_ram)
            backs.append(await to("full", full_ram))
        info = await k.wait_ram()
        reused = [(b["reply"].get("ram") or {}).get("reused_slots") or 0 for b in backs]
        await finish(f"{wl['toggles']} quick switches back and forth", [], {
            "reused_slots": reused, "back_to_full_s": [b["seconds"] for b in backs],
            "note": f"back to full in {max(b['seconds'] for b in backs):.1f} s at most, taking back "
                    f"{', '.join(map(str, reused))} experts' memory"},
            ok_extra=info["ram_slots"] == stopped["target_slots"] and reused[0] > 0)

        # 5. requests around a shrink: one running when it comes (it waits for it), one arriving while it runs
        running = asyncio.create_task(answer())
        await asyncio.sleep(0.3)
        shrink = asyncio.create_task(to("lean", lean_ram))
        await asyncio.sleep(0.5)
        arriving = asyncio.create_task(answer())
        a_run, s, a_arr = await running, await shrink, await arriving
        freeing = (await k.ram()).get("freeing")
        a_free = await answer()                                       # while the freed memory goes back
        await finish("requests around a shrink, and while its memory goes back", [a_run, a_arr, a_free], {
            "shrink_s": s["seconds"], "ttft_arriving_s": a_arr["ttft_s"], "ttft_ref_s": ref["answer"]["ttft_s"],
            "freeing_at_answer": freeing, "ttft_while_freeing_s": a_free["ttft_s"],
            "decode_while_freeing": a_free["decode_tps"],
            "note": f"the shrink took {s['seconds']:.1f} s with a request running; one arriving meanwhile had its "
                    f"first token after {fmt(a_arr['ttft_s'], '{:.1f}')} s, one "
                    f"{'while the memory went back' if freeing else 'right after (already freed)'} after "
                    f"{fmt(a_free['ttft_s'], '{:.1f}')} s"},
            ok_extra=a_free["ttft_s"] <= wl["latency_max_s"])

        # back to full, its RAM cache grown
        await to("full", full_ram)
        await k.wait_ram()
        bad = [v["name"] for v in out["variations"] if not v["ok"]]
        mins = [v["memory"]["windows_free_gb"] for v in out["variations"]]
        out["checks"] = {"all_ok": not bad, "failed": bad, "windows_free_min_gb": min(mins),
                         "engine_ram_max_gb": max(v["memory"]["ram_current_gb"] for v in out["variations"])}
        rec["decode_runs"] = [ref["answer"]]
        rec["decode_tps_median"] = rec["decode_tps_min"] = ref["answer"]["decode_tps"]
        rec["short_ttft_median_s"] = ref["answer"]["ttft_s"]
        if bad:
            rec["error"] = "stress check failed: " + "; ".join(bad)
        self.emit(f"  stress: {len(out['variations']) - len(bad)} of {len(out['variations'])} variations ok"
                  + (f", FAILED: {'; '.join(bad)}" if bad else "")
                  + f"; Windows had at least {min(mins):.1f} GB free at the checks")
        save()
