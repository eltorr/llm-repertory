"""VRAM and RAM sizing for Qwen3.8-Flash-Next on this engine, as plain arithmetic (no torch, no GPU).

One source of truth for run.sh (`python3 engines/exl3_offload/sizing.py env ...`) and the Repertory manager.

Where VRAM goes (code refs: glm53-flash-offload/glm53/expert_cache.py, nv2.py; exllamav3 cache/quant.py):
  fixed     weights that stay on the GPU (attention, shared experts, routers, lm_head, per-expert scales), KV cache,
            DeltaNet state, CUDA context/kernels/graphs. Grows with context only through the KV cache.
  cache     expert cache: S slots of one expert record each (rec bytes), in 32-slot chunks.
  reserve   VRAM left free after the cache is allocated (GLM53_EC_RESERVE_GB).
  prefill   two staging buffers of a whole layer (512 records each) plus activations for one chunk. They come out of
            the reserve plus cache chunks the elastic cache lends back during prefill, so they are not extra.
A plan fits when fixed + cache + reserve <= budget and cache + reserve - 0.35 GB >= prefill(chunk).

Calibration (research/vram-scaling): VRAM used outside the expert cache measured 9.0 GB (3.05 bpw) and 9.44 GB
(4.05 bpw) at 262,144 context and 6.85 GB at 64K (3.05), i.e. 6.15 / 6.62 GB plus the KV cache's 10,752 bytes per
token; an 8 GB-class run used 6.39 GB at 16K. Activations per chunk token come from RTX 4090 runs published with
MaxDevv/lean-qwenflash, which this project began as a fork of (a 24,576-token chunk fit with a 16 GB cap:
https://github.com/MaxDevv/lean-qwenflash/tree/fa9a772d9aa9b94a5653a2c6880eeb97af0d2e9d/research/4090-long). The
reserves include ~0.3 GB of margin found by running long prompts on emulated 8-16 GB cards.

baseline() turns a card and an amount of RAM into the starting configuration (quant, context, RAM cache, the
manager's RAM thresholds); `python3 engines/exl3_offload/sizing.py presets` prints the README's preset and
measurement tables from it and from MEASURED.
"""
import argparse
import json
import math
import sys

GB = 1e9
GIB = 1024 ** 3
EXPERTS_PER_LAYER = 512
MOE_LAYERS = 48
ACT_BYTES_PER_TOKEN = 0.21e6          # the RTX 4090 runs above: a 24,576-token chunk fit, 32,768 did not
PREFILL_EXTRA = 0.3 * GB
FRAGMENTATION = 0.5 * GB   # torch held 0.8-1 GB more than it used when staging was allocated on small cards
ELASTIC_SLACK = 0.35 * GB
CHUNKS = (24576, 16384, 8192, 4096, 2048)   # below 2,048 staging-free 1,536-token chunks do better
# The VRAM and RAM caches together hold this share of the experts; the rest, picked rarely, comes from NVMe. At 4.05 bpw
# on the whole card, 89% and 100% ran alike (research/vram-scaling-torch-2.15: 60.9 and 61.0 tok/s, 1% of picks from
# NVMe in both), and the last 11% took 6 GiB of RAM.
RAM_COVERAGE = 0.9
# Forwards with fewer expert picks per layer than this skip prefill staging, which costs ~5.5 s whatever the prompt's
# size (research/vram-scaling-torch-2.15). nv2's pick buffer holds 16,384: 1,638 tokens at 10 experts each.
STAGE_MIN_PICKS = 16384
NO_STAGE_CHUNK = 1536           # without staging, nv2's pick buffer limits a chunk to 1,638 tokens
KV_BYTES_PER_TOKEN = {2: 7_680, 4: 10_752, 8: 16_896, 16: 28_416}

QUANTS = {
    # rec: one expert record (gate + up + down trellis, 256-aligned); fixed0: non-expert VRAM at zero context
    "3.05": {"rec": 1_843_200, "fixed0": 6.15 * GB, "model_dir": "model", "store": "store/qwen.bin",
             "branch": "3.05bpw_h5_ng5", "download_gb": 85, "store_gb": 47, "label": "3.05 bpw (fastest)"},
    "4.05": {"rec": 2_457_600, "fixed0": 6.62 * GB, "model_dir": "model-4.05", "store": "store/qwen405.bin",
             "branch": "4.05bpw_h6_ng6", "download_gb": 107, "store_gb": 62, "label": "4.05 bpw (higher fidelity)"},
}


def expert_bytes(quant):
    """Every routed expert of the model (what VRAM cache + RAM tier + NVMe hold between them)."""
    return QUANTS[quant]["rec"] * EXPERTS_PER_LAYER * MOE_LAYERS


def ram_target(quant, vram_cache):
    """RAM cache that, with `vram_cache` bytes of experts in VRAM, holds RAM_COVERAGE of the model's experts."""
    return max(0.0, RAM_COVERAGE * expert_bytes(quant) - vram_cache)


def fixed_bytes(quant, context, kv_bits=4):
    return QUANTS[quant]["fixed0"] + KV_BYTES_PER_TOKEN[kv_bits] * context


def stage_bytes(quant):
    return 2 * EXPERTS_PER_LAYER * QUANTS[quant]["rec"]


def prefill_bytes(quant, chunk, staging=True):
    return (stage_bytes(quant) if staging else 0) + ACT_BYTES_PER_TOKEN * chunk + PREFILL_EXTRA


SMALL_CARD = 16 * GB
# the reserve and elastic share engine_env sets on budgets of 16 GB and more (tuned on the RTX 5090)
ENGINE_RESERVE_GIB = 1.7
ENGINE_ELASTIC_GIB = 6.0


def reserve_bytes(budget):
    """Free VRAM the engine keeps after the cache. Below 16 GB: 0.9-0.96 GB (0.4-0.87 GB let long prompts overrun
    emulated 12-16 GB cards by 20-80 MB). From 16 GB: 1.7 GiB."""
    if budget >= SMALL_CARD:
        return ENGINE_RESERVE_GIB * GIB
    return max(0.9 * GB, 0.06 * budget)


def nostage_reserve(quant, chunk):
    return max(0.7 * GB, prefill_bytes(quant, chunk, staging=False) + 0.2 * GB)


def plan(quant, budget, context=262144, kv_bits=4, chunk=None, max_slots=None):
    """Best configuration for `budget` bytes of VRAM for this process (excluding other apps).

    Returns a dict with ok, slots, chunk, staging, reserve and the byte breakdown; ok=False carries a reason and
    the smallest budget that would work at this context."""
    q = QUANTS[quant]
    rec = q["rec"]
    fixed = fixed_bytes(quant, context, kv_bits)
    reserve = reserve_bytes(budget)
    cache = max(0.0, budget - fixed - reserve)
    slots = int(cache // rec) // 32 * 32
    if max_slots is not None:
        slots = min(slots, int(max_slots))
    cache = slots * rec
    out = {"quant": quant, "context": context, "kv_bits": kv_bits, "budget": budget, "fixed": fixed,
           "reserve": reserve, "cache": cache, "slots": slots, "staging": True, "chunk": None, "ok": False,
           "experimental": budget < 11 * GB, "ram_for_all_experts": max(0.0, expert_bytes(quant) - cache),
           "ram_target": ram_target(quant, cache),
           "min_budget": min_budget(quant, context, kv_bits)}
    room = cache + reserve - ELASTIC_SLACK
    candidates = [chunk] if chunk else CHUNKS
    for c in candidates:
        if room >= prefill_bytes(quant, c) + FRAGMENTATION:
            out.update(chunk=c, ok=True)
            return out
    # staging-free fallback: every miss is read zero-copy from pinned RAM (supported by nv2, not yet measured)
    c = min(chunk or NO_STAGE_CHUNK, NO_STAGE_CHUNK)
    r = nostage_reserve(quant, c)
    if budget >= fixed + r:
        cache = max(0.0, budget - fixed - r)
        slots = int(cache // rec) // 32 * 32
        out.update(chunk=c, staging=False, ok=True, experimental=True, reserve=r, slots=slots, cache=slots * rec,
                   ram_for_all_experts=max(0.0, expert_bytes(quant) - slots * rec),
                   ram_target=ram_target(quant, slots * rec))
        return out
    out["reason"] = (f"{budget / GB:.1f} GB is not enough at {context // 1024}K context: the model needs about "
                     f"{fixed / GB:.1f} GB before any expert cache, and {out['min_budget'] / GB:.1f} GB in total")
    return out


def min_budget(quant, context, kv_bits=4):
    """Smallest budget with staging and the smallest chunk."""
    fixed = fixed_bytes(quant, context, kv_bits)
    need_cache_plus_reserve = prefill_bytes(quant, CHUNKS[-1]) + FRAGMENTATION + ELASTIC_SLACK
    return fixed + max(need_cache_plus_reserve, 0.7 * GB + 32 * QUANTS[quant]["rec"])


def max_context(quant, budget, kv_bits=4, step=16384, staging=True):
    """Largest context (multiple of `step`, at most 262,144) that still leaves a workable plan, with staged
    prefill (or, with staging=False, in either prefill mode)."""
    best = 0
    for ctx in range(step, 262144 + 1, step):
        p = plan(quant, budget, ctx, kv_bits)
        if p["ok"] and (p["staging"] or not staging):
            best = ctx
    return best


def engine_env(p, cap_slots=False):
    """Environment for the engine that realizes plan p (run.sh / the manager add the device-wide VRAM limit).

    cap_slots pins the cache size (a budgeted run must not grow if other apps free VRAM later); a full run leaves
    it to the engine, which measures free VRAM itself and gets a slightly bigger cache than this model predicts.
    The reserve and the elastic share are always set, so the plan doesn't depend on the engine's own defaults."""
    env = {"QWEN_CHUNK": str(p["chunk"])}
    if p["staging"]:
        env["GLM53_EC_STAGE_MIN"] = str(STAGE_MIN_PICKS)
    if cap_slots:
        env["GLM53_EC_MAX_SLOTS"] = str(p["slots"])
    if not p["staging"]:
        env.update({"GLM53_EC_STAGE_GB": "0", "GLM53_EC_ELASTIC_GB": "0",
                    "GLM53_EC_RESERVE_GB": f"{p['reserve'] / GIB:.2f}"})
    elif p["budget"] < SMALL_CARD:
        # small cards: a smaller reserve, and the whole cache may be lent back for prefill if needed (the engine's
        # own activation estimate stays: 0.25 MB/token let a 4.05 prefill overrun an emulated 16 GB card)
        env.update({"GLM53_EC_RESERVE_GB": f"{p['reserve'] / GIB:.2f}",
                    "GLM53_EC_ELASTIC_GB": f"{max(6.0, p['cache'] / GIB + 1):.1f}"})
    else:
        env.update({"GLM53_EC_RESERVE_GB": f"{ENGINE_RESERVE_GIB:.2f}",
                    "GLM53_EC_ELASTIC_GB": f"{ENGINE_ELASTIC_GIB:.1f}"})
    return env


def live_resize_args(p, cap_slots=False):
    """What the engine's live resize (server.py, POST /expert_cache/resize) needs to run plan p without a
    restart: the settings engine_env would have started it with. The caller adds the VRAM limits. Only plans with
    staging switch this way; a staging-free plan allocates differently and needs a restart."""
    if not p["staging"]:
        raise ValueError("a staging-free plan needs a restart")
    env = engine_env(p, cap_slots)
    return {"chunk": p["chunk"],
            "max_slots": int(env["GLM53_EC_MAX_SLOTS"]) if cap_slots else None,
            "reserve_bytes": int(float(env.get("GLM53_EC_RESERVE_GB", ENGINE_RESERVE_GIB)) * GIB),
            "elastic_bytes": int(float(env.get("GLM53_EC_ELASTIC_GB", ENGINE_ELASTIC_GIB)) * GIB)}


# ---------------------------------------------------------------------------------------------------------------
# Machine baseline (README "Presets"): what the manager starts from on a given card and amount of RAM
# ---------------------------------------------------------------------------------------------------------------
RAM_OVERHEAD = {"linux": 6.5 * GB, "windows-wsl": 9.0 * GB}  # host RAM the server takes beyond its RAM cache
OS_IDLE = {"linux": 3.0 * GB, "windows-wsl": 8.0 * GB}       # desktop and everyday apps (planning only)
DESKTOP_VRAM = 1.5 * GB                                       # a desktop with a browser open, on the model's card
PLATFORMS = {"linux": "Linux", "windows-wsl": "Windows (WSL2)"}


def headroom_bytes(vram_total, display=True):
    """VRAM left free in full mode so the desktop and other apps can grow: ~10% of the card, 1.2-2.4 GB. A card
    that drives no display keeps only 0.6 GB on small cards (an 8 GB card needs every bit; measured at 8 GB). On the
    32 GB card, full mode ran stably for days with ~2.1 GB actually free (3.2 GB set, ~1.1 GB of it unseen by CUDA
    under WSL2 until the engine counted that)."""
    return min(2.4 * GB, max((1.2 if display else 0.6) * GB, 0.1 * vram_total))


def card_budget(vram_total, display=True):
    """VRAM the model gets on an otherwise idle card."""
    return vram_total - (DESKTOP_VRAM if display else 0.1 * GB) - headroom_bytes(vram_total, display)


def wsl_memory_bytes(ram_total):
    """The WSL memory limit setup writes: most of the RAM, keeping 12.5% (at least 8 GB) for Windows."""
    return max(8 * GB, ram_total - max(8 * GB, 0.125 * ram_total))


def ram_policy(ram_total):
    """The manager's RAM thresholds, scaled to the machine (the defaults suit 48 GB and more)."""
    t = ram_total / GB
    if t >= 48:
        return {"ram_reserve_gb": 6.0, "ram_lean_gb": 6.0, "ram_pause_gb": 3.0, "lean_ram_gb": 8}
    if t >= 24:
        return {"ram_reserve_gb": 4.0, "ram_lean_gb": 4.0, "ram_pause_gb": 2.0, "lean_ram_gb": 6}
    return {"ram_reserve_gb": 2.0, "ram_lean_gb": 2.0, "ram_pause_gb": 1.0, "lean_ram_gb": 4}


def ram_cache_gib(need, ram_avail, policy, overhead, cap_gib=None):
    """RAM cache for a full start, as the manager sizes it: `need` bytes (the plan's ram_target), within what
    the host can spare while free RAM stays clear of the scale-back threshold; never below the lean minimum, unless
    cap_gib is lower (only max_ram_cache_gib can be: learned caps stop at the lean minimum).
    ram_avail counts the model's own RAM if it is running."""
    want = math.ceil(need / GIB)
    floor = policy["lean_ram_gb"]
    if cap_gib:
        want = min(want, int(cap_gib))
        floor = min(floor, int(cap_gib))
    keep = max(policy["ram_reserve_gb"], policy["ram_lean_gb"] + 3) * GB
    spare = (ram_avail - keep - overhead) // GIB
    return int(max(floor, min(want, spare)))


def nostage_min_budget(quant, context, kv_bits=4):
    """Smallest budget at all: staging-free prefill and an empty expert cache."""
    return fixed_bytes(quant, context, kv_bits) + nostage_reserve(quant, NO_STAGE_CHUNK)


def best_context_plan(quant, budget, kv_bits=4):
    """The largest context with staged prefill. Below 64K, a staging-free 64K or 32K window if it still caches 512+
    experts (12 GB cards: agents need the window more than the faster prefill); 16K staging-free as the last resort
    (8 GB cards)."""
    staged = max_context(quant, budget, kv_bits)
    if staged >= 65536:
        return plan(quant, budget, staged, kv_bits)
    for ctx in (65536, 32768):
        if ctx <= staged:
            break
        p = plan(quant, budget, ctx, kv_bits)
        if p["ok"] and p["slots"] >= 512:
            return p
    p = plan(quant, budget, staged or 16384, kv_bits)
    return p if p["ok"] else None


def _candidate(quant, budget, ram_total, platform, kv_bits):
    p = best_context_plan(quant, budget, kv_bits)
    if p is None:
        return None
    policy = ram_policy(ram_total)
    avail = ram_total - OS_IDLE[platform]
    cache = ram_cache_gib(p["ram_target"], avail, policy, RAM_OVERHEAD[platform])
    if platform == "windows-wsl":
        cache = min(cache, int((wsl_memory_bytes(ram_total) - RAM_OVERHEAD[platform]) // GIB))
    ram_used = cache * GIB + RAM_OVERHEAD[platform]
    in_memory = min(1.0, (p["cache"] + cache * GIB) / expert_bytes(quant))
    return {"quant": quant, "context": p["context"], "chunk": p["chunk"], "staging": p["staging"],
            "slots": p["slots"], "ram_cache_gib": cache, "in_memory": in_memory,
            "ram_fits": ram_used <= avail - policy["ram_pause_gb"] * GB, "policy": policy}


def baseline(vram_total, ram_total, display=True, platform="linux", kv_bits=4):
    """Starting configuration for a machine: quant, context, RAM cache and the manager's RAM thresholds.

    4.05 bpw (higher fidelity) where it keeps 256K context, full-size prefill chunks and nearly every expert in VRAM
    or RAM. Otherwise 3.05 bpw: its experts are 25% smaller, so it is faster at every card size
    (research/vram-scaling), at the largest context that keeps prefill staging, or staging-free at 16K on 8 GB
    cards. `settings` holds the manager settings that realize it."""
    budget = card_budget(vram_total, display)
    out = {"vram_total": vram_total, "ram_total": ram_total, "display": display, "platform": platform,
           "budget": budget, "ok": False, "measured": []}
    hi = _candidate("4.05", budget, ram_total, platform, kv_bits)
    lo = _candidate("3.05", budget, ram_total, platform, kv_bits)
    if lo is None:
        out["reason"] = (f"{vram_total / GIB:.0f} GB of VRAM is too little{' with a display attached' * display}: "
                         f"about {budget / GB:.1f} GB would be left for the model, which needs "
                         f"{nostage_min_budget('3.05', 16384, kv_bits) / GB:.1f} GB at 16K context")
        return out
    if not lo["ram_fits"]:
        out["reason"] = (f"{ram_total / GIB:.0f} GB of RAM is too little on {PLATFORMS[platform]}: the model takes "
                         f"about {(lo['ram_cache_gib'] * GIB + RAM_OVERHEAD[platform]) / GB:.0f} GB next to the "
                         f"system's own use")
        return out
    best = hi if (hi and hi["ram_fits"] and hi["context"] == 262144 and hi["staging"] and hi["chunk"] >= 16384
                  and hi["in_memory"] >= RAM_COVERAGE - 0.005) else lo
    alt = lo if best is hi else hi
    out.update(best, ok=True, experimental=budget < 11 * GB,
               alternative=({k: alt[k] for k in ("quant", "context", "chunk", "staging", "ram_cache_gib",
                                                   "in_memory")} if alt else None),
               settings={"quant": best["quant"], "context": best["context"], **best["policy"]})
    out.pop("policy")
    out.pop("ram_fits")
    out["measured"] = measured_for(out)
    return out


def measured_for(b):
    """Measurements of this configuration: same card class, quant, context and prefill mode, and a RAM cache within
    25% (or both holding every expert)."""
    def same_ram(m):
        r, mr = b["ram_cache_gib"], m["ram_cache_gib"]
        return 0.8 * r <= mr <= 1.25 * r or (b["in_memory"] >= 0.97 and mr >= 0.8 * r)
    found = [m for m in MEASURED if m["card"] == round(b["vram_total"] / GIB)
             and (m.get("display", True) == b["display"] or m["card"] > 8)      # a display only matters on 8 GB
             and m["quant"] == b["quant"] and m["context"] == b["context"] and m.get("staging", True) == b["staging"]
             and same_ram(m)]
    # a real card of that size first, then the closest RAM cache
    return sorted(found, key=lambda m: (m.get("capped", False), abs(m["ram_cache_gib"] - b["ram_cache_gib"])))


# Measurements behind the presets. card: VRAM class in GiB; ram: the machine's RAM in GiB; ram_cache_gib: the RAM
# cache the run had; capped: the engine was held to that card size on a bigger card (VRAM capacity only: a real card
# of that size has less bandwidth). Add rows as real cards and other RAM sizes are measured (`repertory bench`,
# then `repertory bench --sizing FOLDER` prints rows in this format).
# prefill: the long retrieval document (up to 100K tokens); decode: short answers, thinking off.
# The 4.05 whole-card 34 GiB row is from the report's 2026-11-06 follow-up (short prompts skip staging); the others
# are from before it, so their prefill is low.
_R5090_FULL = {"ram": 64, "platform": "windows-wsl", "gpu": "RTX 5090", "source": "research/vram-scaling-torch-2.15"}
_R5090 = dict(_R5090_FULL, capped=True)
MEASURED = [
    dict(_R5090_FULL, card=32, quant="3.05", context=262144, ram_cache_gib=26, decode=72.1, prefill=4668),
    dict(_R5090_FULL, card=32, quant="4.05", context=262144, ram_cache_gib=34, decode=59.7, prefill=5312),
    dict(_R5090_FULL, card=32, quant="4.05", context=262144, ram_cache_gib=40, decode=61.0, prefill=4233),
    dict(_R5090, card=32, quant="3.05", context=262144, ram_cache_gib=38, decode=74.4, prefill=4722),
    dict(_R5090, card=24, quant="3.05", context=262144, ram_cache_gib=38, decode=62.5, prefill=4628),
    dict(_R5090, card=16, quant="3.05", context=262144, ram_cache_gib=38, decode=41.9, prefill=2684),
    dict(_R5090, card=12, quant="3.05", context=65536, ram_cache_gib=38, decode=36.1, prefill=688),
    dict(_R5090, card=12, quant="3.05", context=131072, staging=False, ram_cache_gib=38, decode=34.4, prefill=1933),
    dict(_R5090, card=8, display=False, quant="3.05", context=16384, staging=False, ram_cache_gib=38, decode=26.4,
         prefill=1490),
    dict(_R5090, card=8, quant="3.05", context=16384, staging=False, ram_cache_gib=38, decode=20.3, prefill=1637),
    dict(_R5090, card=32, quant="4.05", context=262144, ram_cache_gib=38, decode=63.0, prefill=4296),
    dict(_R5090, card=24, quant="4.05", context=262144, ram_cache_gib=38, decode=51.4, prefill=4252),
    dict(_R5090, card=16, quant="4.05", context=262144, ram_cache_gib=38, decode=34.5, prefill=1183),
    dict(_R5090, card=12, quant="4.05", context=32768, staging=False, ram_cache_gib=38, decode=29.6, prefill=1104),
]
PRESET_CARDS = ((8, False), (8, True), (12, True), (16, True), (24, True), (32, True), (48, True))
PRESET_RAM = (16, 32, 64, 128)


def card_bytes(gib):
    """A card sold as `gib` GB, as NVML reports it (e.g. 24,564 MiB on a 24 GB card)."""
    return gib * GIB * 0.995


def ram_bytes(gib):
    """Installed RAM sold as `gib` GB, as the OS reports it."""
    return gib * GIB * 0.98


def _ctx(c):
    return f"{c // 1024}K"


def _card_name(gib, display):
    if gib > 8:
        return f"{gib} GB"
    return f"{gib} GB, {'with' if display else 'no'} display"


def measured_on(m):
    return f"{m['gpu']}" + (f" capped to {m['card']} GB" if m.get("capped") else "")


def _tested(found, ram):
    """A preset's test status for the matrix: the measured speed and where, or not tested."""
    if not found:
        return "not tested yet"
    m = found[0]
    return (f"measured {m['decode']} tok/s ({measured_on(m)}"
            + ("" if m["ram"] == ram else f", {m['ram']} GB RAM") + ")")


def presets_table(platform="linux"):
    """The README's preset matrix (between <!-- presets:begin/end -->), generated from baseline() and MEASURED."""
    lines = ["| Card (VRAM) | " + " | ".join(f"{r} GB RAM" for r in PRESET_RAM) + " |",
             "|---" * (len(PRESET_RAM) + 1) + "|"]
    for gib, display in PRESET_CARDS:
        cells = []
        for ram in PRESET_RAM:
            b = baseline(card_bytes(gib), ram_bytes(ram), display, platform)
            if not b["ok"]:
                cells.append("not enough VRAM" if "VRAM" in b["reason"] else "not enough RAM")
                continue
            mem = "all in memory" if b["in_memory"] >= 0.97 else f"{b['in_memory']:.0%} in memory"
            stage = "" if b["staging"] else ", no staging"
            cells.append(f"**{b['quant']}** · {_ctx(b['context'])}{stage}<br>{b['ram_cache_gib']} GiB RAM cache, "
                         f"{mem}<br>{_tested(b['measured'], ram)}")
        lines.append(f"| **{_card_name(gib, display)}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def measured_table(link_prefix=""):
    """The README's measurement chart (between <!-- measured:begin/end -->), from MEASURED."""
    lines = ["| Card (VRAM) | Quant | Context | RAM cache | Prefill | Decode | Measured on |",
             "|---" * 7 + "|"]
    for m in sorted(MEASURED, key=lambda m: (m["quant"], -m["card"], m.get("capped", False), m["context"],
                                             m["ram_cache_gib"])):
        ctx = _ctx(m["context"]) + ("" if m.get("staging", True) else ", no staging")
        lines.append(f"| **{_card_name(m['card'], m.get('display', True))}** | {m['quant']} | {ctx} | "
                     f"{m['ram_cache_gib']} GiB | {m['prefill']:,} tok/s | {m['decode']} tok/s | "
                     f"[{measured_on(m)}]({link_prefix}{m['source']}/REPORT.md), {m['ram']} GB RAM, "
                     f"{PLATFORMS[m['platform']]} |")
    return "\n".join(lines)


def measured_summary():
    """The README's short version (between <!-- measured-summary:begin/end -->): the fastest measured setup for each
    card size and quant. The full chart is docs/measured.md."""
    best = {}
    for m in MEASURED:
        key = (m["card"], m.get("display", True), m["quant"])
        if key not in best or m["decode"] > best[key]["decode"]:
            best[key] = m
    lines = ["| Card (VRAM) | Quant | Context | Decode | Prefill |", "|---" * 5 + "|"]
    for (card, display, _), m in sorted(best.items(), key=lambda kv: (-kv[0][0], kv[0][1], kv[0][2])):
        ctx = _ctx(m["context"]) + ("" if m.get("staging", True) else ", no staging")
        lines.append(f"| {_card_name(card, display)} | {m['quant']} bpw | {ctx} | {m['decode']} tok/s | "
                     f"{m['prefill']:,} tok/s |")
    return "\n".join(lines)


def presets_markdown(platform="linux"):
    return presets_table(platform) + "\n\n" + measured_table()


def update_readme(path):
    """Rewrite the generated tables in a doc (sizing.py presets --write README.md, --write docs/measured.md): every
    block whose <!-- NAME:begin/end --> markers the file has."""
    import re
    text, found = open(path, encoding="utf-8").read(), 0
    import os
    up = os.path.relpath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         os.path.dirname(os.path.abspath(path))).replace(os.sep, "/")
    prefix = "" if up == "." else up + "/"          # links to research/ from where the doc is
    for name, body in (("presets", presets_table()), ("measured", measured_table(prefix)),
                       ("measured-summary", measured_summary())):
        text, n = re.subn(rf"(<!-- {name}:begin -->\n).*?(\n<!-- {name}:end -->)",
                          lambda m: m.group(1) + body + m.group(2), text, flags=re.S)  # noqa: B023 (used at once)
        found += n
    if not found:
        raise SystemExit(f"{path}: no generated-table markers")
    open(path, "w", encoding="utf-8", newline="\n").write(text)


def describe(b):
    if not b["ok"]:
        return f"not supported: {b['reason']}"
    s = (f"{b['quant']} bpw, {_ctx(b['context'])} context, prefill chunk {b['chunk']:,}"
         f"{'' if b['staging'] else ' (no staging)'}, {b['slots']:,} experts in VRAM, {b['ram_cache_gib']} GiB RAM "
         f"cache: {b['in_memory']:.0%} of experts in memory")
    for m in b["measured"]:
        s += (f"\n  measured: {m['decode']} tok/s decode, {m['prefill']:,} tok/s prefill ({measured_on(m)}, "
              f"{m['ram']} GB RAM)")
    if b.get("alternative"):
        a = b["alternative"]
        s += (f"\n  alternative: {a['quant']} bpw at {_ctx(a['context'])}, {a['ram_cache_gib']} GiB RAM cache, "
              f"{a['in_memory']:.0%} of experts in memory")
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "env"):
        s = sub.add_parser(name)
        s.add_argument("--quant", default="3.05", choices=sorted(QUANTS))
        s.add_argument("--budget-gb", type=float, required=True, help="VRAM for this process, GB (1e9)")
        s.add_argument("--context", type=int, default=262144)
        s.add_argument("--kv-bits", type=int, default=4, choices=sorted(KV_BYTES_PER_TOKEN))
        s.add_argument("--chunk", type=int)
    s = sub.add_parser("baseline", help="starting configuration for a card and an amount of RAM")
    s.add_argument("--vram-gb", type=float, required=True, help="the card's VRAM, GB as sold (e.g. 24)")
    s.add_argument("--ram-gb", type=float, required=True, help="installed RAM, GB as sold (e.g. 64)")
    s.add_argument("--no-display", action="store_true", help="the card drives no monitor")
    s.add_argument("--platform", default="linux", choices=sorted(PLATFORMS))
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("presets", help="the README's preset and measurement tables (markdown)")
    s.add_argument("--platform", default="linux", choices=sorted(PLATFORMS))
    s.add_argument("--write", metavar="README", help="update the tables in this file instead of printing them")
    a = ap.parse_args(argv)
    if (sys.stdout.encoding or "").lower().replace("-", "") != "utf8":
        sys.stdout.reconfigure(encoding="utf-8")   # the tables use ✅ / ⚪; a Windows pipe defaults to cp1252
    if a.cmd == "presets":
        if a.write:
            update_readme(a.write)
        else:
            print(presets_markdown(a.platform))
        return 0
    if a.cmd == "baseline":
        b = baseline(card_bytes(a.vram_gb), ram_bytes(a.ram_gb), not a.no_display, a.platform)
        print(json.dumps(b, indent=2) if a.json else describe(b))
        return 0 if b["ok"] else 3
    p = plan(a.quant, a.budget_gb * GB, a.context, a.kv_bits, a.chunk)
    if a.cmd == "plan":
        big = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 1e6  # noqa: E731
        print(json.dumps({k + ("_gb" if big(v) else ""): (round(v / GB, 2) if big(v) else v) for k, v in p.items()},
                         indent=2))
    else:
        if not p["ok"]:
            print(f"echo 'sizing: {p['reason']}' >&2; exit 3")
            return 3
        for k, v in engine_env(p).items():
            print(f"export {k}={v}")
        print(f"QWEN_RAM_ALL_GIB={int(p['ram_target'] // GIB) + 1}")
    return 0 if p["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
