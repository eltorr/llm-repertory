"""The results log, and the benchmark page made from it (docs/benchmarks.md): speed and quality by GPU and preset.

    python -m repertory results add PATH...   log benchmark runs (a run's JSON, or a bench/<date-time>/ folder) and
                                              head-to-head results (research/head-to-head/results/LABEL.json)
    python -m repertory results page          write docs/benchmarks.md from the log

The log is two files in results/: speed.jsonl, a line a benchmark run, and quality.jsonl, a line a head-to-head run.
A line holds what the page shows and what it was measured on: the GPU, the preset (the card size the model had, as
the README's presets name them, and the PC's RAM), the platform, the model and its settings, and the stack. No host
names, prompts or answers. Adding a run again replaces its line. Runs that failed or were cut short aren't logged.
"""
import json
import math
import re
from pathlib import Path

from . import bench, catalog
from engines.exl3_offload import sizing

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "results"
PAGE = ROOT / "docs" / "benchmarks.md"
SPEED, QUALITY = "speed.jsonl", "quality.jsonl"
H2H_TASKS = ("calc", "sched", "cache", "paths", "puzzle", "ticket", "log")     # research/head-to-head/tasks.py
WORKLOADS = {"standard": 0, "quick": 1, "reference": 2}                         # the page's order: the benchmark first


# ------------------------------------------------------------------------------------------------------- facts
def gpu_name(full):
    """'NVIDIA GeForce RTX 5090' -> 'RTX 5090'."""
    return re.sub(r"^(NVIDIA\s+)?(GeForce\s+)?", "", full or "").strip() or "unknown GPU"


def ram_preset(gb):
    """The README's RAM size (sizing.PRESET_RAM) nearest the PC's RAM: 68.3 GB is the 64 GB preset."""
    if not gb:
        return None
    gib = gb * 1e9 / 2**30
    return min(sizing.PRESET_RAM, key=lambda r: abs(math.log(gib / r)))


def model_facts(quant):
    """(model name, bpw) from the catalog: the upload's name without -exl3 ('GLM-5.3-Flash'), or the quant alone for
    a model no longer in it."""
    try:
        m = catalog.load().model(quant)
    except KeyError:
        return quant, None
    name = m["source"]["huggingface"].split("/")[-1]
    return re.sub(r"-exl3$", "", name, flags=re.I), m.get("bpw")


def speed_row(rec):
    """A benchmark run's log line, or None when it doesn't count (failed, cancelled, interrupted, unfinished)."""
    if not bench.counts(rec) or rec.get("error") or not rec.get("decode_tps_median"):
        return None
    host, long = rec.get("host") or {}, rec.get("long") or {}
    name, bpw = model_facts(rec["quant"])
    prefill = {int(n): v["median"] for n, v in (rec.get("prefill") or {}).items() if (v or {}).get("median")}
    if long.get("prefill_tps"):
        pf_tps, pf_tokens = long["prefill_tps"], long.get("prompt_tokens")
    elif prefill:
        pf_tokens = max(prefill)
        pf_tps = prefill[pf_tokens]
    else:
        pf_tps = pf_tokens = None
    features = (([f"mtp={rec['mtp']}"] if rec.get("mtp") else []) + (["cpu"] if rec.get("cpu_lane") is not None
                                                                       else [])
                + (["prefetch"] if rec.get("prefetch") else []))
    return {"id": f"{rec.get('date')} {rec['label']} {host.get('gpu')}", "date": rec.get("date"),
            "gpu": gpu_name(host.get("gpu")), "card_gb": rec.get("card_total_gb"),
            "card": bench.card_class(rec["usable_gb"]), "capped": not bench.uncapped(rec),
            "ram_gb": host.get("ram_total_gb"), "ram_preset": ram_preset(host.get("ram_total_gb")),
            "platform": host.get("platform"), "model": name, "quant": rec["quant"], "bpw": bpw,
            "context": rec["context"], "staging": not rec.get("nostage"), "ram_cache_gib": rec.get("ram_gib"),
            "features": features, "workload": rec.get("workload"), "protocol": rec.get("protocol"),
            "decode_tps": rec["decode_tps_median"], "prefill_tps": pf_tps, "prefill_tokens": pf_tokens,
            "ttft_s": rec.get("short_ttft_median_s"), "stack": rec.get("stack")}


def quality_row(res):
    """A head-to-head result's log line (research/head-to-head/run.py's JSON), or None without the setup it ran on."""
    setup = res.get("setup")
    if not setup or not res.get("tasks"):
        return None
    name, bpw = model_facts(setup["quant"])
    tasks = [{"id": t["id"], "ok": bool(t.get("ok")), "tokens": t.get("completion_tokens"),
              "minutes": round(t["seconds"] / 60, 1) if t.get("seconds") else None,
              "cut_off": bool(t.get("timed_out"))} for t in res["tasks"]]
    return {"id": f"{res.get('date')} {res['label']} {setup.get('gpu')}", "date": res.get("date"),
            "suite": "head-to-head", "gpu": gpu_name(setup.get("gpu")), "card_gb": setup.get("card_gb"),
            "card": bench.card_class(setup["vram_budget_gb"]), "capped": False, "ram_gb": setup.get("ram_gb"),
            "ram_preset": ram_preset(setup.get("ram_gb")), "platform": setup.get("platform"), "model": name,
            "quant": setup["quant"], "bpw": bpw, "context": setup.get("context"),
            "ram_cache_gib": setup.get("ram_cache_gib"), "limit_min": (res.get("limit_s") or 0) // 60,
            "passed": sum(t["ok"] for t in tasks), "total": len(tasks), "tasks": tasks}


# ---------------------------------------------------------------------------------------------------------- log
def read(name, folder=LOG):
    p = Path(folder) / name
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(name, rows, folder=LOG):
    Path(folder).mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r.get("date") or "", r["id"]))
    (Path(folder) / name).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8",
                                     newline="\n")


def rows_from(path):
    """[(log file, line)] for a path: a benchmark run's JSON, a bench/<date-time>/ folder, or a head-to-head
    result. Skipped files come back as (None, why)."""
    path = Path(path)
    files = bench.record_files(path) if path.is_dir() else [path]
    out = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            out.append((None, f"{f.name}: not readable ({e})"))
            continue
        if "tasks" in data:
            row = quality_row(data)
            out.append((QUALITY, row) if row else (None, f"{f.name}: no setup recorded (run.py records it)"))
        elif data.get("label"):
            row = speed_row(data)
            out.append((SPEED, row) if row else (None, f"{f.name}: not a completed run"))
        else:
            out.append((None, f"{f.name}: neither a benchmark run nor a head-to-head result"))
    return out


def add(paths, folder=LOG):
    """Log the runs in paths (a run already logged is replaced). Returns the lines to print."""
    logs = {SPEED: {r["id"]: r for r in read(SPEED, folder)}, QUALITY: {r["id"]: r for r in read(QUALITY, folder)}}
    said, changed = [], set()
    for p in paths:
        for name, row in rows_from(p):
            if name is None:
                said.append("skipped " + row)
                continue
            said.append(f"{'replaced' if row['id'] in logs[name] else 'added'} {row['id']} ({name})")
            logs[name][row["id"]] = row
            changed.add(name)
    for name in changed:
        write(name, logs[name].values(), folder)
    return said


# --------------------------------------------------------------------------------------------------------- page
def _ctx(n):
    return f"{n // 1024}K" if n else "-"


def _num(v, spec="{:,.1f}"):
    return "-" if v is None else spec.format(v)


def machine(r):
    return (f"{r['gpu']}" + (f" ({r['card_gb'] * 1e9 / 2**30:.0f} GB)" if r.get("card_gb") else "")
            + (f", {r['ram_preset']} GB RAM" if r.get("ram_preset") else "")
            + (f", {sizing.PLATFORMS.get(r['platform'], r['platform'])}" if r.get("platform") else ""))


def preset(r):
    """The card size the model had (the machine's RAM is in its heading); capped: a smaller card emulated."""
    return f"{r['card']} card" + (" (capped)" if r.get("capped") else "")


def card_size(r):
    return int(r["card"].split()[0])


def quality_table(rows):
    head = "| Model | Quant | Preset | Passed | " + " | ".join(H2H_TASKS) + " | Run on |"
    out = [head, "|---" * (len(H2H_TASKS) + 5) + "|"]
    for r in sorted(rows, key=lambda r: (-r["passed"], r["model"], r["quant"], r["date"] or "")):
        by = {t["id"]: t for t in r["tasks"]}
        cells = []
        for tid in H2H_TASKS:
            t = by.get(tid)
            if not t:
                cells.append("-")
            elif t["cut_off"]:
                cells.append(f"cut off ({r['limit_min']} min)")
            else:
                cells.append(f"{'pass' if t['ok'] else 'fail'} ({_num(t['minutes'], '{:.1f}')} min)")
        out.append(f"| {r['model']} | {r['bpw'] or r['quant']} bpw | {preset(r)} | {r['passed']}/{r['total']} | "
                   + " | ".join(cells) + f" | {machine(r)}, {r['date']} |")
    return "\n".join(out)


def speed_tables(rows):
    """A section per machine (GPU, RAM, platform); its rows by preset (largest card first), workload and model."""
    out = []
    groups = {}
    for r in rows:
        groups.setdefault(machine(r), []).append(r)
    for name in sorted(groups):
        out += [f"### {name}", "",
                "| Preset | Model | Quant | Context | RAM cache | Engine options | Decode | Prefill | First token "
                "| Workload | Date |", "|---" * 11 + "|"]
        for r in sorted(groups[name], key=lambda r: (-card_size(r), r.get("capped"), WORKLOADS.get(r["workload"], 9),
                                                     r["model"], r["quant"], r["features"], r["date"] or "")):
            ctx = _ctx(r["context"]) + ("" if r.get("staging", True) else ", no staging")
            pf = (f"{r['prefill_tps']:,.0f} tok/s ({r['prefill_tokens'] // 1000}K)" if r.get("prefill_tps")
                  and r.get("prefill_tokens") else _num(r.get("prefill_tps"), "{:,.0f} tok/s"))
            out.append(f"| {preset(r)} | {r['model']} | {r['bpw'] or r['quant']} bpw | {ctx} | "
                       f"{_num(r.get('ram_cache_gib'), '{} GiB')} | {', '.join(r['features']) or '-'} | "
                       f"{r['decode_tps']:.1f} tok/s | {pf} | {_num(r.get('ttft_s'), '{:.2f} s')} | "
                       f"{r['workload']} | {r['date']} |")
        out.append("")
    return "\n".join(out).rstrip()


def render(folder=LOG):
    """The generated part of docs/benchmarks.md (between <!-- results:begin/end -->)."""
    quality, speed = read(QUALITY, folder), read(SPEED, folder)
    parts = ["## Quality: head-to-head", "", quality_table(quality) if quality else "No results yet.", "",
             "## Speed, by GPU", "", speed_tables(speed) if speed else "No results yet."]
    return "\n".join(parts)


def write_page(path=PAGE, folder=LOG):
    text = Path(path).read_text(encoding="utf-8")
    new, n = re.subn(r"(<!-- results:begin -->\n)(?:.*?\n)?(<!-- results:end -->)",
                     lambda m: m.group(1) + render(folder) + "\n" + m.group(2), text, flags=re.S)
    if not n:
        raise SystemExit(f"{path}: no <!-- results:begin/end --> markers")
    Path(path).write_text(new, encoding="utf-8", newline="\n")


def main(a):
    if a.results_cmd == "add":
        for line in add(a.paths):
            print(line)
        print("then: python -m repertory results page")
    else:
        write_page()
        print(f"wrote {PAGE.relative_to(ROOT)}")
    return 0
