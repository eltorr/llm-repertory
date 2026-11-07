"""Repertory manager.

  python -m repertory [serve] [--open] [--observe] [--port N]   run the service (dashboard + OpenAI API)
  python -m repertory open                                     open the dashboard, starting the service if needed
  python -m repertory status                                   what the running service is doing
  python -m repertory detect                                   hardware, platform and recommendations
  python -m repertory setup [STEP]                             list setup steps, or run one in this terminal
  python -m repertory plan [--budget-gb N]                     VRAM/RAM plan for this GPU (sizing.py)
  python -m repertory bench [RUNS] [--workload quick]          benchmark this machine (takes the GPU while it runs)
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config
from .config import GB, IS_WINDOWS, REPO
from .httpio import http_get_json, http_ok


def service_url(cfg):
    """Base of the dashboard's control API (append api/...)."""
    return f"http://127.0.0.1:{cfg['listen_port']}/_switcher/"


def dashboard_url(cfg):
    """The dashboard, at the root of the port that serves the API on /v1, opened with its token: the page keeps it in
    a cookie and drops it from the address bar."""
    return f"http://127.0.0.1:{cfg['listen_port']}/?token={cfg['dashboard_token']}"


def auth(cfg):
    """The dashboard's control API needs its token, on this PC too."""
    return {"Authorization": f"Bearer {cfg['dashboard_token']}"}


def cmd_serve(cfg, a):
    from .service import Service
    if a.port:
        cfg["listen_port"] = a.port
    if http_ok(service_url(cfg) + "api/status", 1.5, auth(cfg)):
        if a.open:
            from .host import make_host
            make_host(cfg).open_ui(dashboard_url(cfg))
        else:
            print(f"Repertory is already running on port {cfg['listen_port']}")
        return 0
    if a.observe:
        # never compete with the real service for its ports
        cfg["network_port"] = 0
    asyncio.run(Service(cfg, observe=a.observe).serve(a.open))
    return 0


def cmd_open(cfg, a):
    url = service_url(cfg)
    if not http_ok(url + "api/status", 1.5, auth(cfg)):
        exe = sys.executable
        if IS_WINDOWS:
            pyw = os.path.join(os.path.dirname(exe), "pythonw.exe")
            exe = pyw if os.path.exists(pyw) else exe
            subprocess.Popen([exe, "-m", "repertory", "serve"], cwd=REPO,
                             creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            subprocess.Popen([exe, "-m", "repertory", "serve"], cwd=REPO, start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(40):
            if http_ok(url + "api/status", 1.0, auth(cfg)):
                break
            time.sleep(0.5)
    from .host import make_host
    make_host(cfg).open_ui(dashboard_url(cfg))
    return 0


def cmd_status(cfg, a):
    try:
        s = http_get_json(service_url(cfg) + "api/status", 3, auth(cfg))
    except Exception as e:  # noqa: BLE001
        print(f"Repertory is not running ({e})")
        return 1
    if a.json:
        print(json.dumps(s, indent=2))
        return 0
    g = s["gpu"]
    print(f"model    {s['state']}{' (' + s['tier'] + ')' if s['tier'] else ''}  {s['quant']} bpw  "
          f"{s['context'] // 1024}K context  endpoint {s['endpoint']}")
    if s.get("budget_gb"):
        print(f"budget   {s['budget_gb']} GB VRAM, {s['ram_cache_gib']} GiB RAM cache")
    print(f"gpu      {g['name']}: {(g['used'] or 0) / GB:.1f} of {(g['total'] or 0) / GB:.1f} GB used "
          f"(model {g['own'] / GB:.1f}, other apps {g['ext'] / GB:.1f})")
    print(f"pressure {s['pressure']['level']}  {'; '.join(s['pressure']['reasons'])}")
    mt = s.get("metrics") or {}
    if mt.get("count"):
        print(f"speed    decode {mt['decode_tps']} tok/s, prefill {mt['prefill_tps']} tok/s, "
              f"first token {mt['ttft_s']} s (median of recent requests)")
    print(f"requests served {s['served']}, failed {s['failed']}, in flight {s['in_flight']}")
    if s.get("error"):
        print(f"error    {s['error'][:300]}")
    return 0


def print_recommendations(recs):
    icon = {"ok": "ok  ", "info": "info", "warn": "WARN", "error": "FAIL"}
    for r in recs:
        print(f"  [{icon.get(r['level'], r['level'])}] {r['title']}")
        if r.get("detail"):
            print(f"         {r['detail']}")
        if r.get("settings"):
            print(f"         suggested: {r['settings']}")


def cmd_detect(cfg, a):
    from . import hardware
    from .host import make_host
    hw = hardware.detect(make_host(cfg), cfg)
    if a.json:
        print(json.dumps(hw, indent=2, default=str))
        return 0
    h = hw["host"]
    print(f"platform {h['label']}: {h.get('os')}{' (inside WSL)' if h.get('in_wsl') else ''}")
    print(f"cpu      {hw['cpu'].get('name')} ({hw['cpu'].get('cores')} cores), RAM {hw['ram']['total_gb']} GB")
    for g in hw["gpus"]:
        sm = (g["compute_capability"] or "?").replace(".", "")
        print(f"gpu {g['index']}    {g['name']}  {g['vram_total'] / 2**30:.0f} GB  sm_{sm}"
              f"  PCIe {g.get('pcie_gen_max')}.0 x{g.get('pcie_width_max')}")
    if "wsl" in hw:
        w = hw["wsl"]
        print(f"wsl      {'installed ' + (w['version'] or '') if w['installed'] else 'not installed'}; "
              f"{w['distro']}: "
              f"{'present (WSL ' + str(w['distro_wsl_version']) + ')' if w['distro_present'] else 'missing'}"
              f"; .wslconfig memory {hw['wslconfig'].get('memory_gb')} GB")
    e = hw.get("engine") or {}
    if e:
        built = ", built for " + e["build_arch"] if e.get("build_arch") else ""
        ready = "ready" if e.get("engine") == "1" else "not set up"
        print(f"engine   {ready} at {cfg['repo_dir'] or 'this checkout'}; "
              f"CUDA {e.get('cuda_version')}, torch {e.get('torch')}{built}; "
              f"models: 3.05 {'yes' if e.get('m305') == '1' else 'no'}, 4.05 {'yes' if e.get('m405') == '1' else 'no'}")
    print("recommendations:")
    print_recommendations(hw["recommendations"])
    return 0


def cmd_setup(cfg, a):
    """A step runs in the running service when there is one, as from the dashboard: a step that changes the engine
    stops the model first, and requests wait until it's done. Without the service, it runs in this terminal."""
    base = service_url(cfg) + "api/"
    if a.step and http_ok(base + "status", 1.5, auth(cfg)):
        return setup_via_service(cfg, base, a.step)
    from .service import Service
    svc = Service(cfg)
    asyncio.run(svc.setup.check())
    if not a.step:
        for s in svc.setup.steps:
            print(f"  {s['id']:10s} {svc.setup.status.get(s['id'], '?'):9s} {s['title']}")
        print("run one with: python -m repertory setup STEP")
        return 0

    async def go():
        svc.tlock = asyncio.Lock()
        svc.setup.emit = lambda line: print(line, flush=True)
        ok, msg = await svc.setup.run(a.step)
        if not ok:
            print(msg)
            return 2
        # the step's status comes from the check that follows it
        while svc.setup.running or svc.setup.checking:
            await asyncio.sleep(0.5)
        return 0 if svc.setup.status.get(a.step) in ("ok", "outdated") else 1
    return asyncio.run(go())


def setup_via_service(cfg, base, step):
    seq = http_get_json(base + "setup", 10, auth(cfg))["seq"]
    r = post_json(base + "setup/run", {"step": step}, auth(cfg))
    if not r.get("ok"):
        print(r.get("message"))
        return 2
    while True:
        r = http_get_json(base + f"setup?since={seq}", 10, auth(cfg))
        for _, line in r["lines"]:
            print(line, flush=True)
        seq = r["seq"]
        if not r["running"] and not r["checking"]:      # done, and checked again afterwards
            break
        time.sleep(1)
    status = next((s["status"] for s in r["steps"] if s["id"] == step), None)
    return 0 if status in ("ok", "outdated") else 1


def first_run_baseline(cfg):
    """A new install starts from the preset for this machine (sizing.baseline: card + RAM + platform)."""
    import psutil
    from .hardware import machine_baseline
    from .host import make_host
    try:
        h = make_host(cfg)
        if not h.gpu:
            return None
        b = machine_baseline(h.gpu.info(), psutil.virtual_memory().total, h.kind)
    except Exception as e:  # noqa: BLE001 - no GPU access yet: keep the defaults
        config.log.info("first run: no baseline (%s)", e)
        return None
    if not b["ok"]:
        config.log.info("first run: %s", b["reason"])
        return b
    settings = dict(b["settings"])
    if h.kind == "linux":
        # an engine checkout that already holds the other model: keep it rather than ask for a second download
        from engines.exl3_offload import sizing

        def present(q):
            root = Path(cfg["repo_dir"] or h.engine_dir)
            return (root / sizing.QUANTS[q]["model_dir"]).is_dir() and (root / sizing.QUANTS[q]["store"]).is_file()
        other = "3.05" if settings["quant"] == "4.05" else "4.05"
        if not present(settings["quant"]) and present(other):
            settings.pop("quant")
            settings.pop("context")
    cfg.update(settings)
    config.log.info("first run: baseline for this machine: %s", settings)
    return b


def cmd_plan(cfg, a):
    import psutil

    from engines.exl3_offload import sizing
    from .host import make_host
    from .hardware import gpu_budget, has_display, machine_baseline
    h = make_host(cfg)
    if h.gpu and not a.budget_gb:
        info = h.gpu.info()
        ram = psutil.virtual_memory().total
        print(f"this machine: {info['name']} ({info['vram_total'] / 2**30:.0f} GB, "
              f"{'display' if has_display(info) else 'no display'}), "
              f"{ram / 2**30:.0f} GB RAM, {sizing.PLATFORMS.get(h.kind, h.kind)}")
        baseline = sizing.describe(machine_baseline(info, ram, h.kind))
        print("baseline:     " + baseline.replace("\n  ", "\n              "))
        print(f"your config:  {cfg['quant']} bpw, {cfg['context'] // 1024}K context")
        print()
    budget = a.budget_gb * GB if a.budget_gb else (gpu_budget(h.gpu.info(), cfg) if h.gpu else 0)
    for q in sizing.QUANTS:
        p = sizing.plan(q, budget, a.context or cfg["context"])
        state = "ok" if p["ok"] else "does not fit"
        print(f"{q} bpw, {budget / GB:.1f} GB, {p['context'] // 1024}K: {state}; chunk {p['chunk']}, "
              f"{p['slots']} cached experts, a {p['ram_target'] / 2**30:.0f} GiB RAM cache brings them to "
              f"{sizing.RAM_COVERAGE:.0%}; "
              f"max context {sizing.max_context(q, budget, staging=False) // 1024}K"
              + ("" if p["staging"] else " (staging-free)"))
    return 0


def post_json(url, body, headers, timeout=30):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read())
        except ValueError:
            return {"ok": False, "message": f"HTTP {e.code}"}


def cmd_bench(cfg, a):
    """Benchmark through the running service (as the dashboard does), or in this terminal when it isn't running."""
    from . import bench
    if a.table or a.sizing:
        folder = Path(a.table or a.sizing)
        records = bench.load_records(folder)
        if not records:
            print(f"no results in {folder}")
            return 1
        if a.sizing:
            print(bench.measured_rows(records, folder.as_posix().rstrip("/").removesuffix("/results")))
            return 0
        print(bench.table(records))
        p = bench.prefill_table(records)
        if p:
            print("\nDocument reading by size:\n\n" + p)
        return 0
    if a.list:
        print("Presets:")
        for name, what in bench.PRESETS.items():
            print(f"  {name:<8} {what}")
        past = bench.history(50)
        print("\nEarlier results:" if past else f"\nNo earlier results in {bench.BENCH_DIR}")
        for h in past:
            print(f"  {h['name']}  {h['runs']:>2} run(s)  {bench.BENCH_DIR / h['name']}")
        return 0
    if a.resume and a.runs:
        print("--resume continues the last benchmark's own runs; leave the runs out")
        return 2
    runs = a.runs or ["current"]
    base = service_url(cfg) + "api/"
    try:
        http_get_json(base + "status", 3, auth(cfg))
    except Exception:  # noqa: BLE001 - not running: benchmark here
        return bench_here(cfg, a, runs)
    r = post_json(base + "bench", {"runs": runs, "workload": a.workload, "resume": a.resume}, auth(cfg))
    print(r.get("message"))
    if not r.get("ok"):
        return 2
    seq, cancelled = 0, False
    while True:
        try:
            r = http_get_json(base + f"bench?since={seq}", 10, auth(cfg))
            for _, line in r["lines"]:
                print(line, flush=True)
            seq = r["seq"]
            if not r["running"]:
                break
            time.sleep(2)
        except KeyboardInterrupt:
            if cancelled:
                raise
            cancelled = True
            print("cancelling the benchmark (Ctrl+C again stops watching)")
            post_json(base + "bench/cancel", {}, auth(cfg))
    return bench_results(r["records"], a)


def bench_here(cfg, a, runs):
    """No service: run the benchmark in this process. A model server left running is stopped, and stays stopped."""
    from .service import Service
    print("The Repertory service isn't running, so the benchmark runs in this terminal.")
    svc = Service(cfg)
    svc.mon.start()
    svc.host.start_follower()
    svc.bench.echo = lambda line: print(line, flush=True)

    async def go():
        svc.tlock = asyncio.Lock()
        for _ in range(40):                       # the first GPU reading
            if svc.mon.get():
                break
            await asyncio.sleep(0.25)
        ok, msg = svc.bench.start(runs, a.workload, resume=a.resume)
        if not ok:
            print(msg)
            return None
        await svc.bench.task
        return svc.bench.records
    try:
        records = asyncio.run(go())
    except KeyboardInterrupt:
        svc.host.backend_stop()
        print("stopped")
        return 1
    return 2 if records is None else bench_results(records, a)


def bench_results(records, a):
    from . import bench
    if a.out and records:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        for r in records:
            (out / f"{r['label']}.json").write_text(json.dumps(r, indent=1))
        print(f"wrote {len(records)} result(s) to {out}")
    if records:
        print("\n" + bench.table(records))
        p = bench.prefill_table(records)
        if p:
            print("\nDocument reading by size:\n\n" + p)
    return 0 if records and not any(r.get("error") for r in records) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="repertory", description="Repertory manager")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("serve", help="run the service")
    for p in (ap, s):
        p.add_argument("--open", action="store_true", help="open the dashboard")
        p.add_argument("--observe", action="store_true", help="watch only: never start, stop or resize the model")
        p.add_argument("--port", type=int, help="dashboard/API port (default from config, 30000)")
    sub.add_parser("open", help="open the dashboard (starts the service if needed)")
    st = sub.add_parser("status", help="show what the running service is doing")
    st.add_argument("--json", action="store_true")
    d = sub.add_parser("detect", help="hardware, platform and recommendations")
    d.add_argument("--json", action="store_true")
    su = sub.add_parser("setup", help="list or run setup steps")
    su.add_argument("step", nargs="?")
    pl = sub.add_parser("plan", help="VRAM/RAM plan for this GPU")
    pl.add_argument("--budget-gb", type=float)
    pl.add_argument("--context", type=int)
    b = sub.add_parser("bench", help="benchmark this machine (takes the GPU; the model comes back afterwards)")
    b.add_argument("runs", nargs="*", help="presets: current (default), lean, quants, matrix; or "
                                           "QUANT:VRAM_GB|card:CONTEXT[:nostage][:ram=GIB]")
    b.add_argument("--workload", choices=("quick", "standard", "resize", "resize-stress"), default="standard",
                   help="quick: ~2 min a run; standard: ~3-6 min a run (default): answers, documents and "
                        "a retrieval test. resize: full -> lean -> full without a restart (whole-card runs); "
                        "resize-stress: the RAM cache's background work where it could go wrong")
    b.add_argument("--resume", action="store_true",
                   help="continue the latest unfinished benchmark: its runs left, into its folder")
    b.add_argument("--out", help="also write each run's JSON into this folder, e.g. research/NAME/results")
    b.add_argument("--table", metavar="FOLDER", help="print the report tables of a results folder")
    b.add_argument("--sizing", metavar="FOLDER", help="print sizing.MEASURED rows for a results folder")
    b.add_argument("--list", action="store_true", help="list the presets and earlier results")
    a = ap.parse_args(argv)
    cfg = config.load_config()
    if not config.CONFIG_PATH.exists():
        first_run_baseline(cfg)
        config.save_config(cfg)
    config.ensure_token(cfg)
    handler = {None: cmd_serve, "serve": cmd_serve, "open": cmd_open, "status": cmd_status, "detect": cmd_detect,
               "setup": cmd_setup, "plan": cmd_plan, "bench": cmd_bench}[a.cmd]
    return handler(cfg, a)


if __name__ == "__main__":
    sys.exit(main())
