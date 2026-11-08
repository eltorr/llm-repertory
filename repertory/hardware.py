"""Hardware and platform detection, and the recommendations the dashboard shows.

detect() is cheap enough to run on demand (a few hundred ms; WSL queries are the slow part) and never raises:
missing pieces are reported as such.
"""
import configparser
import ctypes
import math
import os
import re
import subprocess
from pathlib import Path

import psutil


from . import catalog
from .config import GB, GIB, IS_WINDOWS, REPO, WSL_EXE

from engines.exl3_offload import sizing  # noqa: E402  (the engine adapter's, shared with its run.sh)

NO_WINDOW = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
WSL_ENV = dict(os.environ, WSL_UTF8="1")


def _run(args, timeout=20, env=None):
    try:
        r = subprocess.run(args, capture_output=True, timeout=timeout, env=env, creationflags=NO_WINDOW,
                           encoding="utf-8", errors="replace")
        return r.returncode, (r.stdout or "").replace("\x00", ""), (r.stderr or "").replace("\x00", "")
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, "", str(e)


def lib_value(name):
    """A version set in setup/lib.sh, such as CUDA_VER."""
    try:
        m = re.search(rf"^{name}=([0-9.]+)", (REPO / "setup" / "lib.sh").read_text(), re.M)
    except OSError:
        return None
    return m.group(1) if m else None


def engine_cuda():
    """The CUDA version the engine is built with: CUDA_VER in setup/lib.sh."""
    return lib_value("CUDA_VER")


def driver_too_old(hw):
    """(driver's CUDA, engine's CUDA, minimum driver) when the driver can't run the engine's CUDA, else None."""
    need, drv = engine_cuda(), ((hw or {}).get("driver") or {}).get("cuda")
    if need and drv and version(drv) < version(need):
        return drv, need, lib_value("DRIVER_MIN")
    return None


def version(v):
    return tuple(int(x) for x in re.findall(r"\d+", str(v))[:3])


def parse_kv(text):
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


# ---------------------------------------------------------------------------------------------------------------
# Windows / WSL
# ---------------------------------------------------------------------------------------------------------------
def wsl_info(distro):
    info = {"installed": False, "version": None, "kernel": None, "default_version": None, "distros": [],
            "distro": distro, "distro_present": False, "distro_wsl_version": None, "distro_state": None,
            "distro_path": None}
    rc, out, _ = _run([WSL_EXE, "--version"], env=WSL_ENV)
    if rc == 0 and "WSL" in out:
        info["installed"] = True
        m = re.search(r"WSL[^:\n]*:\s*([\d.]+)", out)
        info["version"] = m.group(1) if m else None
        m = re.search(r"Kernel[^:\n]*:\s*([\w.+-]+)", out)
        info["kernel"] = m.group(1) if m else None
    rc, out, _ = _run([WSL_EXE, "--status"], env=WSL_ENV)
    m = re.search(r"Default Version:\s*(\d)", out)
    info["default_version"] = int(m.group(1)) if m else None
    rc, out, _ = _run([WSL_EXE, "-l", "-v"], env=WSL_ENV)
    for line in out.splitlines()[1:]:
        parts = line.replace("*", " ").split()
        if len(parts) >= 3 and parts[-1].isdigit():
            d = {"name": " ".join(parts[:-2]), "state": parts[-2], "version": int(parts[-1]),
                 "default": line.strip().startswith("*")}
            info["distros"].append(d)
            if d["name"] == distro:
                info.update(distro_present=True, distro_wsl_version=d["version"], distro_state=d["state"])
    if info["distros"]:
        info["installed"] = True
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Lxss") as k:
            for i in range(winreg.QueryInfoKey(k)[0]):
                with winreg.OpenKey(k, winreg.EnumKey(k, i)) as d:
                    try:
                        if winreg.QueryValueEx(d, "DistributionName")[0] == distro:
                            info["distro_path"] = winreg.QueryValueEx(d, "BasePath")[0].replace("\\\\?\\", "")
                    except OSError:
                        continue
    except OSError:
        pass
    return info


def wslconfig():
    p = Path.home() / ".wslconfig"
    out = {"path": str(p), "exists": p.exists(), "memory_gb": None, "processors": None, "swap_gb": None}
    if not p.exists():
        return out
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    try:
        cp.read_string(p.read_text(errors="replace"))
    except configparser.Error as e:
        out["error"] = str(e)
        return out
    sec = cp["wsl2"] if cp.has_section("wsl2") else {}

    def size_gb(v):
        m = re.match(r"\s*(\d+(?:\.\d+)?)\s*(GB|MB|TB)?", str(v), re.I)
        if not m:
            return None
        n, unit = float(m.group(1)), (m.group(2) or "B").upper()
        return n * {"TB": 1024, "GB": 1, "MB": 1 / 1024}.get(unit, 1 / GIB)
    out["memory_gb"] = size_gb(sec.get("memory")) if sec.get("memory") else None
    out["swap_gb"] = size_gb(sec.get("swap")) if sec.get("swap") else None
    out["processors"] = int(sec["processors"]) if str(sec.get("processors", "")).isdigit() else None
    return out


def windows_cpu():
    name = None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            name = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except OSError:
        pass
    k32 = ctypes.windll.kernel32
    return {"name": name, "avx2": bool(k32.IsProcessorFeaturePresent(40)),
            "virtualization_firmware": bool(k32.IsProcessorFeaturePresent(21))}


def fixed_drives():
    out = []
    for p in psutil.disk_partitions(all=False):
        if "fixed" in p.opts or not IS_WINDOWS:
            try:
                u = psutil.disk_usage(p.mountpoint)
                out.append({"mount": p.mountpoint, "fs": p.fstype, "free_gb": round(u.free / GB),
                            "total_gb": round(u.total / GB)})
            except OSError:
                continue
    return out


# ---------------------------------------------------------------------------------------------------------------
# common
# ---------------------------------------------------------------------------------------------------------------
def linux_cpu():
    name, flags = None, ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name") and not name:
                name = line.split(":", 1)[1].strip()
            elif line.startswith("flags") and not flags:
                flags = line
    except OSError:
        pass
    return {"name": name, "avx2": " avx2" in flags}


def repo_state():
    rc, head, _ = _run(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    rc2, dirty, _ = _run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"])
    rc3, subject, _ = _run(["git", "-C", str(REPO), "log", "-1", "--format=%h %s"])
    return {"head": head.strip() if rc == 0 else None, "dirty": bool(dirty.strip()) if rc2 == 0 else None,
            "subject": subject.strip() if rc3 == 0 else None}


def engine_check(host):
    """setup/check.sh in the engine's environment (WSL or here), for the catalog's models."""
    from .setup import model_specs
    script = host.to_engine_path(REPO / "setup" / "check.sh")
    try:
        r = host.run("bash", script, host.engine_dir, *model_specs(), timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": str(e)}
    vals = parse_kv(r.stdout or "")
    if "driver" not in vals:
        vals["error"] = ((r.stderr or "") + (r.stdout or "")).strip()[-300:] or f"exit {r.returncode}"
    return vals


def installed(engine):
    """Which of the catalog's models setup/check.sh found downloaded and packed, by quant: {"3.05": True, ...}."""
    return {m["quant"]: engine.get(catalog.setup_id(m)) == "1" for m in catalog.load().models_for(sizing.ENGINE)}


def detect(host, cfg, with_engine=True):
    hw = {"host": host.describe(), "repo": repo_state()}
    vm = psutil.virtual_memory()
    hw["ram"] = {"total_gb": round(vm.total / GB, 1), "avail_gb": round(vm.available / GB, 1)}
    hw["cpu"] = windows_cpu() if IS_WINDOWS else linux_cpu()
    hw["cpu"].update(cores=psutil.cpu_count(logical=False), threads=psutil.cpu_count())
    gpus = []
    if host.nvml:
        for d in host.nvml.devices:
            try:
                gpus.append(d.info())
            except OSError:
                continue
        hw["driver"] = {"version": host.nvml.driver_version(), "cuda": host.nvml.cuda_driver_version()}
    else:
        hw["driver"] = {"error": host.nvml_error}
    hw["gpus"] = gpus
    hw["drives"] = fixed_drives()
    if IS_WINDOWS:
        hw["wsl"] = wsl_info(cfg["distro"])
        hw["wslconfig"] = wslconfig()
    engine_ready = not IS_WINDOWS or hw["wsl"]["distro_present"]
    hw["engine"] = engine_check(host) if with_engine and engine_ready else {}
    hw["recommendations"] = recommend(hw, cfg)
    return hw


# ---------------------------------------------------------------------------------------------------------------
# recommendations
# ---------------------------------------------------------------------------------------------------------------
def wsl_memory_gb(total_gb):
    """Give WSL most of the RAM (the expert cache lives there) but keep Windows responsive."""
    return int(sizing.wsl_memory_bytes(total_gb * GB) / GB)


def display_reliable():
    """NVML's display_active reads 'Disabled' under Windows (WDDM) and inside WSL even on the card that drives the
    desktop; only native Linux reports it truthfully."""
    if IS_WINDOWS:
        return False
    try:
        return "microsoft" not in Path("/proc/version").read_text().lower()
    except OSError:
        return True


def has_display(gpu):
    """Whether the card drives a display; when that can't be known, assume it does (the safe side for VRAM)."""
    return gpu.get("display_active") is not False or not display_reliable()


def gpu_budget(gpu, cfg):
    """VRAM the model could use with nothing else running: total minus the desktop's share and headroom."""
    total = gpu["vram_total"]
    if cfg["full_vram_headroom_gb"] > 0:
        desktop = sizing.DESKTOP_VRAM if has_display(gpu) else 0.1 * GB
        return max(0.0, total - desktop - cfg["full_vram_headroom_gb"] * GB)
    return max(0.0, sizing.card_budget(total, has_display(gpu)))


def machine_baseline(gpu, ram_total, kind):
    """sizing.baseline() for this card, RAM and platform (README "Presets")."""
    return sizing.baseline(gpu["vram_total"], ram_total, has_display(gpu), kind)


def baseline_item(b, cfg, have=None):
    """The baseline as a recommendation: (level, title, detail, settings to apply)."""
    if not b["ok"]:
        return "error", "This machine can't run the model as it is", b["reason"], None
    mem = ("every expert fits in VRAM + RAM" if b["in_memory"] >= 0.97 else
           f"{b['in_memory']:.0%} of the experts fit in VRAM + RAM; the rest is read from the NVMe drive when picked")
    detail = f"{b['slots']:,} experts cached in VRAM; {mem}."
    if not b["staging"]:
        detail += " Prompts are processed staging-free (small card)."
    for m in b["measured"][:1]:
        detail += (f" Measured: {m['decode']} tok/s generation, {m['prefill']:,} tok/s prompt processing "
                   f"({sizing.measured_on(m)}).")
    if b.get("experimental"):
        detail += " Below 12 GB this is experimental."
    settings = {k: v for k, v in b["settings"].items() if cfg.get(k) != v}
    if have is not None and not have.get(b["quant"]) and have.get(cfg["quant"]) and "quant" in settings:
        detail += (f" The {b['quant']} bpw model isn't downloaded yet: run its setup step, then apply this to switch.")
        settings = {k: v for k, v in settings.items() if k not in ("quant", "context")}
    title = (f"{b['quant']} bpw, {b['context'] // 1024}K context, {b['ram_cache_gib']} GiB RAM cache")
    if not settings:
        return "ok", "Baseline for this machine (your settings): " + title, detail, None
    return "info", "Baseline for this machine: " + title, detail, settings


def recommend(hw, cfg):
    """[{level: ok|info|warn|error, title, detail, settings?}] plus the best settings for this machine."""
    out = []

    def add(level, title, detail="", settings=None):
        out.append({"level": level, "title": title, "detail": detail, "settings": settings or {}})
    if not hw["gpus"]:
        add("error", "No NVIDIA GPU found", hw.get("driver", {}).get("error") or "Install the NVIDIA driver.")
        return out
    g = hw["gpus"][0]
    if hw["host"].get("in_wsl"):
        add("warn", "The manager is running inside WSL", "It can't see Windows apps' GPU use from here, so it can't "
                                                         "make room for them. Run Repertory.cmd on Windows instead.")
    old = driver_too_old(hw)
    if old:
        drv, need, least = old
        after = " on Windows, then run wsl --shutdown." if hw["host"].get("kind") == "windows-wsl" else ", then reboot."
        add("error", f"Update the NVIDIA driver: it supports CUDA {drv}",
            f"The engine is built with CUDA {need}. Install driver {least or 'a version that supports it'} or "
            f"newer{after}")
    cc = float(g["compute_capability"] or 0)
    vram = g["vram_total"]
    if cc < 7.5:
        add("error", f"{g['name']} is too old", "The engine needs an NVIDIA GPU of the Turing generation (RTX 20) "
                                                "or newer.")
    elif cc < 8.0:
        add("warn", f"{g['name']}: Turing is untested", "The EXL3 kernels have Turing fallbacks, but this engine has "
                                                        "only been run on Ampere and newer.")
    budget = gpu_budget(g, cfg)
    quant = cfg["quant"]
    kind = hw["host"].get("kind", "linux")
    ram_total = hw["ram"]["total_gb"] * GB
    e = hw.get("engine") or {}
    have = installed(e) if e.get("driver") else None
    base = machine_baseline(g, ram_total, kind)
    level, title, detail, settings = baseline_item(base, cfg, have)
    add(level, title, detail, settings)
    out[-1]["apply_label"] = "Apply baseline"
    ctx_any = sizing.max_context(quant, budget, staging=False)
    ctx_fast = sizing.max_context(quant, budget)
    # 3.05 is the smaller of Qwen's two quants: named here as in the preset rule (sizing.baseline), until another
    # family needs its own
    if sizing.max_context("3.05", budget, staging=False) == 0:
        pass                                    # the baseline item says why
    elif ctx_any == 0:
        add("warn", f"{quant} bpw does not fit this GPU", "Use 3.05 bpw.", {"quant": "3.05"})
    else:
        level = "ok" if vram >= 15e9 else "warn"
        tested = "" if vram >= 15e9 else " Below 16 GB is experimental: smaller prefill chunks and a smaller cache."
        staged = (f" Prompt processing is staging-free (slower) above {ctx_fast // 1024}K."
                  if ctx_fast < ctx_any else "")
        add(level, f"{g['name']} ({vram / GIB:.0f} GB): up to {ctx_any // 1024}K context at {quant} bpw",
            f"About {budget / GB:.1f} GB for the model with nothing else on the GPU.{staged}{tested}",
            {"context": ctx_any} if cfg["context"] > ctx_any else None)
    if len(hw["gpus"]) > 1:
        add("info", f"{len(hw['gpus'])} GPUs found: the engine uses GPU 0 ({g['name']})",
            "Multi-GPU is a research item; see docs/research/multi-gpu.md.")
    gen, width = g.get("pcie_gen_max"), g.get("pcie_width_max")
    if gen and width and (gen < 4 or width < 16):
        add("warn", f"GPU link is PCIe {gen}.0 x{width}",
            "Experts that miss the VRAM cache cross this link every token; Gen4 x16 or better is recommended. "
            "Check the slot (a second card or M.2 drive can halve the lanes).")
    elif gen and width:
        add("ok", f"GPU link PCIe {gen}.0 x{width}")
    if g.get("display_active"):
        add("info", "This GPU also drives the display", "About 1-2 GB of VRAM stays with the desktop.")

    # RAM: with the VRAM cache, the RAM cache should hold RAM_COVERAGE of the experts; the rest are rarely picked
    host_ram = hw["ram"]["total_gb"]
    plan = sizing.plan(quant, budget, min(cfg["context"], ctx_any or cfg["context"]))
    need_gib = math.ceil(plan["ram_target"] / GIB)
    share = f"{sizing.RAM_COVERAGE:.0%} of the experts"
    over = sizing.RAM_OVERHEAD.get(kind, 8 * GB)
    cache = sizing.ram_cache_gib(plan["ram_target"], ram_total - sizing.OS_IDLE.get(kind, 4 * GB), cfg, over)
    if IS_WINDOWS:
        wsl_gb = hw.get("wslconfig", {}).get("memory_gb") or host_ram / 2
        cache = min(cache, int((wsl_gb * GB - over) // GIB))
    if cache >= need_gib:
        add("ok", f"RAM: a {need_gib} GiB expert cache, which with VRAM holds {share}",
            "The rest are rarely picked and are read from the NVMe drive.")
    elif cache > cfg["lean_ram_gb"]:
        add("info", f"RAM: about {cache} of the {need_gib} GiB expert cache",
            f"With VRAM, {need_gib} GiB would hold {share}; more is read from the NVMe drive instead. More RAM, or "
            "3.05 bpw, keeps more of the model in memory.")
    else:
        add("warn", "Little RAM for the expert cache", f"About {cache} GiB of the {need_gib} GiB the model could use "
                                                       f"can be spared.")
    if IS_WINDOWS:
        w, wc = hw["wsl"], hw["wslconfig"]
        if not w["installed"]:
            add("error", "WSL is not installed", "Run the 'WSL' setup step (needs administrator approval and "
                                                  "usually a restart).")
        elif w.get("distro_present") and w.get("distro_wsl_version") == 1:
            add("error", f"{w['distro']} runs under WSL 1", f"Convert it: wsl --set-version {w['distro']} 2")
        want = wsl_memory_gb(host_ram)
        have = wc.get("memory_gb")
        if have is None:
            add("warn", f"WSL may use only half the RAM ({host_ram / 2:.0f} GB)",
                f"Run the 'WSL memory' step to allow {want} GB.")
        elif have < want * 0.9:
            add("info", f"WSL is limited to {have:.0f} GB", f"{want} GB would leave more for the expert cache.")
        if not hw["cpu"].get("virtualization_firmware") and not w["installed"]:
            add("warn", "Virtualization looks disabled in the firmware", "WSL 2 needs it (Intel VT-x / AMD SVM).")
    if e.get("fs") and e["fs"] not in ("ext2/ext3", "xfs", "btrfs"):
        add("warn", f"Engine is on a '{e['fs']}' filesystem", "The expert store needs O_DIRECT reads (ext4 or xfs).")
    if e.get("disk_free_gb"):
        q = sizing.QUANTS[quant]
        need = 0 if installed(e).get(quant) else q["download_gb"] + q["store_gb"]
        if need and int(e["disk_free_gb"]) < need:
            add("error", f"Not enough disk space for {quant} bpw", f"Needs ~{need} GB, {e['disk_free_gb']} GB free.")
    return out
