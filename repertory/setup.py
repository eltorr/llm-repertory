"""Deployment steps for both platforms, their status (from hardware detection) and a runner that streams output.

Windows + WSL2: WSL itself, its memory limit, the Ubuntu distro, then the shared Linux steps run inside the distro.
Native Linux:   the NVIDIA driver (checked; installing one needs a reboot), then the same shared steps.
The Linux steps are the scripts in setup/ (10_base, 20_env, 30_build, 50_model); every step can be re-run safely.
The model steps come from the catalog: one per model the engine runs, with its label, sizes, paths and pin.
"""
import asyncio
import collections
import os
import re
import time
from pathlib import Path

import psutil

from . import catalog, hardware
from .config import GB, IS_WINDOWS, POWERSHELL_EXE, REPO, SYSTEM32, WSL_EXE, log

from engines.exl3_offload import sizing  # noqa: E402

CUDA = hardware.engine_cuda()
DRIVER = hardware.lib_value("DRIVER_MIN")

W = "windows-wsl"
L = "linux"
# The engine's steps. They are this engine's (exl3-offload, the only one the manager runs today), so they stay
# written out here rather than read from the catalog's build steps.
BASE_STEPS = [
    {"id": "wsl", "title": "WSL 2", "platforms": [W], "blocks": True,
     "desc": "Installs or updates the Windows Subsystem for Linux. Windows asks for administrator approval; a first "
             "install usually needs a restart."},
    {"id": "wslconfig", "title": "WSL memory limit", "platforms": [W], "blocks": True,
     "desc": "Writes %USERPROFILE%\\.wslconfig so WSL can use most of the RAM (its default is half) for the expert "
             "cache. Restarts WSL."},
    {"id": "distro", "title": "Ubuntu 24.04", "platforms": [W], "blocks": True,
     "desc": "Installs the Ubuntu-24.04 distro on the chosen drive (use your fastest NVMe) with systemd enabled."},
    {"id": "driver", "title": "NVIDIA driver", "platforms": [L], "blocks": False,
     "desc": f"Checks for driver {DRIVER} or newer (CUDA {CUDA}). Installing one needs a reboot (and key enrollment "
             "with Secure Boot), so this step shows the commands instead of running them."},
    {"id": "base", "title": f"Build tools + CUDA {CUDA}", "platforms": [W, L], "blocks": True, "script": "10_base.sh",
     "desc": f"Compiler, git, Python headers and NVIDIA's CUDA {CUDA} toolkit (no driver packages). Needs root/sudo."},
    {"id": "engine", "title": "Engine + Python environment", "platforms": [W, L], "blocks": True,
     "script": "20_env.sh",
     "desc": "Puts this repo's engine code where it runs (on Windows: a clone inside WSL, updated to this "
             f"checkout's commit) and creates its Python environment with PyTorch (CUDA {CUDA})."},
    {"id": "build", "title": "Compile GPU kernels", "platforms": [W, L], "blocks": True, "script": "30_build.sh",
     "desc": "Builds exllamav3 and the offload kernels for this GPU's architecture. Re-run after changing GPUs."},
]


def model_steps(cat=None):
    """One step per catalog model the engine runs: setup/50_model.sh downloads it at its pinned revision (checking
    every file) and packs its expert store. The step id (m305) is also the key setup/check.sh reports it under."""
    out = []
    for m in (cat or catalog.load()).models_for(sizing.ENGINE):
        src, paths, disk = m["source"], m["paths"], m["disk"]
        args = [m["quant"], "--dir", paths["model_dir"], "--store", paths["store"], "--repo", src["huggingface"],
                "--revision", src["revision"]] + (["--branch", src["branch"]] if src.get("branch") else [])
        out.append({"id": catalog.setup_id(m), "title": f"Model: {m['label']}", "platforms": [W, L], "blocks": False,
                    "script": "50_model.sh", "args": args,
                    # prepared_gb is this engine's expert store
                    "desc": f"~{disk['download_gb']:g} GB download + ~{disk['prepared_gb']:g} GB expert store. "
                            "Resumable.",
                    "model": m["id"], "quant": m["quant"], "curation": catalog.status(m),
                    "offered": catalog.offered(m)})
    return out


def all_steps():
    return BASE_STEPS + model_steps()


def model_specs(cat=None):
    """setup/check.sh's arguments for the models: KEY=MODEL_DIR:STORE, e.g. m305=model:store/qwen.bin."""
    return [f"{catalog.setup_id(m)}={m['paths']['model_dir']}:{m['paths']['store']}"
            for m in (cat or catalog.load()).models_for(sizing.ENGINE)]


class Setup:
    def __init__(self, svc):
        self.svc = svc
        self.host = svc.host
        self.status = {}
        self.status = {s["id"]: "unknown" for s in self.steps}
        self.hw = None
        self.out = collections.deque(maxlen=5000)
        self.seq = 0
        self.running = None
        self.checking = False

    def emit(self, line):
        self.seq += 1
        self.out.append((self.seq, line.rstrip("\r\n")))

    @property
    def steps(self):
        """This platform's steps, as setup shows them: a model's only when it is on offer (catalog.offered: curated,
        or from before curation), or the offer_uncurated setting is on, or it is installed, running or the chosen
        model already."""
        cfg = getattr(self.svc, "cfg", None) or {}
        out = []
        for s in all_steps():
            if self.host.kind not in s["platforms"]:
                continue
            if "model" in s and not (s["offered"] or cfg.get("offer_uncurated") or s["quant"] == cfg.get("quant")
                                     or self.status.get(s["id"]) in ("ok", "running")):
                continue
            out.append(s)
        return out

    @property
    def blocking(self):
        step = self.step(self.running) if self.running is not None else None
        return bool(step and step["blocks"])

    def step(self, sid):
        """A step by id, shown or not: `repertory setup STEP` can run a model step that isn't on offer."""
        return next((s for s in all_steps() if s["id"] == sid and self.host.kind in s["platforms"]), None)

    async def check(self):
        # one at a time, and a reading from now: one already running may have started before a step finished
        while self.checking:
            await asyncio.sleep(0.2)
        self.checking = True
        try:
            hw = await asyncio.to_thread(hardware.detect, self.host, self.svc.cfg)
            self.hw = hw
            self.status.update(self.statuses(hw))
            if self.running:
                self.status[self.running] = "running"
            log.info("setup check: %s", {k: v for k, v in self.status.items()})
        except Exception as e:  # noqa: BLE001 - detection must never take the service down
            log.exception("setup check failed: %s", e)
        finally:
            self.checking = False
        return self.hw

    def statuses(self, hw):
        e = hw.get("engine") or {}
        st = {}
        if IS_WINDOWS:
            w, wc = hw["wsl"], hw["wslconfig"]
            st["wsl"] = "ok" if w["installed"] and w.get("default_version") in (2, None) else "missing"
            want = hardware.wsl_memory_gb(hw["ram"]["total_gb"])
            st["wslconfig"] = "ok" if (wc.get("memory_gb") or 0) >= want * 0.9 else "missing"
            st["distro"] = "ok" if w["distro_present"] and w["distro_wsl_version"] == 2 else "missing"
        else:
            st["driver"] = "missing" if not hw["gpus"] else "outdated" if hardware.driver_too_old(hw) else "ok"
        models = [s["id"] for s in model_steps()]
        if not e or "error" in e:
            for sid in ("base", "engine", "build", *models):
                st[sid] = "missing" if (IS_WINDOWS and st.get("distro") == "missing") else "unknown"
            return st
        st["base"] = "ok" if e.get("cuda") == "1" and e.get("driver") == "1" else "missing"
        if e.get("engine") != "1" and e.get("engine_head") and IS_WINDOWS:
            st["engine"] = "outdated"           # an older (pre-integration) engine checkout: updating converts it
        elif e.get("engine") != "1" or e.get("env") != "1":
            st["engine"] = "missing"
        elif IS_WINDOWS and hw["repo"]["head"] and e.get("engine_head") != hw["repo"]["head"]:
            st["engine"] = "outdated"
        else:
            st["engine"] = "ok"
        # built, but from older kernel sources than the engine checkout's (an update changed them): compile again
        st["build"] = ("missing" if e.get("build") != "1" else "outdated" if e.get("build_src") == "0" else "ok")
        for sid in models:
            st[sid] = "ok" if e.get(sid) == "1" else "missing"
        return st

    async def run(self, sid):
        if self.running:
            return False, f"'{self.running}' is already running"
        step = self.step(sid)
        if not step:
            return False, "unknown step"
        self.running = sid
        self.status[sid] = "running"
        asyncio.create_task(self._run(step))
        return True, "started"

    def suggested_location(self):
        """Distro location: the configured one, else \\WSL\\<distro> on the fixed drive with the most free space."""
        cfg = self.svc.cfg
        if cfg["wsl_location"]:
            return cfg["wsl_location"]
        best = max((p for p in psutil.disk_partitions() if "fixed" in p.opts),
                   key=lambda p: psutil.disk_usage(p.mountpoint).free, default=None)
        drive = best.mountpoint if best else "C:\\"
        return str(Path(drive) / "WSL" / cfg["distro"])

    async def _run(self, step):
        host = self.host
        self.emit(f"===== {step['title']} =====")
        rc = 1
        try:
            if step["blocks"] and self.svc.be.state != "stopped" and not self.svc.observe:
                self.emit("stopping the model first (this step changes the engine or WSL)")
                await self.svc.stop("setup: " + step["title"])
            sid = step["id"]
            if sid == "wsl":
                rc = await self._wsl()
            elif sid == "wslconfig":
                rc = await self._wslconfig()
            elif sid == "distro":
                rc = await self._distro()
            elif sid == "driver":
                rc = self._driver()
            else:
                script = host.to_engine_path(REPO / "setup" / step["script"])
                args = [host.engine_dir, *step.get("args", [])]
                if sid == "engine" and IS_WINDOWS:
                    args.append(host.to_engine_path(REPO))        # clone/update the engine from this checkout
                if step.get("script") == "50_model.sh" and host.cfg.get("legacy_engine_dir"):
                    args.append(host.cfg["legacy_engine_dir"])    # models QwenFlash already downloaded: link them
                rc = await host.stream(host.engine_argv("bash", script, *args), self.emit)
                if rc == 10:
                    self.emit("(needs sudo: run the command above in a terminal, then press Re-check)")
        except Exception as e:  # noqa: BLE001
            self.emit(f"ERROR: {e}")
            rc = 1
        ok = rc == 0
        self.emit(f"===== {step['title']}: {'done' if ok else f'FAILED (exit {rc})'} =====")
        self.svc.event("setup", f"{step['title']}: {'done' if ok else 'failed'}")
        if step["id"] in ("engine", "build") and ok:
            # an updated engine, or newly compiled kernels, may resize in place
            self.svc.live_unsupported = self.svc.live_ram_unsupported = False
        self.running = None
        await self.check()
        if not ok:
            self.status[step["id"]] = "failed"

    async def _wsl(self):
        hw = self.hw or await self.check()
        if hw and hw["wsl"]["installed"]:
            self.emit("WSL is installed; updating it")
            return await self.host.stream([WSL_EXE, "--update"], self.emit)
        self.emit("installing WSL (approve the administrator prompt)...")
        ps = ("Start-Process ($env:SystemRoot + '/System32/wsl.exe') -ArgumentList '--install','--no-distribution' "
              "-Verb RunAs -Wait -PassThru | ForEach-Object { exit $_.ExitCode }")
        rc = await self.host.stream([POWERSHELL_EXE, "-NoProfile", "-Command", ps], self.emit)
        if rc == 0:
            self.emit("WSL installed. If Windows asks for a restart, restart, then reopen Repertory and continue.")
        return rc

    async def _wslconfig(self):
        total_gb = psutil.virtual_memory().total / GB
        mem = hardware.wsl_memory_gb(total_gb)
        p = Path.home() / ".wslconfig"
        lines = p.read_text(errors="replace").splitlines() if p.exists() else []
        if p.exists():
            bak = p.with_name(f".wslconfig.{time.strftime('%Y%m%d-%H%M%S')}.bak")   # every earlier version kept
            bak.write_text("\n".join(lines) + "\n")
            self.emit(f"backed up the existing file to {bak}")
        # update [wsl2] memory/processors/swap in place; keep every other setting the user has
        want = {"memory": f"{mem}GB", "processors": str(os.cpu_count()), "swap": "8GB"}
        out, section, seen = [], None, set()
        for line in lines:
            m = re.match(r"\s*\[(.+?)\]", line)
            if m:
                if section == "wsl2":
                    out += [f"{k}={v}" for k, v in want.items() if k not in seen]
                section = m.group(1).strip().lower()
            elif section == "wsl2":
                k = line.split("=", 1)[0].strip().lower()
                if k in want:
                    out.append(f"{k}={want[k]}")
                    seen.add(k)
                    continue
            out.append(line)
        if section == "wsl2":
            out += [f"{k}={v}" for k, v in want.items() if k not in seen]
        elif "wsl2" not in {re.sub(r"[\[\]\s]", "", x).lower() for x in lines if x.strip().startswith("[")}:
            out = ["[wsl2]", "# Repertory: most of the RAM for the expert cache",
                   *[f"{k}={v}" for k, v in want.items()], ""] + out
        p.write_text("\n".join(out).strip() + "\n")
        self.emit(f"wrote {p}: memory={mem}GB processors={os.cpu_count()} swap=8GB")
        self.emit("restarting WSL so the limit applies (wsl --shutdown)")
        return await self.host.stream([WSL_EXE, "--shutdown"], self.emit)

    async def restrict_folder(self, path):
        """Let only this user, SYSTEM and Administrators into the distro's folder. Under a drive root, Windows lets
        every signed-in account modify new files, and the disk image holds the engine that runs as root."""
        import win32api
        import win32security
        token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32security.TOKEN_QUERY)
        me = win32security.ConvertSidToStringSid(win32security.GetTokenInformation(token, win32security.TokenUser)[0])
        Path(path).mkdir(parents=True, exist_ok=True)
        self.emit(f"limiting {path} to this user, SYSTEM and Administrators")
        return await self.host.stream([str(SYSTEM32 / "icacls.exe"), path, "/inheritance:r", "/grant:r",
                                       f"*{me}:(OI)(CI)F", "*S-1-5-18:(OI)(CI)F", "*S-1-5-32-544:(OI)(CI)F"],
                                      self.emit)

    async def _distro(self):
        cfg = self.svc.cfg
        hw = self.hw or await self.check()
        if not (hw and hw["wsl"]["distro_present"]):
            loc = self.suggested_location()
            rc = await self.restrict_folder(loc)
            if rc != 0:
                return rc
            self.emit(f"installing {cfg['distro']} at {loc}")
            rc = await self.host.stream([WSL_EXE, "--install", "-d", cfg["distro"], "--location", loc,
                                         "--no-launch"], self.emit)
            if rc != 0:
                return rc
            self.emit("first start (unpacks the distro)...")
            await self.host.stream([WSL_EXE, "-d", cfg["distro"], "-u", "root", "--exec", "true"], self.emit)
        # the engine's launcher uses systemd scopes for its RAM guard
        rc = await self.host.stream(self.host.engine_argv(
            "bash", "-c", "grep -q '^systemd=true' /etc/wsl.conf 2>/dev/null && echo 'systemd already enabled' || "
                          "{ printf '[boot]\\nsystemd=true\\n' >> /etc/wsl.conf; echo 'enabled systemd'; }"), self.emit)
        await self.host.stream([WSL_EXE, "--terminate", cfg["distro"]], self.emit)
        return rc

    def _driver(self):
        hw = self.hw or {}
        old = hardware.driver_too_old(hw)
        if hw.get("gpus") and not old:
            self.emit(f"driver {hw['driver'].get('version')} (CUDA {hw['driver'].get('cuda')}) sees "
                      f"{', '.join(g['name'] for g in hw['gpus'])}")
            return 0
        if old:
            self.emit(f"Driver {hw['driver'].get('version')} supports CUDA {old[0]}; the engine is built with CUDA "
                      f"{old[1]}.")
        else:
            self.emit("No working NVIDIA driver.")
        self.emit(f"Install driver {DRIVER} or newer, then reboot:")
        self.emit(f"  Ubuntu: sudo apt install nvidia-driver-{DRIVER}-open   (or a newer nvidia-driver-*-open)")
        self.emit("  Other distributions: https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/")
        return 1

    def report(self, since=0):
        return {"platform": self.host.kind, "steps": [dict(s, status=self.status.get(s["id"], "unknown"))
                                                      for s in self.steps],
                "running": self.running, "checking": self.checking, "seq": self.seq,
                "lines": [[n, t] for n, t in self.out if n > since][-800:],
                "location": self.suggested_location() if IS_WINDOWS else None}
