"""Where the engine runs and how the machine is observed.

  WindowsWSL   manager on Windows, engine in a WSL2 distro (commands go through wsl.exe; Windows performance
               counters see other apps' GPU use, NVML sees the whole device)
  NativeLinux  manager and engine on the same Linux host (NVML per-process data separates our server from others)

Both drive the engine through the same scripts in the engine checkout: run.sh and bin/repertory-backend.
"""
import asyncio
import collections
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import psutil

from .config import IS_WINDOWS, REPO, WSL_EXE, log
from .nvml import Nvml

NO_WINDOW = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
BROWSERS = [r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe",
            r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe",
            r"%ProgramFiles%\Google\Chrome\Application\chrome.exe",
            r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe",
            r"%ProgramFiles%\BraveSoftware\Brave-Browser\Application\brave.exe",
            r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe"]


class LogFollower(threading.Thread):
    """`tail -F` of the server log in the engine's environment, kept in memory for the dashboard. On Windows the
    long-lived wsl.exe also keeps the distro from idling out while the manager runs."""

    def __init__(self, argv, env=None):
        super().__init__(daemon=True)
        self.argv, self.env = argv, env
        self.lines = collections.deque(maxlen=3000)
        self.proc = None
        self.stopped = False
        self.on_line = None

    def run(self):
        wait = 3
        while not self.stopped:
            started = time.time()
            try:
                self.proc = subprocess.Popen(self.argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                             stdin=subprocess.DEVNULL, env=self.env, creationflags=NO_WINDOW)
                _kill_with_parent(self.proc)
                for raw in iter(self.proc.stdout.readline, b""):
                    line = raw.decode("utf-8", "replace").rstrip("\r\n")
                    self.lines.append(line)
                    if self.on_line:
                        try:
                            self.on_line(line)
                        except Exception:  # noqa: BLE001
                            pass
            except OSError as e:
                log.warning("log follower failed: %s", e)
            # back off while the engine environment doesn't exist yet (e.g. before WSL is set up)
            wait = 3 if time.time() - started > 30 else min(60, wait * 2)
            time.sleep(wait)

    def clear(self):
        self.lines.clear()

    def tail(self, n=300):
        return "\n".join(list(self.lines)[-n:])

    def stop(self):
        self.stopped = True
        if self.proc:
            try:
                self.proc.kill()
            except OSError:
                pass


_JOB = None


def _kill_with_parent(proc):
    """Windows: put a long-lived child in a job object that closes with us, so a crashed manager leaves no wsl.exe."""
    global _JOB
    if not IS_WINDOWS:
        return
    try:
        import win32api
        import win32con
        import win32job
        if _JOB is None:
            _JOB = win32job.CreateJobObject(None, "")
            info = win32job.QueryInformationJobObject(_JOB, win32job.JobObjectExtendedLimitInformation)
            info["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            win32job.SetInformationJobObject(_JOB, win32job.JobObjectExtendedLimitInformation, info)
        h = win32api.OpenProcess(win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE, False, proc.pid)
        win32job.AssignProcessToJobObject(_JOB, h)
    except Exception as e:  # noqa: BLE001 - best effort
        log.warning("could not tie child %s to the manager's lifetime: %s", proc.pid, e)


class Host:
    kind = "linux"
    label = "Linux"
    ram_settle_s = 0        # seconds after a stop before host RAM reflects it
    ram_return_s = 0        # how long a start after a stop may wait for the stopped server's RAM (Service.ram_back)
    env = None

    def __init__(self, cfg):
        self.cfg = cfg
        self.nvml_error = None
        try:
            self.nvml = Nvml()
            self.gpu = self.nvml.devices[0]
        except OSError as e:
            self.nvml, self.gpu, self.nvml_error = None, None, str(e)
        self.follower = None
        self._own_pids, self._own_t = set(), 0.0

    # ------------------------------------------------------------------ engine environment
    @property
    def engine_dir(self):
        return self.cfg["repo_dir"] or str(REPO)

    def engine_argv(self, *args):
        return list(args)

    def run(self, *args, timeout=60):
        return subprocess.run(self.engine_argv(*args), capture_output=True, timeout=timeout, env=self.env,
                              creationflags=NO_WINDOW, encoding="utf-8", errors="replace")

    async def stream(self, argv, emit):
        """Run argv (already wrapped for the engine environment), emitting output lines. Returns the exit code."""
        proc = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.STDOUT, env=self.env,
                                                    creationflags=NO_WINDOW)
        buf = b""
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk.replace(b"\x00", b"")
            *lines, buf = re.split(rb"\r\n|\n|\r", buf)
            for line in lines:
                if line.strip():
                    emit(line.decode("utf-8", "replace"))
        if buf.strip():
            emit(buf.decode("utf-8", "replace"))
        return await proc.wait()

    def to_engine_path(self, local):
        return str(Path(local).resolve())

    def helper(self, *args, timeout=60):
        return self.run("bash", f"{self.engine_dir}/bin/repertory-backend", *args, timeout=timeout)

    def backend_start(self, env, args):
        r = self.helper("start", *[f"{k}={v}" for k, v in env.items()], "--", *args, timeout=60)
        out = (r.stdout or r.stderr).strip()
        m = re.match(r"started (\d+)", out)
        return (True, int(m.group(1))) if m else (False, out or f"exit {r.returncode}")

    def backend_stop(self, timeout=30):
        return self.helper("stop", str(timeout), timeout=timeout * 3 + 30).stdout.strip()

    def backend_status(self):
        """('running', pid) | ('stray', [pids]) | ('stopped', None) | ('unknown', error)"""
        try:
            r = self.helper("status", timeout=30)
        except (OSError, subprocess.TimeoutExpired) as e:
            return "unknown", str(e)
        parts = (r.stdout or "").split()
        if not parts:
            return "unknown", (r.stderr or "").strip()[-200:]
        if parts[0] == "running":
            return "running", int(parts[1])
        if parts[0] == "stray":
            return "stray", [int(p) for p in parts[1:]]
        return "stopped", None

    def start_follower(self):
        if self.follower and self.follower.is_alive():
            return self.follower
        log_path = f"{self.engine_dir}/.repertory/backend.log"
        d, log = shlex.quote(f"{self.engine_dir}/.repertory"), shlex.quote(log_path)
        self.follower = LogFollower(self.engine_argv("bash", "-c", f"mkdir -p {d}; touch {log}; "
                                                                   f"exec tail -n 400 -F {log}"), self.env)
        self.follower.start()
        return self.follower

    def after_ready(self):
        pass

    def after_stop(self):
        pass

    # ------------------------------------------------------------------ observation
    def own_pids(self):
        """PIDs of our model server, when it runs on this OS (refreshed every few seconds)."""
        now = time.time()
        if now - self._own_t > 4:
            pids = set()
            for p in psutil.process_iter(["pid", "cmdline"]):
                try:
                    if any("engines/exl3_offload/server.py" in (a or "") for a in (p.info["cmdline"] or [])):
                        pids.add(p.info["pid"])
                except (psutil.Error, TypeError):
                    continue
            self._own_pids, self._own_t = pids, now
        return self._own_pids

    def gpu_sample(self):
        if not self.gpu:
            return {"vram_total": 0.0, "vram_used": 0.0, "gpu_util": 0.0}
        total, used = self.gpu.memory()
        return {"vram_total": total, "vram_used": used, "gpu_util": self.gpu.utilization()}

    def processes(self):
        names, running = {}, set()
        for p in psutil.process_iter(["pid", "name"]):
            n = (p.info["name"] or "").lower()
            names[p.info["pid"]] = n
            running.add(n)
        return names, running

    def sample(self, own_busy=False):
        s = self.gpu_sample()
        names, running = self.processes()
        ignore = {x.lower() for x in self.cfg["util_ignore"]}
        own = self.own_pids()
        ext_abs, top = None, []
        if self.gpu:
            procs = self.gpu.processes()
            known_own = [procs.get(p) for p in own if p in procs]
            if not own:
                ext_abs = s["vram_used"]
            elif known_own and all(v is not None for v in known_own):
                ext_abs = max(0.0, s["vram_used"] - sum(known_own))
            util = self.gpu.process_utilization()
            if util is not None:
                top = sorted(((v, names.get(pid, f"pid {pid}")) for pid, v in util.items()
                              if pid not in own and v >= 1 and names.get(pid, "") not in ignore), reverse=True)[:5]
            elif not own_busy and s["gpu_util"] >= 1:
                top = [(s["gpu_util"], "GPU (per-app use not reported)")]
        vm = psutil.virtual_memory()
        s.update({"t": time.time(), "ext_vram_abs": ext_abs, "perproc_ded": 0.0,
                  "ext_util": min(100.0, top[0][0]) if top else 0.0,
                  "ext_util_top": [[n, round(min(100.0, v))] for v, n in top],
                  "ram_avail": float(vm.available), "ram_total": float(vm.total), "running": running})
        return s

    # ------------------------------------------------------------------ desktop integration
    def autostart_enabled(self):
        return self._unit_path().exists()

    def _unit_path(self):
        return Path.home() / ".config" / "systemd" / "user" / "repertory.service"

    def set_autostart(self, on):
        unit = self._unit_path()
        if on:
            unit.parent.mkdir(parents=True, exist_ok=True)
            unit.write_text(
                "[Unit]\nDescription=Repertory manager (model lifecycle, OpenAI API, dashboard)\n"
                "After=network-online.target\n\n[Service]\n"
                f"WorkingDirectory={REPO}\nExecStart={sys.executable} -m repertory serve\nRestart=on-failure\n"
                "# the model server runs in its own scope and must survive manager restarts\nKillMode=process\n\n"
                "[Install]\nWantedBy=default.target\n")
            cmds = [["systemctl", "--user", "daemon-reload"], ["systemctl", "--user", "enable", "repertory.service"]]
        else:
            cmds = [["systemctl", "--user", "disable", "repertory.service"]]
        for c in cmds:
            subprocess.run(c, capture_output=True, timeout=30)
        if not on and unit.exists():
            unit.unlink()
            subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True, timeout=30)
        return True, ("enabled; for start at boot without a login run: sudo loginctl enable-linger $USER"
                      if on else "disabled")

    def open_ui(self, url):
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            webbrowser.open(url)
        else:
            print(f"Repertory dashboard: {url}")

    def describe(self):
        info = {"kind": self.kind, "label": self.label}
        try:
            osr = dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines() if "=" in line)
            info["os"] = osr.get("PRETTY_NAME", "").strip('"')
        except OSError:
            info["os"] = "Linux"
        info["kernel"] = os.uname().release
        info["in_wsl"] = "microsoft" in info["kernel"].lower()
        info["systemd"] = Path("/run/systemd/system").exists()
        return info


class NativeLinux(Host):
    kind = "linux"
    label = "Linux"


class WindowsWSL(Host):
    kind = "windows-wsl"
    label = "Windows + WSL2"
    ram_settle_s = 30       # WSL hands freed memory back to Windows over ~15-30 s
    ram_return_s = 90       # ... and Windows has it all only after 60-80 s
    env = dict(os.environ, WSL_UTF8="1")

    def __init__(self, cfg):
        super().__init__(cfg)
        self._pdh = None

    @property
    def engine_dir(self):
        return self.cfg["repo_dir"] or "/root/repertory"

    def engine_argv(self, *args):
        return [WSL_EXE, "-d", self.cfg["distro"], "-u", "root", "--exec", *args]

    def to_engine_path(self, local):
        p = str(Path(local).resolve())
        return f"/mnt/{p[0].lower()}{p[2:].replace(chr(92), '/')}"

    def after_ready(self):
        # the launcher's library pre-read and checkpoint loading leave page cache in the VM that Windows counts as
        # used; the loaded model no longer needs it (mapped pages in use are kept)
        self.helper("dropcache", "1", timeout=60)

    def after_stop(self):
        self.helper("dropcache", "3", timeout=60)

    # Windows' per-process GPU counters do not see WSL at all, so they measure other apps cleanly: utilization
    # exactly, dedicated memory as a change since our start (shared surfaces are over-counted in absolute terms).
    def _pdh_open(self):
        import pywintypes
        import win32pdh
        if self._pdh:
            try:
                win32pdh.CloseQuery(self._pdh[0])
            except pywintypes.error:
                pass
        q = win32pdh.OpenQuery()
        c_mem = win32pdh.AddEnglishCounter(q, r"\GPU Process Memory(*)\Dedicated Usage")
        c_eng = win32pdh.AddEnglishCounter(q, r"\GPU Engine(*)\Utilization Percentage")
        win32pdh.CollectQueryData(q)
        self._pdh = (q, c_mem, c_eng)

    def sample(self, own_busy=False):
        import pywintypes
        import win32pdh
        if not self._pdh:
            self._pdh_open()
            time.sleep(1)
        q, c_mem, c_eng = self._pdh

        def arr(c, fmt):
            try:
                return win32pdh.GetFormattedCounterArray(c, fmt)
            except pywintypes.error:
                return {}
        s = self.gpu_sample()
        try:
            win32pdh.CollectQueryData(q)
        except pywintypes.error:
            self._pdh_open()
            raise
        ded = float(sum(arr(c_mem, win32pdh.PDH_FMT_LARGE).values()))
        by_type = collections.Counter()
        for inst, v in arr(c_eng, win32pdh.PDH_FMT_DOUBLE).items():
            m = re.match(r"pid_(\d+)_.*engtype_(\w+)", inst)
            if m:
                by_type[(int(m.group(1)), m.group(2))] += v
        per_pid = {}
        for (pid, _eng), v in by_type.items():
            per_pid[pid] = max(per_pid.get(pid, 0.0), v)
        names, running = self.processes()
        ignore = {x.lower() for x in self.cfg["util_ignore"]}
        top = sorted(((v, names.get(pid, f"pid {pid}")) for pid, v in per_pid.items()
                      if v >= 1 and names.get(pid, "") not in ignore), reverse=True)[:5]
        vm = psutil.virtual_memory()
        s.update({"t": time.time(), "ext_vram_abs": None, "perproc_ded": ded,
                  "ext_util": min(100.0, top[0][0]) if top else 0.0,
                  "ext_util_top": [[n, round(min(100.0, v))] for v, n in top],
                  "ram_avail": float(vm.available), "ram_total": float(vm.total), "running": running})
        return s

    STARTUP_LNK = Path(os.environ.get("APPDATA", "")) / r"Microsoft\Windows\Start Menu\Programs\Startup\Repertory.lnk"

    def autostart_enabled(self):
        return self.STARTUP_LNK.exists()

    def set_autostart(self, on):
        if on:
            pyw = Path(sys.executable).with_name("pythonw.exe")
            make_shortcut(self.STARTUP_LNK, pyw if pyw.exists() else Path(sys.executable),
                          "-m repertory serve", REPO, "Repertory service")
        elif self.STARTUP_LNK.exists():
            self.STARTUP_LNK.unlink()
        return True, "enabled" if on else "disabled"

    def open_ui(self, url):
        for exe in [self.cfg.get("ui_browser") or ""] + [os.path.expandvars(b) for b in BROWSERS]:
            if exe and Path(exe).exists():
                subprocess.Popen([exe, f"--app={url}", "--window-size=1360,940"], creationflags=NO_WINDOW)
                return
        webbrowser.open(url)

    def describe(self):
        import platform
        return {"kind": self.kind, "label": self.label, "os": f"Windows {platform.release()} "
                f"(build {platform.version()})", "in_wsl": False, "systemd": None}


def make_shortcut(lnk, target, args, workdir, desc="Repertory"):
    """A Windows shortcut, made through COM (pywin32): no command line, so no quoting to get wrong."""
    import win32com.client
    s = win32com.client.Dispatch("WScript.Shell").CreateShortcut(str(lnk))
    s.TargetPath, s.Arguments, s.WorkingDirectory, s.Description = str(target), args, str(workdir), desc
    s.Save()


def make_host(cfg):
    return WindowsWSL(cfg) if IS_WINDOWS else NativeLinux(cfg)


def which(name):
    return shutil.which(name)
