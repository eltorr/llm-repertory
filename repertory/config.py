"""Paths, settings and their defaults.

Settings and logs live outside the checkout, so updating or resetting the repo never loses them:
  Windows  %LOCALAPPDATA%\\Repertory
  Linux    $XDG_STATE_HOME/repertory (default ~/.local/state/repertory)
REPERTORY_HOME overrides both. On its first start Repertory takes over the settings of QwenFlash, as it was called
before (import_legacy).
"""
import json
import logging
import math
import os
import re
import secrets
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GB = 1e9
GIB = 1024 ** 3
SERVED_NAME = "qwen3.8-flash-next"
IS_WINDOWS = sys.platform == "win32"
# Windows programs by full path, and never a same-named program from the current folder (git, nvidia-smi...)
SYSTEM32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
WSL_EXE = str(SYSTEM32 / "wsl.exe")
POWERSHELL_EXE = str(SYSTEM32 / "WindowsPowerShell" / "v1.0" / "powershell.exe")
if IS_WINDOWS:
    os.environ["NoDefaultCurrentDirectoryInExePath"] = "1"


LEGACY = "QwenFlash"                    # the project's earlier name, whose settings import_legacy takes over
LEGACY_ENGINE_DIR = "/root/qwen-flash"  # its engine checkout (and models) inside WSL, unless repo_dir said otherwise


def default_state_dir(product):
    if IS_WINDOWS:
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / product
    return Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / product.lower()


def state_dir():
    d = Path(os.environ["REPERTORY_HOME"]) if os.environ.get("REPERTORY_HOME") else default_state_dir("Repertory")
    d.mkdir(parents=True, exist_ok=True)
    (d / "logs").mkdir(exist_ok=True)
    if not IS_WINDOWS:
        try:
            os.chmod(d, 0o700)    # config.json holds the private link's token (on Windows the profile is private)
        except OSError:
            pass
    return d


STATE = state_dir()
CONFIG_PATH = STATE / "config.json"
LOG_DIR = STATE / "logs"

DEFAULTS = {
    # where the engine lives. Windows: inside this WSL distro; Linux: this checkout ("" = the repo itself)
    "distro": "Ubuntu-24.04",
    "wsl_location": "",                 # where a new distro is installed ("" = suggest the drive with most space)
    "repo_dir": "",                     # engine checkout as the engine sees it ("" = platform default)
    "legacy_engine_dir": "",            # QwenFlash's engine checkout (set when its settings are imported): setup links
                                        # the models and expert stores there instead of downloading them again
    "quant": "3.05",
    "context": 262144,
    "listen_host": "127.0.0.1",
    "listen_port": 30000,
    "backend_port": 30001,
    "dashboard_token": "",              # the dashboard's private link for other devices (generated when needed)
    "remote_host": "",                  # this PC's name for other devices, used in that link ("" = computer name)
    "auto_switch": True,
    "wake_on_request": True,
    "preload_on_launch": False,
    "preload_when_free": False,
    "upgrade_when_idle": True,
    "idle_unload_min": 0,
    "wake_override_min": 5,
    "pressure_response": "lean",        # lean | pause
    "live_resize": True,                # full <-> lean without a restart when only VRAM is short (the engine resizes
                                        # its VRAM cache in place, ~0.5 s); RAM pressure and other changes still
                                        # restart. Checked by: repertory bench current --workload resize
    "live_ram_resize": True,            # with live_resize, the RAM cache too: RAM pressure shrinks it in place
                                        # (3-4 s) and re-saturating grows it back (10-20 s), instead of restarting.
                                        # Checked by: repertory bench current --workload resize
    "gpu_busy_action": "lean",          # lean | pause
    "ext_util_pct": 50,
    "ext_vram_lean_gb": 0.0,            # other apps' VRAM that sends the model to lean (0 = a quarter of the card)
    "vram_min_free_gb": 0.8,
    "ram_lean_gb": 6.0,
    "ram_pause_gb": 3.0,
    "ram_reserve_gb": 6.0,              # RAM always left to the host OS and other apps
    "lean_vram_gb": 0.0,                # VRAM the model keeps when other apps need the GPU (0 = automatic)
    "lean_ram_gb": 8,
    "full_vram_headroom_gb": 0.0,       # VRAM left free in full mode (0 = ~10% of the card, 1.2-2.4 GB)
    "max_ram_cache_gib": 0,             # 0 = automatic (with the VRAM cache, 90% of the experts)
    "auto_recover": True,               # restart a crashed model, stepping its VRAM budget down after an OOM
    "watchlist": [],
    "util_ignore": ["dwm.exe", "csrss.exe", "system", "explorer.exe", "xorg", "gnome-shell", "kwin_x11",
                    "kwin_wayland"],
    "moderate_after_s": 10,
    "severe_after_s": 3,
    "recover_after_s": 30,
    "min_dwell_s": 60,
    "severe_max_wait_s": 120,
    "busy_wait_s": 90,
    "start_timeout_s": 600,
    "poll_s": 1.5,
    "ui_browser": "",
    # Optional port for other devices, such as over a tailnet (docs/manager.md): /v1 is the API, and the root is the
    # dashboard with its private link. 0 (the default) leaves it off.
    "network_port": 0,
}
# settings that only take effect when the service restarts
SERVICE_KEYS = {"listen_port", "backend_port", "listen_host", "repo_dir", "distro", "network_port"}
# settings that name programs, paths, ports or the private link: only config.json changes them, never the HTTP API
FILE_ONLY_KEYS = SERVICE_KEYS | {"wsl_location", "ui_browser", "dashboard_token", "legacy_engine_dir"}
# settings only this PC's own dashboard may change, not other devices with the private link
LOCAL_ONLY_KEYS = {"remote_host"}
# settings with a fixed set of values
CHOICES = {"pressure_response": ("lean", "pause"), "gpu_busy_action": ("lean", "pause")}
RENAMED = {"windows_ram_reserve_gb": "ram_reserve_gb"}

logging.basicConfig(filename=LOG_DIR / "manager.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("repertory")


def legacy_state_dir():
    """QwenFlash's settings folder: QWENFLASH_HOME, else its default. None when REPERTORY_HOME names a folder of its
    own and QWENFLASH_HOME doesn't (as in the tests): then nothing is imported."""
    if os.environ.get("QWENFLASH_HOME"):
        return Path(os.environ["QWENFLASH_HOME"])
    if os.environ.get("REPERTORY_HOME"):
        return None
    return default_state_dir(LEGACY)


def import_legacy():
    """The first start on a PC that ran QwenFlash: take over its config.json (every setting, and the dashboard token,
    so the private link and signed-in devices keep working) and learned.json (the limits it learned). Only while
    Repertory has no config.json of its own, so it happens once. On Windows, Repertory keeps an engine checkout of its
    own (it never takes over QwenFlash's, which may still be serving): QwenFlash's is remembered as legacy_engine_dir,
    where setup finds the models and stores to link. True if it imported."""
    old = legacy_state_dir()
    if CONFIG_PATH.exists() or old is None or not (old / "config.json").is_file():
        return False
    try:
        data = json.loads((old / "config.json").read_text())
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as e:
        log.warning("QwenFlash's settings in %s are unreadable (%s); starting from the defaults", old, e)
        return False
    if IS_WINDOWS:
        data["legacy_engine_dir"] = data.pop("repo_dir", "") or LEGACY_ENGINE_DIR
    save_config(data)
    if (old / "learned.json").is_file() and not (STATE / "learned.json").exists():
        shutil.copy2(old / "learned.json", STATE / "learned.json")
    log.info("imported QwenFlash's settings from %s", old)
    return True


def load_config():
    import_legacy()
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text())
            if not isinstance(data, dict):
                raise ValueError("not a JSON object")
            for a, b in RENAMED.items():
                if a in data and b not in data:
                    data[b] = data.pop(a)
        except Exception as e:  # noqa: BLE001 - a bad config must not stop the service
            log.warning("config.json unreadable (%s); using defaults", e)
            data = {}
        for k, v in data.items():
            if k in DEFAULTS:
                try:
                    cfg[k] = coerce(k, v)   # values reach scripts and processes: keep each to its type
                except (TypeError, ValueError):
                    log.warning("config.json: ignoring the bad value of %s", k)
    return cfg


def save_config(cfg):
    """Write config.json readable by this user only: it holds the private link's token."""
    tmp = CONFIG_PATH.with_suffix(".tmp")
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        f.write(json.dumps(cfg, indent=2))
    if not IS_WINDOWS:
        os.chmod(tmp, 0o600)
    os.replace(tmp, CONFIG_PATH)


# number ranges; other numbers must be 0 or more
MINIMUM = {"poll_s": 0.5, "context": 4096, "listen_port": 1, "backend_port": 1}
MAXIMUM = {"context": 262144, "listen_port": 65535, "backend_port": 65535, "network_port": 65535}


def coerce(key, value):
    """A setting's value as its default's type; ValueError if it doesn't fit."""
    d = DEFAULTS[key]
    if isinstance(d, bool):
        return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "on", "yes")
    if isinstance(d, (int, float)):
        x = float(value)
        if not math.isfinite(x) or not MINIMUM.get(key, 0) <= x <= MAXIMUM.get(key, math.inf):
            raise ValueError(f"{key} is out of range")
        return int(x) if isinstance(d, int) else x
    if isinstance(d, list):
        if isinstance(value, str):
            value = [x.strip() for x in re.split(r"[,\n]", value)]
        return [str(x).strip() for x in value if str(x).strip()]
    if key in CHOICES and str(value) not in CHOICES[key]:
        raise ValueError(f"{key} must be one of {', '.join(CHOICES[key])}")
    return str(value)


def load_json(name, default):
    p = STATE / name
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def save_json(name, data):
    p = STATE / name
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, p)


def secret(name):
    """A random secret kept in the state folder, readable by this user only; made on first use."""
    p = STATE / name
    try:
        s = p.read_text().strip()
        if s:
            return s
    except OSError:
        pass
    s = secrets.token_urlsafe(24)
    with os.fdopen(os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        f.write(s)
    return s


def ensure_token(cfg, save=True):
    """The dashboard's token (its private link), made on first use. Every port needs it for the dashboard."""
    if not cfg["dashboard_token"]:
        cfg["dashboard_token"] = secrets.token_urlsafe(18)
        if save:
            save_config(cfg)
            log.info("generated the dashboard token (config.json dashboard_token)")
    return cfg["dashboard_token"]
