"""Repertory service: the OpenAI-compatible API, the resource-aware model lifecycle, the dashboard and its control API.

Clients use http://127.0.0.1:30000/v1 (this process). The engine's server runs on the backend port behind it and
is started, resized or stopped depending on what other apps are doing:

  full    the biggest VRAM budget that is free (minus headroom): fastest
  lean    a smaller budget that coexists with games and other GPU apps (same context window)
  paused  stopped: no VRAM/RAM; a request arriving while paused wakes it

Budgets come from the engine adapter's sizing.py (engines/exl3_offload/, shared with its run.sh). The engine runs
detached (bin/repertory-backend), so restarting
the manager adopts the running model instead of restarting it. Crashes are recovered automatically; a crash from
running out of VRAM or RAM lowers that budget for next time ("learned limits").
"""
import asyncio
import calendar
import collections
import hmac
import http.client
import ipaddress
import json
import math
import re
import secrets
import socket
import statistics
import subprocess
import threading
import time
import urllib.parse

from . import catalog, config
from .config import GB, GIB, LOG_DIR, REPO, SERVICE_KEYS, log
from .host import make_host
from .httpio import (BACKEND_KEY_HEADER, HTTPError, StreamStats, backend_request, close, http_get_json, http_ok,
                     http_post_json, oai_error, pipe, read_body, read_head, respond)
from .bench import Bench
from .setup import Setup

from engines.exl3_offload import sizing  # noqa: E402

# RAM the model uses beyond its expert RAM tier (process, torch, page cache; under WSL also the VM's own share).
# A first guess per platform; after each start the manager measures the real figure and uses that from then on.
RAM_OVERHEAD = sizing.RAM_OVERHEAD
RAM_GRACE_S = 90                 # after a start, page cache from loading takes a while to be handed back
FREED_S = 300                    # how long a stopped server's RAM counts as reusable for the next start (WSL)
RAM_BACK_OUT = 0.2               # a start after a stop waits while more than this share of the stopped server's RAM,
RAM_BACK_OUT_GB = 4.0            # or more than this if that is more, is still on its way back to Windows
RAM_GROW_MARGIN_GB = 1.0         # a RAM cache growing in the background stops below lean's threshold plus this
SOFT_CAP_S = 3600                # how long a RAM cache that ran short caps the next full starts
# How long the model's host may not answer (WSL down: out of disk or memory, or a crash) before a running model is
# taken as lost; checks run every 10 s, and a busy WSL can miss one
UNREACHABLE_S = 60
SWITCH_DRAIN_S = 600             # a request for another model waits this long for the running requests to finish
UI_FILE = REPO / "repertory" / "ui" / "index.html"
SIGNIN_FILE = REPO / "repertory" / "ui" / "signin.html"   # the root without the private link
PAGES = ("/", "/_switcher", "/_switcher/", "/_switcher/index.html")
COMPLETIONS = ("/v1/chat/completions", "/v1/responses", "/v1/completions")
# actions other devices may not take, even with the private link
LOCAL_ONLY_ACTIONS = {"quit", "quit_keep_model", "autostart_on", "autostart_off", "new_remote_link"}
# the server's own watchdogs (engines/exl3_offload/memory_guard.py) print "GUARD vram: ..." and "GUARD ram: ..."
OOM_VRAM = re.compile(r"out of memory|OutOfMemoryError|GUARD vram\b|Insufficient VRAM|cudaErrorMemoryAllocation", re.I)
OOM_RAM = re.compile(r"GUARD ram\b|MemoryError|Killed\b|oom-kill", re.I)
# the dashboard's cookie, and QwenFlash's (this project's earlier name): devices signed in before keep working
COOKIE, LEGACY_COOKIE = "repertory_token", "qf_token"


def model_not_found(msg):
    """OpenAI's error for a model it doesn't have."""
    return {"error": {"message": msg, "type": "invalid_request_error", "param": "model", "code": "model_not_found"}}


class Backend:
    def __init__(self):
        self.state = "stopped"      # stopped | starting | running | resizing | stopping | error
        self.tier = self.quant = self.context = self.pid = None
        self.target = None          # the tier a resize is switching to
        self.since = time.time()
        self.ready_at = self.startup_s = self.error = None
        self.own_vram = self.ram_gib = self.budget = None
        self.plan = None
        self.adopted = False
        self.ram_measured = True

    def set(self, state):
        self.state = state
        self.since = time.time()


class Monitor(threading.Thread):
    def __init__(self, svc):
        super().__init__(daemon=True)
        self.svc = svc
        self.lock = threading.Lock()
        self.latest = None

    def run(self):
        while True:
            try:
                s = self.svc.host.sample(own_busy=self.svc.in_flight > 0 or self.svc.bench.running)
                with self.lock:
                    self.latest = s
            except Exception as e:  # noqa: BLE001 - keep monitoring through transient counter errors
                log.warning("monitor sample failed: %s", e)
            time.sleep(max(0.5, self.svc.cfg["poll_s"]))

    def get(self):
        with self.lock:
            return self.latest


class Service:
    def __init__(self, cfg, observe=False):
        self.cfg = cfg
        self.observe = observe
        self.host = make_host(cfg)
        self.gpu_display = self._gpu_display()
        self.mon = Monitor(self)
        self.be = Backend()
        self.setup = Setup(self)
        self.bench = Bench(self)
        self.tlock = None
        self.task = None
        self.in_flight = self.held = self.served = self.failed = 0
        self.switching = None            # the quant a request is switching the model to (one switch at a time)
        self.routing = 0                 # requests choosing their model: not yet sent to the model server
        self.last_request = self.last_done = None
        self.override_until = 0.0
        self.paused_reason = None
        self.ram_before_start = None
        self.manual_lean = False
        self.ext_at_start = self.perproc_at_start = None
        self.live_unsupported = False    # the engine checkout has no live resize: switch tiers by restarting
        self.live_ram_unsupported = False    # it can't resize its RAM cache (older kernels): RAM changes restart
        self.ram_growing = None          # the engine is growing its RAM cache in the background: {"since", "from_gib"}
        self.ram_watch = False           # a check of that growth is in flight
        self.pressure = {"level": "none", "reasons": []}
        self.sev_since = self.mod_since = self.clear_since = None
        self.last_transition = 0.0
        self.last_stop_t = time.time()
        self.freed = None                # (time, free RAM, the server's RAM) at the last stop of a running server
        self.ram_soft_cap = None         # (time, GiB) after full ran short of RAM
        self.events = collections.deque(maxlen=300)
        self.hist = collections.deque(maxlen=400)
        self.metrics = collections.deque(maxlen=200)
        self._last_hist_t = 0
        self._last_alive_check = 0.0
        self.unreachable_since = None    # when the host first stopped answering the alive check
        self.recover = {"attempts": 0, "window_start": 0.0, "next_at": None, "tier": None}
        self.learned = config.load_json("learned.json", {})
        self.quit = None
        self.front_server, self._front_warned = None, False
        # only this manager may call the model server: it sends this key, which the server checks
        self.backend_key = config.secret("backend.key")
        self.backend_auth = {BACKEND_KEY_HEADER: self.backend_key}

    def _gpu_display(self):
        from .hardware import has_display
        try:
            return has_display(self.host.gpu.info()) if self.host.gpu else True
        except OSError:
            return True

    # ================================================================ bookkeeping
    def event(self, kind, msg):
        self.events.append([time.time(), kind, msg])
        log.info("[%s] %s", kind, msg)

    @property
    def gpu_name(self):
        return self.host.gpu.name if self.host.gpu else "no GPU"

    def learned_key(self, quant=None, context=None):
        return f"{self.gpu_name}|{quant or self.cfg['quant']}|{context or self.cfg['context']}"

    def learn(self, **kw):
        entry = self.learned.setdefault(self.learned_key(), {})
        entry.update(kw, t=time.time())
        config.save_json("learned.json", self.learned)

    def ext_vram(self, m):
        if m.get("ext_vram_abs") is not None:
            return m["ext_vram_abs"]
        b = self.bench.base
        if b is not None:   # a benchmark's server: other apps used what was in use before it started, plus their growth
            return max(0.0, min(m["vram_total"], b["vram_used"] + m["perproc_ded"] - b["perproc_ded"]))
        if self.perproc_at_start is None or self.be.state in ("stopped", "error"):
            return m["vram_used"]
        # Windows: per-process counters don't see WSL; other apps' change since our start is exact
        return max(0.0, min(m["vram_total"], self.ext_at_start + m["perproc_ded"] - self.perproc_at_start))

    # ================================================================ budgets (sizing.py)
    def lean_budget(self, m):
        if self.cfg["lean_vram_gb"] > 0:
            return self.cfg["lean_vram_gb"] * GB
        floor = sizing.min_budget(self.cfg["quant"], self.cfg["context"]) + 1 * GB
        return max(floor, min(16 * GB, 0.55 * m["vram_total"]))

    def headroom(self, m):
        """VRAM left free in full mode: the setting, or ~10% of the card (1.2-2.4 GB; 0.6 GB minimum on a card that
        drives no display) when 0 (automatic)."""
        if self.cfg["full_vram_headroom_gb"] > 0:
            return self.cfg["full_vram_headroom_gb"] * GB
        return sizing.headroom_bytes(m["vram_total"], self.gpu_display)

    def full_budget(self, m):
        b = m["vram_total"] - self.ext_vram(m) - self.headroom(m)
        cap = self.learned.get(self.learned_key(), {}).get("vram_cap")
        return min(b, cap) if cap else b

    def ext_lean_threshold(self, m):
        return self.cfg["ext_vram_lean_gb"] * GB if self.cfg["ext_vram_lean_gb"] > 0 else 0.25 * m["vram_total"]

    def plan_for(self, tier, m):
        cfg = self.cfg
        budget = self.full_budget(m) if tier == "full" else min(self.lean_budget(m), self.full_budget(m))
        return sizing.plan(cfg["quant"], budget, cfg["context"])

    def own_ram_estimate(self):
        if self.be.state != "running" or not self.be.tier:
            return 0.0
        return (self.be.ram_gib or self.cfg["lean_ram_gb"]) * GIB + self.ram_overhead()

    def ram_spare(self, m, reserve_gb=None):
        reserve = (reserve_gb if reserve_gb is not None else self.cfg["ram_reserve_gb"]) * GB
        return m["ram_avail"] + self.own_ram_estimate() - reserve

    def ram_for_start(self, m):
        """RAM a (re)started model can have: what is free, plus the model's own if it is running. Right after a
        stop, also what the stopped server held: WSL hands freed memory back to Windows only gradually (minutes),
        but a new server in the VM reuses it at once. Sized from Windows' figure alone, a restart got a third of
        the RAM cache it had, then restarted again to grow it."""
        avail = m["ram_avail"] + self.own_ram_estimate()
        f = self.freed
        if f and self.be.state not in ("running", "starting") and time.time() - f[0] < FREED_S:
            avail = max(avail, f[1] + f[2])
        return avail

    async def ram_back(self, m, cancelled=None):
        """Right after a stop, wait (up to host.ram_return_s) until Windows has most of the stopped server's RAM back.
        WSL returns it over a minute or so, and a server started meanwhile loads on top of what Windows hasn't got
        back yet: a switch from GLM-5.3-Flash to Qwen took the PC down to 4 GB free 15 s after the stop. Returns the
        last sample."""
        f = self.freed
        if not f or not self.host.ram_return_s or not f[2]:
            return m
        want = f[1] + f[2] - max(RAM_BACK_OUT_GB * GB, RAM_BACK_OUT * f[2])
        if m["ram_avail"] < want and time.time() - f[0] < self.host.ram_return_s:
            self.event("start", f"waiting for WSL to hand the last model's RAM back to Windows "
                                f"({(want - m['ram_avail']) / GB:.0f} GB to go, up to "
                                f"{self.host.ram_return_s - (time.time() - f[0]):.0f} s)")
        while m["ram_avail"] < want and time.time() - f[0] < self.host.ram_return_s and not (cancelled and cancelled()):
            await asyncio.sleep(2)
            m = await self.fresh_sample(time.time())
        return m

    def soft_ram_cap(self):
        c = self.ram_soft_cap
        return c[1] if c and time.time() - c[0] < SOFT_CAP_S else None

    def cap_after_ram_pressure(self, m, reasons):
        """Full (or lean that kept full's RAM cache) ran short of RAM: the next full start gets a smaller RAM cache for
        a while, or full -> lean -> full repeats every few minutes (lean frees the RAM, full takes it again)."""
        if not self.be.ram_gib or not any(r.startswith("RAM") for r in reasons) or (
                self.be.tier != "full" and self.be.ram_gib <= self.cfg["lean_ram_gb"]):
            return
        short = math.ceil(max(0.0, (self.cfg["ram_lean_gb"] + 2) * GB - m["ram_avail"]) / GIB)
        cap = int(max(self.cfg["lean_ram_gb"], self.be.ram_gib - short - 2))
        self.ram_soft_cap = (time.time(), cap)
        self.event("learned", f"RAM ran short with a {self.be.ram_gib} GiB RAM cache: full starts get at most {cap} "
                              f"GiB for the next {SOFT_CAP_S // 60} min")

    def full_ram_gib(self, m, plan):
        """RAM tier for a full start: enough that VRAM + RAM hold 90% of the experts (sizing.RAM_COVERAGE), shrunk
        to what the host can spare (never below lean's). VRAM is what makes full fast; a smaller RAM tier only sends
        rarer experts to NVMe."""
        cfg = self.cfg
        caps = [c for c in (cfg["max_ram_cache_gib"], self.learned.get(self.learned_key(), {}).get("ram_cap"),
                            self.soft_ram_cap()) if c]
        # free RAM after the start stays clearly above the scale-back threshold, or full -> lean -> full would loop
        return sizing.ram_cache_gib(plan["ram_target"], self.ram_for_start(m), cfg, self.ram_overhead(),
                                    min(caps) if caps else None)

    def keeps_ram(self, tier, m):
        """Whether an in-place switch to tier suits the RAM cache, which it leaves as it is: yes for lean (other apps
        short of VRAM); for full, unless a full start would get a clearly bigger one (lean began with a restart that
        shrank it)."""
        if tier != "full":
            return True
        if not m or not self.be.ram_gib:
            return False
        return self.full_ram_gib(m, self.plan_for("full", m)) - self.be.ram_gib < 8

    def ram_resizable(self):
        """The running engine can resize its RAM cache in place (the switches that change it restart otherwise)."""
        return self.cfg["live_resize"] and self.cfg["live_ram_resize"] and not self.live_ram_unsupported

    def ram_need_lean(self):
        return self.cfg["lean_ram_gb"] * GIB + self.ram_overhead()

    def ram_overhead(self, quant=None):
        learned = self.learned.get("_ram_overhead", {}).get(f"{self.host.kind}|{quant or self.cfg['quant']}")
        return learned if learned else RAM_OVERHEAD.get(self.host.kind, 8 * GB)

    def measure_ram_overhead(self, m):
        """Once the model has settled after a start: what it really took from the host beyond its RAM tier."""
        self.be.ram_measured = True
        if self.ram_before_start is None or not self.be.ram_gib:
            return
        over = (self.ram_before_start - m["ram_avail"]) - self.be.ram_gib * GIB
        if not 1 * GB < over < 30 * GB:
            return              # other apps changed a lot meanwhile: not a usable measurement
        key = f"{self.host.kind}|{self.cfg['quant']}"
        store = self.learned.setdefault("_ram_overhead", {})
        prev = store.get(key)
        # quick to rise, slow to fall: underestimating it is what sends full back to lean
        store[key] = over if prev is None or over > prev else 0.8 * prev + 0.2 * over
        config.save_json("learned.json", self.learned)
        self.event("learned", f"the model uses {over / GB:.1f} GB of RAM beyond its {self.be.ram_gib} GiB cache; "
                              f"sizing the RAM cache with {store[key] / GB:.1f} GB from now on")

    def best_tier(self, m, level=None):
        """Biggest profile that fits right now (respecting current pressure)."""
        level = level or self.pressure["level"]
        full_p, lean_p = self.plan_for("full", m), self.plan_for("lean", m)
        ram_ok = self.ram_spare(m) >= self.ram_need_lean()
        vram_for_us = m["vram_total"] - self.ext_vram(m) - 0.8 * GB
        full_ok = full_p["ok"] and ram_ok
        lean_ok = lean_p["ok"] and ram_ok and vram_for_us >= lean_p["budget"] + 0.5 * GB
        distinct = lean_p["budget"] < full_p["budget"] - 1.5 * GB
        if full_ok and level == "none":
            return "full"
        if lean_ok and distinct:
            return "lean"
        return "full" if full_ok else ("lean" if lean_ok else None)

    def evaluate(self, m):
        cfg = self.cfg
        sev, mod = [], []
        ext = self.ext_vram(m)
        free = m["vram_total"] - m["vram_used"]
        hits = sorted({w.lower() for w in cfg["watchlist"]} & m["running"])
        if hits:
            sev.append("watched app running: " + ", ".join(hits))
        # right after a start, page cache from loading is still being handed back: only a real emergency counts
        settling = self.be.state == "starting" or (self.be.state == "running" and self.be.ready_at
                                                   and time.time() - self.be.ready_at < RAM_GRACE_S)
        if m["ram_avail"] < (1.0 if settling else cfg["ram_pause_gb"]) * GB:
            sev.append(f"RAM very low ({m['ram_avail'] / GB:.1f} GB free)")
        elif m["ram_avail"] < cfg["ram_lean_gb"] * GB and not settling:
            mod.append(f"RAM low ({m['ram_avail'] / GB:.1f} GB free)")
        if self.be.state in ("running", "starting") and self.ext_at_start is not None:
            grew = ext - self.ext_at_start
            if free < cfg["vram_min_free_gb"] * GB and grew > 0.75 * GB:
                (sev if self.be.tier == "lean" else mod).append(
                    f"other apps need VRAM (+{grew / GB:.1f} GB since start, {free / GB:.1f} GB free)")
        lean_need = min(self.lean_budget(m), self.full_budget(m) + ext)
        if ext > m["vram_total"] - (lean_need + 1.0 * GB):
            sev.append(f"other apps using {ext / GB:.1f} GB VRAM")
        elif ext > self.ext_lean_threshold(m):
            mod.append(f"other apps using {ext / GB:.1f} GB VRAM")
        if m["ext_util"] >= cfg["ext_util_pct"]:
            who = m["ext_util_top"][0][0] if m["ext_util_top"] else "another app"
            (sev if cfg["gpu_busy_action"] == "pause" else mod).append(f"GPU busy: {who} at {m['ext_util']:.0f}%")
        level = "severe" if sev else "moderate" if mod else "none"
        return level, sev + mod, ext

    # ================================================================ lifecycle
    async def fresh_sample(self, after):
        for _ in range(30):
            m = self.mon.get()
            if m and m["t"] > after:
                return m
            await asyncio.sleep(0.25)
        return self.mon.get()

    # ================================================================ the catalog's models
    def model(self):
        """The configured model's catalog entry (settings only take quants the catalog has)."""
        cat = catalog.load()
        return cat.find(self.cfg["quant"]) or cat.models_for(sizing.ENGINE)[0]

    def model_choices(self):
        """The models the dashboard offers (catalog.offered, or every one with offer_uncurated), and the chosen one
        whatever its state, with their curation state."""
        show_all = self.cfg["offer_uncurated"]
        return [{"quant": m["quant"], "label": m["label"], "model": m["id"], "curation": catalog.status(m),
                 "offered": catalog.offered(m)}
                for m in catalog.load().models_for(sizing.ENGINE)
                if catalog.offered(m, show_all) or m["quant"] == self.cfg["quant"]]

    def profile_env(self, tier, m, plan):
        cfg = self.cfg
        paths = self.model()["paths"]
        env = {"QUANT": cfg["quant"], "CONTEXT": str(cfg["context"]), "PORT": str(cfg["backend_port"]),
               # where the model and its expert store are, from the catalog (run.sh has Qwen's as defaults)
               "MODEL_DIR": "./" + paths["model_dir"], "STORE": "./" + paths["store"],
               # loopback only: WSL forwards Windows' localhost to it, and nothing else on the VM's network reaches it
               "HOST": "127.0.0.1",
               # the manager handles contention; never let the in-process guard hard-kill a request
               "QWEN_VRAM_GUARD_BYTES": str(int(m["vram_total"] + 2 * GB))}
        capped = tier == "lean" or bool(self.learned.get(self.learned_key(), {}).get("vram_cap"))
        if tier == "full" and not capped:
            env["QWEN_VRAM_LIMIT_BYTES"] = str(int(m["vram_total"] - self.headroom(m)))
            env["RAM_GB"] = str(self.full_ram_gib(m, plan))
        else:
            env["QWEN_VRAM_LIMIT_BYTES"] = str(int(m["vram_used"] + plan["budget"]))
            env["RAM_GB"] = str(self.full_ram_gib(m, plan) if tier == "full" else cfg["lean_ram_gb"])
        env.update(sizing.engine_env(plan, cap_slots=capped))
        return env

    async def start(self, tier, reason):
        async with self.tlock:
            return await self._start_locked(tier, reason)

    async def _start_locked(self, tier, reason):
        cfg, host = self.cfg, self.host
        if self.observe:
            return False
        if (self.be.state in ("running", "starting") and self.be.tier == tier and self.be.quant == cfg["quant"]
                and self.be.context == cfg["context"]):
            return True
        if self.be.state in ("running", "starting", "error"):
            await self._stop_locked(f"switching to {tier}")
        m = await self.ram_back(await self.fresh_sample(time.time()))
        plan = self.plan_for(tier, m)
        if not plan["ok"]:
            self.be.error = f"cannot start {tier}: {plan.get('reason')}"
            self.be.set("error")
            self.event("error", self.be.error)
            return False
        self.ram_before_start = m["ram_avail"]
        self.be.ram_measured = False
        self.ram_growing = None
        self.ext_at_start, self.perproc_at_start = self.ext_vram(m), m["perproc_ded"]
        if m.get("ext_vram_abs") is None:
            self.ext_at_start = m["vram_used"]
        env = self.profile_env(tier, m, plan)
        self.be.tier, self.be.quant, self.be.context, self.be.error = tier, cfg["quant"], cfg["context"], None
        self.be.ram_gib, self.be.budget, self.be.plan, self.be.adopted = int(env["RAM_GB"]), plan["budget"], plan, False
        self.be.set("starting")
        self.paused_reason = None
        mode = "" if plan["staging"] else ", staging-free (experimental)"
        self.event("start", f"starting {tier} ({cfg['quant']} bpw, {cfg['context'] // 1024}K ctx, "
                            f"{plan['budget'] / GB:.1f} GB VRAM budget, {self.be.ram_gib} GiB RAM cache, "
                            f"chunk {plan['chunk']}{mode}): {reason}")
        url = f"http://127.0.0.1:{cfg['backend_port']}/v1/models"
        st, _ = await asyncio.to_thread(host.backend_status)
        if st in ("running", "stray"):
            await asyncio.to_thread(host.backend_stop)
            self.event("cleanup", "stopped a model server that was not started by this service")
        for _ in range(30):
            if not await asyncio.to_thread(http_ok, url, 1.0, self.backend_auth):
                break
            await asyncio.sleep(1)
        if host.follower:
            host.follower.clear()
        t0 = time.time()
        # the key joins the environment only here, so nothing that records env (backend.json) ever holds it
        ok, pid = await asyncio.to_thread(host.backend_start, {**env, "REPERTORY_BACKEND_KEY": self.backend_key},
                                          ["server"])
        if not ok:
            return await self._start_failed(tier, f"launcher refused: {pid}")
        self.be.pid = pid
        self.save_backend_meta(env)
        alive_t = 0.0
        while time.time() - t0 < cfg["start_timeout_s"]:
            if await asyncio.to_thread(http_ok, url, 2.0, self.backend_auth):
                self.be.startup_s = time.time() - t0
                self.be.ready_at = time.time()
                self.be.set("running")
                m2 = await self.fresh_sample(time.time())
                self.be.own_vram = max(0.0, m2["vram_used"] - (self.ext_at_start or 0.0))
                self.last_transition = time.time()
                self.recover["attempts"] = 0 if time.time() - self.recover["window_start"] > 600 else \
                    self.recover["attempts"]
                self.save_backend_meta(env)
                self.event("ready", f"{tier} ready in {self.be.startup_s:.0f}s, using "
                                    f"{self.be.own_vram / GB:.1f} GB VRAM")
                await asyncio.to_thread(host.after_ready)
                return True
            if time.time() - alive_t > 5:
                alive_t = time.time()
                st, cur = await asyncio.to_thread(host.backend_status)
                if st != "running":
                    break
            await asyncio.sleep(1)
        timeout = time.time() - t0 >= cfg["start_timeout_s"]
        return await self._start_failed(tier, "timeout" if timeout else "the server exited")

    async def _start_failed(self, tier, why):
        tail = self.host.follower.tail(12) if self.host.follower else ""
        kind = self.classify(tail)
        self.be.error = f"start failed ({why}{', ' + kind if kind != 'crash' else ''}): {tail[-500:]}"
        self.event("error", self.be.error.replace("\n", " | ")[:600])
        budget = self.be.budget
        await self._stop_locked("start failed", final_state="error")
        if kind != "crash":
            self.learn_from(kind, budget)
        return False

    def classify(self, tail):
        if OOM_VRAM.search(tail or ""):
            return "out of VRAM"
        if OOM_RAM.search(tail or ""):
            return "out of RAM"
        return "crash"

    def learn_from(self, kind, budget=None, ram_gib=None):
        if kind == "out of VRAM" and budget:
            cap = budget - 2 * GB
            self.learn(vram_cap=cap)
            self.event("learned", f"VRAM budget for {self.cfg['quant']} bpw at {self.cfg['context'] // 1024}K on "
                                  f"this GPU capped at {cap / GB:.1f} GB after running out of VRAM")
        elif kind == "out of RAM":
            cap = max(self.cfg["lean_ram_gb"], (ram_gib or self.be.ram_gib or 16) - 4)
            self.learn(ram_cap=cap)
            self.event("learned", f"RAM cache capped at {cap} GiB after running out of RAM")

    def save_backend_meta(self, env):
        config.save_json("backend.json", {
            "pid": self.be.pid, "tier": self.be.tier, "quant": self.be.quant, "context": self.be.context,
            "ram_gib": self.be.ram_gib, "budget": self.be.budget, "own_vram": self.be.own_vram,
            "started": self.be.since, "ready_at": self.be.ready_at, "startup_s": self.be.startup_s,
            "engine_dir": self.host.engine_dir, "port": self.cfg["backend_port"],
            "env": {k: v for k, v in env.items() if k != "REPERTORY_BACKEND_KEY"},
            "plan": {k: v for k, v in (self.be.plan or {}).items() if isinstance(v, (int, float, str, bool))}})

    async def stop(self, reason, paused_reason="manual"):
        async with self.tlock:
            await self._stop_locked(reason)
            self.paused_reason = paused_reason

    async def _stop_locked(self, reason, final_state="stopped"):
        if self.observe:
            return
        if self.be.state == "stopped" and not self.be.pid:
            return
        was = self.be.tier
        m = self.mon.get() if self.be.state == "running" else None
        own = self.own_ram_estimate() if m else 0.0
        self.be.set("stopping")
        # Check that it's gone: under severe memory pressure the stop command itself can fail, and a server still
        # holding the GPU and RAM must not be taken for stopped.
        out, st = "", "unknown"
        for attempt in range(3):
            try:
                out = await asyncio.to_thread(self.host.backend_stop)
                st, _ = await asyncio.to_thread(self.host.backend_status)
            except (OSError, subprocess.SubprocessError) as e:
                out, st = str(e), "unknown"
            if st not in ("running", "stray", "unknown"):
                break
            log.warning("model server %s after a stop (%s); retrying", st, str(out)[:200])
            await asyncio.sleep(5 * (attempt + 1))
        else:
            if st == "unknown":
                # the host doesn't answer (WSL down): whether the server is still there can't be told, so it isn't
                # shown as running; the next start stops whatever is left first
                self.be.error = f"can't reach the model server's host: {str(out)[:200] or 'no answer'}"
                self.be.set("error")
                self.event("error", self.be.error.replace("\n", " | "))
                return
            self.be.set("running")                  # still there: keep tracking it, and the next tick tries again
            self.event("error", f"could not stop the model server ({str(out)[:160] or st}); will try again")
            return
        await asyncio.to_thread(self.host.after_stop)
        try:
            (config.STATE / "backend.json").unlink()
        except OSError:
            pass
        self.be.pid = self.be.tier = self.be.ready_at = self.be.own_vram = self.be.ram_gib = None
        self.be.budget = self.be.plan = None
        self.ext_at_start = self.perproc_at_start = None
        self.be.set(final_state)
        self.last_transition = self.last_stop_t = time.time()
        if m:
            self.freed = (self.last_stop_t, m["ram_avail"], own)
        if final_state == "stopped":
            self.event("stop", f"stopped{' ' + was if was else ''}: {reason}")

    async def restart(self, tier, reason):
        async with self.tlock:
            await self._stop_locked(reason)
            return await self._start_locked(tier, reason)

    async def change_tier(self, tier, reason, in_place=True, manual=False, ram_gib=None):
        """To another profile (or, with ram_gib, the same one with another RAM cache): in place when in_place, else,
        or if the engine can't, by a restart. ram_gib: the RAM cache in GiB the switch also sets (None: it stays as it
        is). A resize put off by a request still running after a minute is retried at the next tick; one chosen in the
        dashboard restarts then, as that button always did."""
        async with self.tlock:
            if in_place:
                done = await self._shift_locked(tier, reason, ram_gib)
                if done:
                    return True
                if done is None and not manual:
                    return False
                if done is None:
                    self.event("shift", f"a request was still running after a minute: restarting into {tier} instead")
            await self._stop_locked(reason)
            return await self._start_locked(tier, reason)

    async def to_full(self, reason, m, manual=False):
        """Lean -> full. Where the engine can resize its RAM cache, in place, growing it to what a full start would
        get; else in place only if the RAM cache already suits full (keeps_ram), or by a restart."""
        if m and self.ram_resizable():
            want = self.full_ram_gib(m, self.plan_for("full", m))
            return await self.change_tier("full", reason, manual=manual,
                                          ram_gib=want if want > (self.be.ram_gib or 0) else None)
        return await self.change_tier("full", reason, manual=manual, in_place=self.keeps_ram("full", m))

    def shift_request(self, tier, m, current=None, ram_gib=None):
        """The engine's live resize to tier (sizing.live_resize_args plus the VRAM limits a start would get from
        profile_env), and the plan; None if only a restart can switch (a staging-free plan on either side). current:
        the running server's plan (the backend's, unless a benchmark's). ram_gib: the RAM cache to set as well."""
        plan = self.plan_for(tier, m)
        current = self.be.plan if current is None else current
        if not plan["ok"] or not plan["staging"] or not (current or {}).get("staging"):
            return None, plan
        capped = tier == "lean" or bool(self.learned.get(self.learned_key(), {}).get("vram_cap"))
        ext = self.ext_vram(m)       # the model is running, so other apps' VRAM is this, not vram_used
        limit = ext + plan["budget"] if capped else m["vram_total"] - self.headroom(m)
        body = dict(sizing.live_resize_args(plan, cap_slots=capped), vram_limit_bytes=int(limit),
                    torch_limit_bytes=int(limit - ext + 0.5 * GB))     # as run.sh: the budget plus 0.5 GB
        if ram_gib is not None:
            body["ram_bytes"] = int(ram_gib * GIB)
        return body, plan

    async def _shift_locked(self, tier, reason, ram_gib=None):
        """True: switched in place. None: put off (a request is running). False: a restart is needed."""
        be, cfg = self.be, self.cfg
        ram_gib = None if ram_gib == be.ram_gib else ram_gib
        if (self.observe or not cfg["live_resize"] or self.live_unsupported or be.state != "running"
                or (be.tier == tier and ram_gib is None) or be.quant != cfg["quant"] or be.context != cfg["context"]
                or (ram_gib is not None and not self.ram_resizable())):
            return False
        m = await self.fresh_sample(time.time())
        body, plan = self.shift_request(tier, m, ram_gib=ram_gib)
        if body is None:
            return False
        was, since = be.tier, be.since
        be.target = tier
        be.set("resizing")               # requests arriving meanwhile are held; the engine waits for running ones
        url = f"http://127.0.0.1:{cfg['backend_port']}/expert_cache/resize"
        try:
            st, r = await asyncio.to_thread(http_post_json, url, body, 120.0, self.backend_auth)
        except (OSError, ValueError, http.client.HTTPException) as e:
            st, r = None, {"error": str(e)}
        finally:
            be.target = None
            be.set("running")
            be.since = since             # the same server: "up" keeps counting from its start
        if st != 200:
            err = str((r or {}).get("error", ""))
            if st == 404:
                self.live_unsupported = True
                self.event("shift", "this engine checkout can't resize in place (update it in Setup); restarting")
            elif st == 400 and err.startswith("ram_bytes:"):
                self.live_ram_unsupported = True
                self.event("shift", f"the engine can't resize its RAM cache in place ({err[10:].strip()}); "
                                    f"restarting")
            elif st != 409:
                self.event("error", f"in-place switch to {tier} failed ({st}: {err}); restarting instead")
            return None if st == 409 else False
        self.ext_at_start, self.perproc_at_start = self.ext_vram(m), m["perproc_ded"]
        be.tier, be.plan, be.budget = tier, plan, plan["budget"]
        ram, ram_was = r.get("ram") or {}, be.ram_gib
        if ram.get("growing"):
            # it grows in the background (watch_ram_growth): count the size it grows to, the higher estimate
            self.ram_growing = {"since": time.time(), "from_gib": ram_was}
            be.ram_gib = max(1, round(ram["target_bytes"] / GIB))
        elif ram.get("ram_bytes"):
            self.ram_growing = None
            be.ram_gib = max(1, round(ram["ram_bytes"] / GIB))
        m2 = await self.fresh_sample(time.time()) or m
        be.own_vram = max(0.0, m2["vram_used"] - self.ext_vram(m2))
        self.last_transition = time.time()
        meta = config.load_json("backend.json", {}) or {}
        env = dict(meta.get("env") or {}, QWEN_VRAM_LIMIT_BYTES=str(body["vram_limit_bytes"]),
                   QWEN_TORCH_LIMIT_BYTES=str(body["torch_limit_bytes"]), QWEN_CHUNK=str(body["chunk"]))
        env.pop("GLM53_EC_MAX_SLOTS", None)
        if body["max_slots"] is not None:
            env["GLM53_EC_MAX_SLOTS"] = str(body["max_slots"])
        if be.ram_gib:
            env["RAM_GB"] = str(be.ram_gib)
        self.save_backend_meta(env)
        p = (r.get("pools") or [{}])[0]
        what = f"{was} -> {tier}" if was != tier else tier
        ram_txt = ("" if not ram else f"; RAM cache {ram_was} -> {be.ram_gib} GiB, growing in the background"
                   if ram.get("growing") else f"; RAM cache {ram_was} -> {be.ram_gib} GiB "
                   f"({ram.get('dropped', 0)} experts dropped, {ram.get('moved', 0)} moved)")
        self.event("shift", f"{what} without a restart in {r.get('seconds', 0):.1f}s: VRAM expert cache "
                            f"{p.get('slots_before')} -> {p.get('slots')} experts "
                            f"({p.get('arena_bytes', 0) / GB:.1f} GB), prefill chunk {r.get('chunk')}, "
                            f"{be.own_vram / GB:.1f} GB VRAM in use{ram_txt}: {reason}")
        return True

    async def watch_ram_growth(self, m):
        """While the engine grows its RAM cache in the background: stop it as soon as free RAM nears lean's threshold
        (other apps want it; no waiting for an idle gap, as scaling back does), and take its size when it is done."""
        cfg, be = self.cfg, self.be
        url = f"http://127.0.0.1:{cfg['backend_port']}/expert_cache/ram"
        short = m["ram_avail"] < (cfg["ram_lean_gb"] + RAM_GROW_MARGIN_GB) * GB
        try:
            if short:
                st, info = await asyncio.to_thread(http_post_json, url + "/stop", {}, 30.0, self.backend_auth)
                info = info if st == 200 else None
            else:
                info = await asyncio.to_thread(http_get_json, url, 5.0, self.backend_auth)
        except (OSError, ValueError, http.client.HTTPException) as e:
            log.info("RAM growth check failed: %s", e)
            info = None
        finally:
            self.ram_watch = False
        if not info or info.get("growing") or not self.ram_growing or be.state != "running":
            return
        was = self.ram_growing.get("from_gib")
        self.ram_growing = None
        be.ram_gib = max(1, round(info["ram_bytes"] / GIB))
        meta = config.load_json("backend.json", {}) or {}
        self.save_backend_meta(dict(meta.get("env") or {}, RAM_GB=str(be.ram_gib)))
        if short:
            self.event("shift", f"stopped growing the RAM cache at {be.ram_gib} GiB: only "
                                f"{m['ram_avail'] / GB:.1f} GB of RAM free")
        else:
            stopped = f" (stopped: {info['stopped']})" if info.get("stopped") else ""
            self.event("shift", f"RAM cache {was} -> {be.ram_gib} GiB in the background in "
                                f"{info.get('grow_s', 0):.0f} s{stopped}")

    def spawn(self, coro):
        if self.task and not self.task.done():
            coro.close()
            return False
        self.task = asyncio.create_task(coro)
        return True

    async def adopt_or_clean(self):
        """At service start: keep a model server we started earlier (manager restart/upgrade), stop any other."""
        host, cfg = self.host, self.cfg
        try:
            st, pid = await asyncio.to_thread(host.backend_status)
        except Exception as e:  # noqa: BLE001 - WSL may not be installed yet (setup handles that)
            log.warning("backend status failed: %s", e)
            return
        meta = config.load_json("backend.json", None)
        url = f"http://127.0.0.1:{cfg['backend_port']}/v1/models"
        if self.observe:
            if await asyncio.to_thread(http_ok, url, 2.0, self.backend_auth):
                self.be.set("running")
                self.be.tier = (meta or {}).get("tier") or "unknown"
                m = await self.fresh_sample(0)
                if m and m.get("ext_vram_abs") is None:
                    # not our start: take other apps' current per-process usage as the baseline
                    self.ext_at_start, self.perproc_at_start = min(m["vram_used"], m["perproc_ded"]), m["perproc_ded"]
                self.event("observe", "observe mode: a model server is answering on the backend port")
            return
        answering = st == "running" and await asyncio.to_thread(http_ok, url, 3.0, self.backend_auth)
        if answering and meta and meta.get("pid") == pid:
            m = await self.fresh_sample(0)
            b = self.be
            b.pid, b.tier, b.quant, b.context = pid, meta["tier"], meta["quant"], meta["context"]
            b.ram_gib, b.budget, b.own_vram = meta.get("ram_gib"), meta.get("budget"), meta.get("own_vram")
            b.plan, b.startup_s, b.adopted = meta.get("plan"), meta.get("startup_s"), True
            b.set("running")
            b.ready_at = meta.get("ready_at") or time.time()
            self.ext_at_start = m["vram_used"] - (b.own_vram or 0.0) if m else None
            self.perproc_at_start = m["perproc_ded"] if m else None
            self.last_transition = time.time()
            self.event("adopt", f"kept the running model ({b.tier}, pid {pid}) across the manager restart")
        elif st in ("running", "stray"):
            await asyncio.to_thread(host.backend_stop)
            self.event("cleanup", "stopped a model server left over from a previous session")
            self.last_stop_t = time.time()

    async def check_alive(self):
        st, pid = await asyncio.to_thread(self.host.backend_status)
        if st != "unknown" or self.be.state != "running":
            self.unreachable_since = None              # an answer, or nothing to lose: the next miss counts afresh
        if self.be.state != "running" or (self.task and not self.task.done()):
            return
        if st == "running" and pid == self.be.pid:
            return
        tier, budget, ram = self.be.tier, self.be.budget, self.be.ram_gib
        if st == "unknown":
            # the host doesn't answer (for unknown, pid holds why): a missed check is fine, a minute of them isn't
            now = time.time()
            self.unreachable_since = self.unreachable_since or now
            if now - self.unreachable_since < UNREACHABLE_S:
                return
            self.unreachable_since = None
            self.be.error = (f"can't reach the model server's host for {UNREACHABLE_S} s: "
                             f"{str(pid or 'no answer')[:300]}")
            self.event("error", self.be.error.replace("\n", " | ")[:600])
            self.spawn(self._recover(tier, "unreachable", budget, ram))
            return
        tail = self.host.follower.tail(30) if self.host.follower else ""
        kind = self.classify(tail)
        self.be.error = f"model server exited unexpectedly ({kind}): {tail[-400:]}"
        self.event("error", self.be.error.replace("\n", " | ")[:600])
        self.spawn(self._recover(tier, kind, budget, ram))

    async def _recover(self, tier, kind, budget, ram):
        async with self.tlock:
            await self._stop_locked("model server exited", final_state="error")
        if kind != "crash":
            self.learn_from(kind, budget, ram)
        r = self.recover
        if time.time() - r["window_start"] > 600:
            r["window_start"], r["attempts"] = time.time(), 0
        if not self.cfg["auto_recover"] or r["attempts"] >= 3:
            self.event("error", "not restarting automatically (3 failures in 10 minutes); use Start in the dashboard")
            return
        delay = (10, 30, 90)[r["attempts"]]
        r["attempts"] += 1
        self.event("recover", f"restarting in {delay}s (attempt {r['attempts']} of 3)")
        await asyncio.sleep(delay)
        if self.be.state == "error":
            m = self.mon.get()
            await self.start(self.best_tier(m) or tier, "automatic recovery")

    # ================================================================ request path
    async def ensure_ready(self):
        cfg = self.cfg
        busy_deadline = None
        tried = False
        while True:
            st = self.be.state
            if self.bench.running:
                return False, "a benchmark is running (Repertory dashboard, Benchmark tab); try again when it finishes"
            if st == "running":
                return True, ""
            if st == "error" and tried:
                # the start this request triggered failed: report it instead of retrying forever
                return False, self.be.error or "model failed to start"
            if self.observe:
                return False, "this manager is in observe mode and does not start the model"
            if self.setup.blocking:
                return False, f"setup step '{self.setup.running}' is running; try again when it finishes"
            if st in ("starting", "stopping", "resizing"):
                await asyncio.sleep(0.5)
                continue
            if not cfg["wake_on_request"]:
                return False, "model is paused and wake-on-request is off (Repertory dashboard)"
            m = self.mon.get()
            tier = self.best_tier(m) if m else None
            # Right after a stop, WSL is still handing RAM back to Windows, so free RAM reads low and full would be
            # rejected for lean only to be re-saturated a minute later. Let it settle first.
            settling = time.time() - self.last_stop_t < self.host.ram_settle_s
            if tier == "lean" and settling and self.pressure["level"] == "none" and \
                    self.plan_for("full", m)["ok"]:
                await asyncio.sleep(2)
                continue
            if tier:
                if self.pressure["level"] != "none":
                    self.override_until = time.time() + cfg["wake_override_min"] * 60
                if not self.spawn(self.start(tier, "request arrived")):
                    await asyncio.sleep(0.5)
                    continue
                tried = True
                await asyncio.sleep(0.5)
                continue
            busy_deadline = busy_deadline or time.time() + cfg["busy_wait_s"]
            if time.time() > busy_deadline:
                ext = self.ext_vram(m) / GB if m else 0
                return False, (f"GPU/RAM busy (other apps using {ext:.1f} GB VRAM, "
                               f"{(m['ram_avail'] / GB) if m else 0:.1f} GB RAM free); model not started")
            await asyncio.sleep(2)

    async def route(self, body):
        """Before a request goes to the model server: the model it names (resolve_model), switched to if it isn't the
        configured one (switch_on_request). One switch at a time; requests arriving meanwhile wait for it, then
        choose. The switch waits (up to SWITCH_DRAIN_S) for requests already sent to the model to finish. Returns
        (ok, error, HTTP status, switched)."""
        try:
            name = json.loads(body).get("model") if body else None
        except (ValueError, AttributeError, UnicodeDecodeError):
            name = None                                  # the model server answers what it makes of it
        self.routing += 1
        try:
            while self.switching:
                await asyncio.sleep(0.5)
            want, err = self.resolve_model(name)
            if err:
                return False, err, 404, False
            if want is None or want["quant"] == self.cfg["quant"] or not self.cfg["switch_on_request"]:
                return True, "", 200, False
            if self.bench.running:
                return (False, "a benchmark is running (Repertory dashboard, Benchmark tab); try again when it "
                               "finishes", 503, False)
            if self.observe or self.setup.blocking:
                return False, ("this manager is in observe mode and does not switch models" if self.observe else
                               f"setup step '{self.setup.running}' is running; try again when it finishes"), 503, False
            self.switching = want["quant"]
            try:
                before = self.model()
                deadline = time.time() + SWITCH_DRAIN_S
                while self.in_flight - self.routing > 0:       # requests the model server is answering
                    if time.time() > deadline:
                        return False, (f"{before['label']} is still answering other requests; ask for "
                                       f"{want['id']} again later"), 503, False
                    await asyncio.sleep(0.5)
                self.event("switch", f"a request asked for {want['label']}: switching from {before['label']}")
                self.cfg["quant"] = want["quant"]
                config.save_config(self.cfg)
                if self.be.state in ("running", "starting", "resizing"):
                    # start stops a server of another model first, and is a no-op if this one is already up
                    await self.start(self.be.tier or "full", f"a request asked for {want['label']}")
                return True, "", 200, True                    # stopped or paused: ensure_ready starts it, as ever
            finally:
                self.switching = None
        finally:
            self.routing -= 1

    async def forward(self, method, target, headers, body, writer):
        cfg = self.cfg
        self.in_flight += 1
        self.last_request = t0 = time.time()
        if self.pressure["level"] != "none" and self.be.state == "running":
            self.override_until = time.time() + cfg["wake_override_min"] * 60
        held = was_held = self.be.state != "running"
        if held:
            self.held += 1
        bw = None
        stats = StreamStats()
        try:
            ok, msg, code, switched = await self.route(body)
            was_held = was_held or switched
            if ok:
                ok, msg = await self.ensure_ready()
            if held:
                self.held -= 1
                held = False
            if not ok:
                self.failed += 1
                await respond(writer, code if code != 200 else 503, model_not_found(msg) if code == 404 else
                              oai_error(msg))
                return
            try:
                br, bw = await asyncio.open_connection("127.0.0.1", cfg["backend_port"])
            except OSError as e:
                self.failed += 1
                await respond(writer, 502, oai_error(f"model server unreachable: {e}", "bad_gateway", 502))
                return
            t_send = time.time()
            bw.write(backend_request(method, target, headers, body, cfg["backend_port"], self.backend_key))
            await bw.drain()
            await pipe(br, writer, lambda c: stats.feed(c, time.time()))
            self.served += 1
            self.record_metrics(stats, t0, t_send, time.time(), was_held)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            if held:
                self.held -= 1
            self.in_flight -= 1
            self.last_done = time.time()
            close(bw, writer)

    def record_metrics(self, stats, t0, t_send, t_end, was_held):
        u = stats.usage() or {}
        if stats.status != 200 or not isinstance(u, dict) or not u:
            return
        # chat: prompt/completion_tokens; Responses: input/output_tokens
        pt = token_count(u.get("prompt_tokens"), u.get("input_tokens"))
        ct = token_count(u.get("completion_tokens"), u.get("output_tokens"))
        # tokens the prefix cache served (chat: prompt_tokens_details; Responses: input_tokens_details) weren't
        # processed: prompt speed counts only the ones that were
        details = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
        cached = min(pt, token_count(details.get("cached_tokens") if isinstance(details, dict) else None, None))
        new = pt - cached
        ttft = (stats.t_first - t_send) if stats.t_first else None
        decode_s = (t_end - stats.t_first) if stats.t_first else None
        self.metrics.append({
            "t": t_end, "prompt": pt, "cached": cached, "completion": ct, "wait_s": round(t_send - t0, 2),
            "ttft_s": round(ttft, 3) if ttft is not None else None, "total_s": round(t_end - t_send, 2),
            # an answer that arrives in one piece at the end (a tool call) has no decode time to measure
            "decode_tps": round(ct / decode_s, 1) if decode_s and decode_s >= 0.25 and ct > 8 else None,
            "prefill_tps": round(new / ttft) if ttft and new > 256 else None,
            "tier": self.be.tier, "budget_gb": round((self.be.budget or 0) / GB, 1), "cold": was_held})

    def metrics_summary(self):
        recent = list(self.metrics)[-30:]

        def med(key):
            vals = [r[key] for r in recent if r.get(key) is not None]
            return round(statistics.median(vals), 1) if vals else None
        return {"count": len(self.metrics), "last": recent[-1] if recent else None,
                "decode_tps": med("decode_tps"), "prefill_tps": med("prefill_tps"), "ttft_s": med("ttft_s"),
                "tokens_out": sum(r["completion"] for r in self.metrics),
                "tokens_in": sum(r["prompt"] for r in self.metrics)}

    # ================================================================ the network port
    def client_models(self):
        """The catalog models clients can name (/v1/models): the configured one first (what answers now), then the
        others installed here (the setup check) that the dashboard offers."""
        eng = (self.setup.hw or {}).get("engine") or {}
        current = self.model()
        return [current] + [m for m in catalog.load().models_for(sizing.ENGINE)
                            if m["id"] != current["id"] and eng.get(catalog.setup_id(m)) == "1"
                            and catalog.offered(m, self.cfg["offer_uncurated"])]

    def model_object(self, m):
        """A model as OpenAI's API lists it; created: the day its catalog entry was curated (0 before)."""
        day = (m.get("curation") or {}).get("date")
        try:
            created = calendar.timegm(time.strptime(day, "%Y-%m-%d")) if day else 0
        except ValueError:
            created = 0
        return {"id": catalog.served_name(m), "object": "model", "created": created, "owned_by": "repertory",
                "max_model_len": min(self.cfg["context"], (m.get("facts") or {}).get("max_context") or 262144)}

    def models_list(self):
        return {"object": "list", "data": [self.model_object(m) for m in self.client_models()]}

    def resolve_model(self, name):
        """(model, error) for the `model` a request names. A model clients can name, by its id or quant; a family
        (qwen3.8-flash-next): the configured model if it is of that family, else the family's largest quant there
        is. A catalog model this PC can't serve is an error. Any other name (an old one, "repertory"): (None, None),
        and the configured model answers."""
        if not isinstance(name, str) or not name.strip():
            return None, None
        name, cat, ours = name.strip(), catalog.load(), self.client_models()
        m = cat.find(name)
        if m is not None:
            if any(o["id"] == m["id"] for o in ours):
                return m, None
            eng = (self.setup.hw or {}).get("engine") or {}
            if eng.get(catalog.setup_id(m)) == "1":
                return None, f"the model {name!r} hasn't passed curation on this PC, so it isn't offered"
            return None, f"the model {name!r} isn't installed on this PC (Repertory dashboard, Setup tab)"
        family = [o for o in ours if o["family"] == name]
        if family:
            return (ours[0] if ours[0]["family"] == name else max(family, key=lambda o: o["bpw"])), None
        if any(c["family"] == name for c in cat.models):
            return None, f"no {name} model is installed on this PC (Repertory dashboard, Setup tab)"
        return None, None

    async def bind_front(self):
        """Listen on the network port (127.0.0.1:network_port), which `tailscale serve` or similar exposes."""
        if self.front_server or not self.cfg["network_port"] or self.observe:
            return
        try:
            self.front_server = await asyncio.start_server(self.handle, ["127.0.0.1", "::1"],
                                                           self.cfg["network_port"], limit=8 * 1024 * 1024)
            self.event("service", f"network port listening on 127.0.0.1:{self.cfg['network_port']}")
        except OSError as e:
            if not self._front_warned:
                self.event("warning", f"network port {self.cfg['network_port']} is busy ({e.strerror}); "
                                      f"retrying every 30 s.")
                self._front_warned = True

    # ================================================================ HTTP entry
    def token_ok(self, headers, target=None):
        """The request carries the dashboard token: its cookie, a Bearer header, or, only when opening the page from
        the private link, ?token= in `target`."""
        tok = self.cfg["dashboard_token"]
        if not tok:
            return False
        given = [header(headers, "authorization").removeprefix("Bearer ").strip()]
        if target is not None:
            given.append(query_token(target))
        for part in header(headers, "cookie").split(";"):
            name, _, value = part.strip().partition("=")
            if name in (COOKIE, LEGACY_COOKIE):
                given.append(value)
        return any(g and hmac.compare_digest(g.encode(), tok.encode()) for g in given)

    def refusal(self, method, headers, front):
        """(status, reason) for a request refused before its body is read, else None.
        - Main port: only requests addressed to 127.0.0.1, localhost or ::1. A web page that points its own domain at
          127.0.0.1 (DNS rebinding) would otherwise count as this PC.
        - Network port: a browser page must come from one of this PC's names, for the same reason.
        - Any port: a browser request from another site (cross-site request forgery)."""
        if not front and self.cfg["listen_host"] in LOCAL_NAMES and host_name(headers) not in LOCAL_NAMES:
            return 403, "this port only answers requests addressed to 127.0.0.1, localhost or ::1"
        if front and header(headers, "origin") and not self.network_origin_ok(headers):
            return 403, "requests from other websites are refused"
        if method not in ("GET", "HEAD") and not same_origin(headers):
            return 403, "requests from other websites are refused"
        return None

    def network_origin_ok(self, headers):
        try:
            name = urllib.parse.urlsplit(header(headers, "origin")).hostname or ""
        except ValueError:
            return False
        names = {*LOCAL_NAMES, socket.gethostname().lower(), self.cfg["remote_host"].lower()} - {""}
        return name in names or is_ip(name)

    async def handle(self, reader, writer):
        try:
            method, target, headers = await read_head(reader)
        except HTTPError as e:
            await respond(writer, e.status, {"error": str(e)})
            close(writer)
            return
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
            close(writer)
            return
        path = target.split("?", 1)[0]
        p = path.rstrip("/") or "/"
        # Every port has the same layout: /v1 is the API, and the root is the dashboard, which needs the dashboard
        # token (the private link's cookie). Without it, the root is a sign-in page that takes the private link.
        # The network port is the one a tailnet reaches (via `tailscale serve`); the main port is this PC only.
        front = writer.get_extra_info("sockname")[1] == self.cfg["network_port"]
        try:
            refused = self.refusal(method, headers, front)
            if refused:
                await respond(writer, refused[0], {"error": refused[1]})
            elif path.startswith("/_switcher") or path == "/":
                if self.token_ok(headers, target if path in PAGES else None):
                    body = await read_body(reader, writer, headers, 1024 * 1024) if method == "POST" else b""
                    await self.api(method, path, target, body, writer, local=not front)
                elif path in PAGES and method in ("GET", "HEAD"):
                    await respond(writer, 200, SIGNIN_FILE.read_bytes(), "text/html; charset=utf-8")
                else:
                    await respond(writer, 401, {"error": "the dashboard needs its private link"})
            elif p == "/v1/models" and method in ("GET", "HEAD"):
                await respond(writer, 200, self.models_list())
            elif p.startswith("/v1/models/") and method in ("GET", "HEAD"):
                name = urllib.parse.unquote(p[len("/v1/models/"):])
                m, err = self.resolve_model(name)
                if m is not None:
                    await respond(writer, 200, self.model_object(m))
                else:
                    await respond(writer, 404, model_not_found(err or f"The model {name!r} does not exist"))
            elif p in COMPLETIONS:
                if method != "POST":
                    # a GET (an image tag on any web page) must not wake the model
                    await respond(writer, 405, {"error": {"message": f"{path} takes POST", "type": "bad_method"}},
                                  extra=("Allow: POST",))
                else:
                    body = await read_body(reader, writer, headers, 64 * 1024 * 1024)
                    await self.forward(method, target, headers, body, writer)   # one model answers any model name
                    return
            elif front or path.startswith("/v1/"):
                await respond(writer, 404, {"error": {
                    "message": f"{path} is not available: this server is Repertory (OpenAI-compatible: "
                               f"/v1/chat/completions, /v1/responses, /v1/completions, /v1/models). Embeddings, "
                               f"images and the Anthropic /v1/messages API are not supported.",
                    "type": "not_found", "code": 404}})
            else:
                # favicon.ico, scanners...: never wake the model for these
                await respond(writer, 404, {"error": {"message": f"unknown path {path}", "type": "not_found"}})
            close(writer)
        except HTTPError as e:
            await respond(writer, e.status, {"error": str(e)})
            close(writer)
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
            close(writer)
        except Exception as e:  # noqa: BLE001
            log.exception("request failed: %s", e)
            close(writer)

    # ================================================================ control API
    def engine_sync(self):
        hw = self.setup.hw or {}
        e, repo = hw.get("engine") or {}, hw.get("repo") or {}
        if not e.get("engine_head") or not repo.get("head"):
            return None
        same = e["engine_head"] == repo["head"]
        return {"repo": repo["head"][:7], "engine": e["engine_head"][:7], "in_sync": same,
                "repo_dirty": repo.get("dirty"), "separate": self.host.kind == "windows-wsl"}

    def status(self):
        m = self.mon.get() or {}
        now = time.time()
        cfg = self.cfg
        ext = self.ext_vram(m) if m else 0.0
        listen = cfg["listen_host"]
        host = "127.0.0.1" if listen == "127.0.0.1" or unspecified(listen) else listen
        plans = {}
        if m:
            for t in ("full", "lean"):
                p = self.plan_for(t, m)
                plans[t] = {"ok": p["ok"], "budget_gb": round(p["budget"] / GB, 1), "chunk": p["chunk"],
                            "slots": p["slots"], "staging": p["staging"], "experimental": p["experimental"],
                            "reason": p.get("reason")}
            plans["best"] = self.best_tier(m)
        hw = self.setup.hw or {}
        return {
            "now": now, "state": self.be.state, "tier": self.be.tier, "target": self.be.target,
            "ram_growing": bool(self.ram_growing),
            "quant": self.be.quant or cfg["quant"],
            "context": cfg["context"], "endpoint": f"http://{host}:{cfg['listen_port']}/v1",
            "served_name": catalog.served_name(self.model()), "since": self.be.since, "startup_s": self.be.startup_s,
            "error": self.be.error, "paused_reason": self.paused_reason, "ram_cache_gib": self.be.ram_gib,
            "budget_gb": round(self.be.budget / GB, 1) if self.be.budget else None, "adopted": self.be.adopted,
            "manual_lean": self.manual_lean, "ram_overhead_gb": round(self.ram_overhead() / GB, 1),
            "in_flight": self.in_flight, "held": self.held, "served": self.served, "failed": self.failed,
            "last_request": self.last_request,
            "idle_s": (now - self.last_done) if self.last_done and not self.in_flight else None,
            "override_until": self.override_until if self.override_until > now else None,
            "transition": bool(self.task and not self.task.done()),
            "pressure": self.pressure, "plans": plans,
            "gpu": {"name": self.gpu_name, "total": m.get("vram_total"), "used": m.get("vram_used"),
                    "ext": ext, "own": max(0.0, m.get("vram_used", 0) - ext) if m else 0.0,
                    "util": m.get("gpu_util"), "ext_util": m.get("ext_util"), "ext_util_top": m.get("ext_util_top")},
            "ram": {"total": m.get("ram_total"), "avail": m.get("ram_avail"), "model_est": self.own_ram_estimate()},
            "history": list(self.hist)[-240:], "events": list(self.events)[-80:],
            "metrics": self.metrics_summary(), "learned": self.learned.get(self.learned_key()),
            "recover": {"attempts": self.recover["attempts"]},
            "autostart": self.host.autostart_enabled(), "platform": self.host.kind, "platform_label": self.host.label,
            "observe": self.observe, "engine_sync": self.engine_sync(),
            "recommendations": hw.get("recommendations"),
            "quants": {c["quant"]: c for c in self.model_choices()},
            # where other devices reach the API and, with the private link, the dashboard
            "remote": {"port": cfg["network_port"] if self.front_server else 0,
                       "host": cfg["remote_host"] or socket.gethostname().lower()},
            "setup_running": self.setup.running,
            "bench_running": self.bench.running, "bench_current": self.bench.current,
            "bench_waiting": self.bench.waiting,
        }

    async def api(self, method, path, target, body, writer, local):
        """The dashboard and its control API, for a request carrying the token. `local`: it came in on the main port,
        from this PC; the network port can't run setup, quit, change autostart, rotate the link or rename the PC."""
        if path in PAGES:  # the dashboard, at the root
            tok = query_token(target)
            if tok and self.token_ok((), target):
                # a private link: remember the token for a year, then drop it from the address bar
                await respond(writer, 302, b"", "text/plain", (
                    "Location: /", f"Set-Cookie: {COOKIE}={tok}; Path=/; Max-Age=31536000; HttpOnly; SameSite=Lax"))
                return
            await respond(writer, 200, UI_FILE.read_bytes(), "text/html; charset=utf-8")
            return
        try:
            data = json.loads(body or b"{}")
            if not isinstance(data, dict):
                raise ValueError
        except ValueError:
            await respond(writer, 400, {"error": "bad json"})
            return
        route = path[len("/_switcher/api/"):] if path.startswith("/_switcher/api/") else None
        local_only = (route in ("setup/run", "bench/cancel") or (route == "bench" and method == "POST")
                      or (route == "action" and data.get("action") in LOCAL_ONLY_ACTIONS))
        if not local and local_only:
            await respond(writer, 403, {"ok": False, "message": "only on the PC itself"})
        elif route == "status":
            await respond(writer, 200, self.status())
        elif route == "log":
            await respond(writer, 200, {"backend": self.host.follower.tail(300) if self.host.follower else "",
                                        "switcher": tail_file(LOG_DIR / "manager.log", 150)})
        elif route == "config" and method == "GET":
            # other devices already hold the link; they don't need it handed back
            cfg = self.cfg if local else {**self.cfg, "dashboard_token": ""}
            await respond(writer, 200, {"config": cfg, "defaults": config.DEFAULTS,
                                        "service_keys": sorted(SERVICE_KEYS), "models": self.model_choices()})
        elif route == "config" and method == "POST":
            changes = data.get("config", {})
            await respond(writer, *self.update_config(changes if isinstance(changes, dict) else {}, local))
        elif route == "action" and method == "POST":
            await respond(writer, *self.action(str(data.get("action", ""))))
        elif route == "setup" and method == "GET":
            since = int((re.search(r"since=(\d+)", target) or [0, 0])[1])
            await respond(writer, 200, self.setup.report(since))
        elif route == "hardware" and method in ("GET", "POST"):
            if not self.setup.hw or method == "POST":     # POST: look again
                await self.setup.check()
            await respond(writer, 200, self.setup.hw or {})
        elif route == "metrics" and method == "GET":
            await respond(writer, 200, {"recent": list(self.metrics)[-100:], "summary": self.metrics_summary()})
        elif route == "setup/check" and method == "POST":
            asyncio.create_task(self.setup.check())
            await respond(writer, 200, {"ok": True})
        elif route == "setup/run" and method == "POST":
            if self.observe:
                await respond(writer, 409, {"ok": False, "message": "observe mode"})
                return
            if self.bench.running:
                await respond(writer, 409, {"ok": False, "message": "a benchmark is running"})
                return
            ok, msg = await self.setup.run(str(data.get("step", "")))
            await respond(writer, 200 if ok else 409, {"ok": ok, "message": msg})
        elif route == "bench" and method == "GET":
            q = urllib.parse.parse_qs(urllib.parse.urlsplit(target).query)
            since = int(q["since"][0]) if q.get("since", [""])[0].isdigit() else 0
            await respond(writer, 200, self.bench.report(since, (q.get("run") or [None])[0]))
        elif route == "bench" and method == "POST":
            runs = data.get("runs", ["current"])
            if isinstance(runs, str):
                runs = [runs]
            if not isinstance(runs, list) or len(runs) > 64 or not all(isinstance(x, str) and len(x) <= 64
                                                                       for x in runs):
                await respond(writer, 400, {"ok": False, "message": "runs: a list of presets or run specs"})
                return
            ok, msg = self.bench.start(runs, str(data.get("workload", "standard")), resume=data.get("resume") is True)
            await respond(writer, 200 if ok else 409, {"ok": ok, "message": msg})
        elif route == "bench/cancel" and method == "POST":
            ok, msg = await self.bench.cancel()
            await respond(writer, 200 if ok else 409, {"ok": ok, "message": msg})
        else:
            await respond(writer, 404, {"error": "not found"})

    def update_config(self, changes, local=True):
        restart_keys = {"quant", "context", "lean_vram_gb", "lean_ram_gb", "full_vram_headroom_gb", "max_ram_cache_gib"}
        changed, service = [], []
        new = {}
        for k, v in changes.items():   # check every value before changing any
            if k not in config.DEFAULTS:
                continue
            try:
                v = config.coerce(k, v)
            except (TypeError, ValueError):
                return 400, {"ok": False, "message": f"bad value for {k}"}
            if k in config.FILE_ONLY_KEYS and v != self.cfg[k]:
                return 400, {"ok": False, "message": f"{k} can only be changed in config.json"}
            if k in config.LOCAL_ONLY_KEYS and v != self.cfg[k] and not local:
                # the private link is built from it: another device could point the next link at itself
                return 403, {"ok": False, "message": f"{k} can only be changed on the PC itself"}
            if k == "quant" and v not in sizing.QUANTS:
                return 400, {"ok": False, "message": "unknown quant"}
            if k == "context" and not 4096 <= v <= 262144:
                return 400, {"ok": False, "message": "context must be between 4096 and 262144"}
            if k == "remote_host" and not re.fullmatch(r"[A-Za-z0-9.-]*", v):
                return 400, {"ok": False, "message": "the PC's name can only have letters, digits, dots and dashes"}
            new[k] = v
        for k, v in new.items():
            if self.cfg[k] != v:
                self.cfg[k] = v
                (service if k in SERVICE_KEYS else changed).append(k)
        config.save_config(self.cfg)
        msg = "saved"
        if changed or service:
            self.event("config", "changed: " + ", ".join(changed + service))
        if restart_keys & set(changed) and self.be.state == "running" and not self.observe:
            self.spawn(self.restart(self.be.tier, "settings changed"))
            msg = "saved; restarting the model to apply"
        if service:
            msg += "; " + ", ".join(service) + " apply when the service restarts"
        return 200, {"ok": True, "message": msg, "changed": changed + service}

    def action(self, a):
        cfg = self.cfg
        lifecycle = {"start_full", "start_lean", "pause", "restart"}
        if self.observe and a in lifecycle:
            return 409, {"ok": False, "message": "observe mode: this manager does not control the model"}
        if self.bench.running and a in lifecycle:
            return 409, {"ok": False, "message": "a benchmark is running; cancel it in the Benchmark tab first"}
        msg = None
        if a in ("start_full", "start_lean"):
            # a lean start chosen by hand stays lean until you choose otherwise (no automatic re-saturation)
            self.manual_lean = a == "start_lean"
            tier = a.split("_")[1]
            if self.be.state == "running" and self.be.tier != tier:
                ok = self.spawn(self.to_full("chosen in the dashboard", self.mon.get(), manual=True) if tier == "full"
                                else self.change_tier(tier, "chosen in the dashboard", manual=True))
            else:
                ok = self.spawn(self.start(tier, "started from dashboard"))
        elif a == "pause":
            self.manual_lean = False
            ok = self.spawn(self.stop("paused from dashboard", "manual"))
        elif a == "restart":
            ok = self.spawn(self.restart(self.be.tier or "full", "restarted from dashboard"))
        elif a in ("auto_on", "auto_off"):
            cfg["auto_switch"] = a == "auto_on"
            config.save_config(cfg)
            self.event("config", f"auto-switch {'on' if cfg['auto_switch'] else 'off'}")
            ok = True
        elif a in ("wake_on", "wake_off"):
            cfg["wake_on_request"] = a == "wake_on"
            config.save_config(cfg)
            ok = True
        elif a in ("autostart_on", "autostart_off"):
            ok, msg = self.host.set_autostart(a == "autostart_on")
        elif a == "new_remote_link":
            cfg["dashboard_token"] = secrets.token_urlsafe(18)
            config.save_config(cfg)
            self.event("config", "new private link for other devices; the old one no longer works")
            ok = True
        elif a == "forget_learned":
            self.learned.pop(self.learned_key(), None)
            config.save_json("learned.json", self.learned)
            self.event("config", "forgot the learned limits for this GPU/quant/context")
            ok = True
        elif a == "quit":
            self.spawn(self.shutdown(stop_model=True))
            ok = True
        elif a == "quit_keep_model":
            self.spawn(self.shutdown(stop_model=False))
            ok = True
        else:
            return 400, {"ok": False, "message": "unknown action"}
        if not ok:
            return 409, {"ok": False, "message": msg or "another change is in progress"}
        return 200, {"ok": True, "message": msg}

    async def shutdown(self, stop_model=True):
        self.event("quit", "service stopping" + ("" if stop_model else " (model left running for the next start)"))
        if stop_model:
            await self.stop("service quit")
        self.quit.set()

    # ================================================================ control loop
    async def control_loop(self):
        n = 0
        while True:
            await asyncio.sleep(1.0)
            n += 1
            try:
                await self.tick()
                if n % 10 == 0 and self.be.state == "running" and not self.observe:
                    await self.check_alive()
                if n % 30 == 0:
                    await self.bind_front()
                if n % 600 == 0:
                    asyncio.create_task(self.setup.check())
            except Exception as e:  # noqa: BLE001
                log.exception("tick failed: %s", e)

    async def tick(self):
        cfg = self.cfg
        m = self.mon.get()
        if not m:
            return
        now = time.time()
        level, reasons, ext = self.evaluate(m)
        self.pressure = {"level": level, "reasons": reasons}
        self.sev_since = (self.sev_since or now) if level == "severe" else None
        self.mod_since = (self.mod_since or now) if level != "none" else None
        self.clear_since = (self.clear_since or now) if level == "none" else None
        if m["t"] != self._last_hist_t:
            self._last_hist_t = m["t"]
            self.hist.append([round(m["t"], 1), round(ext / GB, 2), round(max(0.0, m["vram_used"] - ext) / GB, 2),
                              round((m["ram_total"] - m["ram_avail"]) / GB, 2), round(m["gpu_util"])])
        # a RAM cache growing in the background is watched whatever else happens (a brake, not a switch)
        if self.ram_growing and self.be.state == "running" and not self.ram_watch and not self.observe:
            self.ram_watch = True
            asyncio.create_task(self.watch_ram_growth(m))
        if (self.observe or not cfg["auto_switch"] or self.setup.blocking or self.bench.running
                or (self.task and not self.task.done())):
            return
        st, tier = self.be.state, self.be.tier
        idle = self.in_flight == 0 and (self.last_done is None or now - self.last_done >= 10)
        overridden = now < self.override_until
        dwell_ok = now - self.last_transition >= cfg["min_dwell_s"]
        if st == "running":
            if self.sev_since and now - self.sev_since >= cfg["severe_after_s"] and not overridden:
                waited = now - self.sev_since
                if self.in_flight == 0 or waited >= cfg["severe_max_wait_s"]:
                    self.cap_after_ram_pressure(m, reasons)
                    self.spawn(self.stop("pressure: " + "; ".join(reasons), "pressure"))
                return
            # scaling DOWN is never delayed by the dwell time (backing off from a game should be quick);
            # the sustain window (moderate_after_s) is its hysteresis. Dwell only guards scaling back up.
            ram_short = any(r.startswith("RAM") for r in reasons)
            # a model switched to lean in place still has full's RAM cache: short of RAM, it gets lean's
            big_lean = tier == "lean" and ram_short and (self.be.ram_gib or 0) > cfg["lean_ram_gb"]
            if ((tier == "full" or big_lean) and self.mod_since and now - self.mod_since >= cfg["moderate_after_s"]
                    and idle and not overridden):
                why = "pressure: " + "; ".join(reasons)
                if tier == "full":
                    self.manual_lean = False
                self.cap_after_ram_pressure(m, reasons)
                if cfg["pressure_response"] == "lean" and self.best_tier(m, level="moderate") == "lean":
                    # short of VRAM only: resize in place and keep the RAM cache. Short of RAM: shrink it to
                    # lean's in place too where the engine can, else restart smaller
                    live_ram = ram_short and self.ram_resizable()
                    self.spawn(self.change_tier("lean", why, in_place=not ram_short or live_ram,
                                                ram_gib=cfg["lean_ram_gb"] if live_ram else None))
                else:
                    self.spawn(self.stop(why, "pressure"))
                return
            # Re-saturate lean -> full. A busy agent never leaves a 10 s idle gap, so don't wait for one: once lean
            # has run 2 min with no pressure, restart between requests (requests arriving meanwhile are held).
            if (self.be.ready_at and not self.be.ram_measured and now - self.be.ready_at > RAM_GRACE_S
                    and self.in_flight == 0):
                self.measure_ram_overhead(m)
            if (tier == "lean" and cfg["upgrade_when_idle"] and not self.manual_lean and self.clear_since
                    and now - self.clear_since >= cfg["recover_after_s"] and self.in_flight == 0 and dwell_ok
                    and (idle or now - (self.be.ready_at or now) >= 120)
                    and self.best_tier(m, level="none") == "full"):
                self.spawn(self.to_full("resources free again: re-saturating", m))
                return
            # Grow a RAM cache that was sized while RAM was short (e.g. right after another big process exited).
            if (tier == "full" and cfg["upgrade_when_idle"] and self.be.ram_gib and self.be.ram_measured
                    and self.clear_since and now - self.clear_since >= cfg["recover_after_s"] and dwell_ok
                    and self.in_flight == 0 and (idle or now - (self.be.ready_at or now) >= 120)):
                want = self.full_ram_gib(m, self.plan_for("full", m))
                if want - self.be.ram_gib >= 8:
                    why = f"more RAM is free: growing the RAM cache from {self.be.ram_gib} to about {want} GiB"
                    self.spawn(self.change_tier("full", why, ram_gib=want) if self.ram_resizable()
                               else self.restart("full", why))
                    return
            if (cfg["idle_unload_min"] > 0 and self.in_flight == 0
                    and now - (self.last_done or self.be.ready_at or now) >= cfg["idle_unload_min"] * 60):
                self.spawn(self.stop(f"idle for {cfg['idle_unload_min']} min", "idle"))
        elif st == "stopped":
            if (cfg["preload_when_free"] and self.paused_reason == "pressure" and self.clear_since
                    and now - self.clear_since >= cfg["recover_after_s"] and dwell_ok):
                t = self.best_tier(m, level="none")
                if t:
                    self.spawn(self.start(t, "resources free again: preloading"))

    # ================================================================ main
    async def serve(self, open_window=False):
        self.tlock = asyncio.Lock()
        self.quit = asyncio.Event()
        cfg = self.cfg
        config.ensure_token(cfg, save=not self.observe)
        self.mon.start()
        hosts = ["127.0.0.1", "::1"] if cfg["listen_host"] == "127.0.0.1" else [cfg["listen_host"]]
        try:
            server = await asyncio.start_server(self.handle, hosts, cfg["listen_port"], limit=8 * 1024 * 1024)
        except OSError as e:
            raise SystemExit(f"port {cfg['listen_port']} is in use: {e}") from e
        try:
            self.host.start_follower()
        except Exception as e:  # noqa: BLE001 - no WSL yet: setup will create it
            log.warning("log follower not started: %s", e)
        await self.adopt_or_clean()
        self.event("service", f"{'observing' if self.observe else 'listening'} on {cfg['listen_host']}:"
                              f"{cfg['listen_port']} ({self.host.label}; model on :{cfg['backend_port']})")
        await self.bind_front()
        asyncio.create_task(self.control_loop())
        asyncio.create_task(self.setup.check())
        if cfg["preload_on_launch"] and self.be.state == "stopped" and not self.observe:
            await asyncio.sleep(3)
            m = self.mon.get()
            t = self.best_tier(m) if m else None
            if t:
                self.spawn(self.start(t, "preload on launch"))
        if open_window:
            self.host.open_ui(f"http://127.0.0.1:{cfg['listen_port']}/?token={cfg['dashboard_token']}")
        async with server:
            await self.quit.wait()
        if self.host.follower:
            self.host.follower.stop()


LOCAL_NAMES = ("127.0.0.1", "localhost", "::1")


def header(headers, name):
    return next((v for k, v in headers if k.lower() == name), "")


def host_name(headers):
    """The Host header's name without the port: 'localhost:30000' -> 'localhost', '[::1]:30000' -> '::1'."""
    try:
        return urllib.parse.urlsplit("//" + header(headers, "host")).hostname or ""
    except ValueError:
        return ""


def token_count(*values):
    """The first whole number from 0 to 10M; else 0. A client can echo its own 'usage' text into a response, so the
    figures read from it are never trusted as they are."""
    for v in values:
        if isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 10_000_000:
            return v
    return 0


def unspecified(name):
    """An address that listens on every interface (all zeros)."""
    try:
        return ipaddress.ip_address(name).is_unspecified
    except ValueError:
        return False


def is_ip(name):
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        return False


def same_origin(headers):
    """False for a request that a web page on another site sent here. Browsers name the sending page in Origin;
    programs (curl, Hermes, the CLI) send none and pass."""
    origin = header(headers, "origin")
    if not origin:
        return True
    try:
        return urllib.parse.urlsplit(origin).netloc.lower() == header(headers, "host").lower()
    except ValueError:
        return False


def query_token(target):
    """The token= value in a request target's query string, or ""."""
    return urllib.parse.parse_qs(urllib.parse.urlsplit(target).query).get("token", [""])[0]


def tail_file(path, lines=200):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 96 * 1024))
            return "\n".join(f.read().decode("utf-8", "replace").splitlines()[-lines:])
    except OSError:
        return ""
