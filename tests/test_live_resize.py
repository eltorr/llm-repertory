"""Switching between full and lean without a restart: the arena arithmetic, the request the manager sends, the
server's check of it, and when the manager resizes in place or restarts. No GPU: the engine call is faked."""
import asyncio
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

os.environ["REPERTORY_HOME"] = tempfile.mkdtemp(prefix="repertory-test-")      # never a real settings dir
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engines" / "exl3_offload"))
sys.path.append(str(ROOT / "glm53-flash-offload" / "glm53"))
from engines.exl3_offload import sizing  # noqa: E402
from repertory import config  # noqa: E402
import server  # noqa: E402  (the model server: nothing heavy is imported until it runs)

try:
    import vram_arena    # the engine's, once Repertory's live resize is in it
except ImportError:
    vram_arena = None

GB, GIB = 1e9, 1024 ** 3
REC, SPC = 2_457_600, 32          # a 4.05 bpw expert slot; slots per chunk


@unittest.skipUnless(vram_arena, "the engine has no vram_arena yet (Repertory's live resize is not in it)")
class ArenaTests(unittest.TestCase):
    def test_a_lower_limit_frees_whole_chunks_covering_the_shortfall(self):
        # 20 GB in use, limit 15 GB, 1.7 GB reserve: 6.7 GB over, so 2,727 slots go, rounded to whole chunks
        n = vram_arena.target_slots(7488, REC, SPC, 24576, used_bytes=20e9, limit_bytes=15e9, reserve_bytes=1.7e9)
        self.assertEqual(n % SPC, 0)
        self.assertGreaterEqual(7488 - n, 6.7e9 / REC)
        self.assertLess(7488 - n, 6.7e9 / REC + SPC)

    def test_a_higher_limit_grows_it_up_to_the_cap_and_the_capacity(self):
        grow = dict(used_bytes=10e9, limit_bytes=30e9, reserve_bytes=1.7e9)
        want = (2272 + int(18.3e9 // REC)) // SPC * SPC
        self.assertEqual(vram_arena.target_slots(2272, REC, SPC, 24576, **grow), want)
        self.assertEqual(vram_arena.target_slots(2272, REC, SPC, 24576, max_slots=4000, **grow), 4000 // SPC * SPC)
        self.assertEqual(vram_arena.target_slots(2272, REC, SPC, 3200, **grow), 3200)

    def test_without_a_limit_it_only_shrinks_and_never_below_zero(self):
        self.assertEqual(vram_arena.target_slots(7488, REC, SPC, 24576, max_slots=2272), 2272)
        self.assertEqual(vram_arena.target_slots(2272, REC, SPC, 24576, max_slots=7488), 2272)
        self.assertEqual(vram_arena.target_slots(2272, REC, SPC, 24576), 2272)
        self.assertEqual(vram_arena.target_slots(64, REC, SPC, 24576, used_bytes=30e9, limit_bytes=10e9), 0)
        with self.assertRaises(ValueError):
            vram_arena.target_slots(64, REC, SPC, 24576, limit_bytes=10e9)

    def test_buffers_kept_since_load_count_against_the_reserve(self):
        self.assertEqual(vram_arena.reserve_left(1.7e9, 0.5e9), 1.2e9)
        self.assertEqual(vram_arena.reserve_left(1.7e9, 3e9), 0)
        self.assertEqual(vram_arena.reserve_left(1.7e9, -0.2e9), 1.7e9)

    def test_flex_chunks_are_the_last_ones_as_at_load(self):
        chunk = SPC * REC
        for n, elastic in ((234, 6 * GIB), (71, 6.5 * GIB), (3, 6 * GIB), (10, 0)):
            # expert_cache.Pool's own formula before the resize existed
            old = min(n, int(-(-elastic // chunk))) if elastic else 0
            self.assertEqual(vram_arena.flex_chunks(n, elastic, chunk), list(range(n - old, n)))


class RequestTests(unittest.TestCase):
    def test_lean_caps_the_slots_and_keeps_upstream_reserve_on_big_budgets(self):
        p = sizing.plan("4.05", 16e9, 262144)
        a = sizing.live_resize_args(p, cap_slots=True)
        self.assertEqual(a, {"chunk": p["chunk"], "max_slots": p["slots"],
                             "reserve_bytes": int(sizing.ENGINE_RESERVE_GIB * GIB),
                             "elastic_bytes": int(sizing.ENGINE_ELASTIC_GIB * GIB)})

    def test_small_budgets_pass_the_plans_own_reserve_and_elastic(self):
        p = sizing.plan("3.05", 14e9, 262144)
        self.assertTrue(p["staging"])
        env = sizing.engine_env(p, cap_slots=True)
        a = sizing.live_resize_args(p, cap_slots=True)
        self.assertEqual(a["reserve_bytes"], int(float(env["GLM53_EC_RESERVE_GB"]) * GIB))
        self.assertEqual(a["elastic_bytes"], int(float(env["GLM53_EC_ELASTIC_GB"]) * GIB))

    def test_full_lets_the_engine_size_it_and_staging_free_needs_a_restart(self):
        self.assertIsNone(sizing.live_resize_args(sizing.plan("4.05", 29.6e9, 262144))["max_slots"])
        with self.assertRaises(ValueError):
            sizing.live_resize_args(dict(sizing.plan("4.05", 29.6e9, 262144), staging=False))


class ServerCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.check = staticmethod(server.resize_args)
        cls.move = staticmethod(server.move_ram_limits)
        ServerCheckTests.module = server

    def test_a_manager_request_passes(self):
        body = {"vram_limit_bytes": 18_000_000_000, "max_slots": 2272, "chunk": 8192, "torch_limit_bytes": 16.5e9,
                "reserve_bytes": 1825361100, "elastic_bytes": 6442450944, "ram_bytes": 8 * GIB}
        self.assertEqual(self.check(body)["torch_limit_bytes"], 16_500_000_000)
        self.assertEqual(self.check(body)["ram_bytes"], 8 * GIB)
        self.assertEqual(self.check({}), {})
        self.assertEqual(self.check({"max_slots": None}), {})

    def test_bad_values_are_refused(self):
        for body in ([], {"slots": 5}, {"max_slots": -1}, {"max_slots": 1.5}, {"max_slots": True},
                     {"max_slots": "64"}, {"vram_limit_bytes": 0}, {"elastic_bytes": 0}, {"ram_bytes": 0},
                     {"ram_bytes": -GIB}, {"chunk": 1000},
                     {"chunk": 131072},
                     {"vram_limit_bytes": float("nan")}, {"vram_limit_bytes": float("inf")}):
            with self.assertRaises(ValueError, msg=body):
                self.check(body)


class RamLimitTests(unittest.TestCase):
    """server.move_ram_limits: the memory cgroup's limits and the RAM guard follow the RAM cache."""
    @classmethod
    def setUpClass(cls):
        ServerCheckTests.setUpClass()
        cls.move = staticmethod(ServerCheckTests.move)

    def host(self=None, used=20 * GIB, mx=53 * GIB, high=50 * GIB):
        d = Path(tempfile.mkdtemp(prefix="repertory-cg-"))
        (d / "memory.current").write_text(str(used))
        for name, v in (("memory.max", mx), ("memory.high", high)):
            if v is not None:
                (d / name).write_text(str(v))
        h = types.SimpleNamespace(path=d, limit=int(53 * GIB - 1.5e9))
        h.sample = lambda: int((d / "memory.current").read_text())
        return h, d

    def read(self, d, name):
        return (d / name).read_text().strip()

    def test_growing_raises_every_limit_by_the_growth(self):
        h, d = self.host()
        self.assertTrue(self.move(h, 26 * GIB))
        self.assertEqual((self.read(d, "memory.max"), self.read(d, "memory.high")), (str(79 * GIB), str(76 * GIB)))
        self.assertEqual(h.limit, int(53 * GIB - 1.5e9) + 26 * GIB)

    def test_shrinking_lowers_them_but_never_below_whats_in_use(self):
        h, d = self.host(used=20 * GIB)
        self.assertTrue(self.move(h, -26 * GIB))
        self.assertEqual((self.read(d, "memory.max"), self.read(d, "memory.high")), (str(27 * GIB), str(24 * GIB)))
        h, d = self.host(used=30 * GIB)                    # still in use (not freed yet): 2 GiB above that
        self.move(h, -26 * GIB)
        self.assertEqual((self.read(d, "memory.max"), self.read(d, "memory.high")), (str(32 * GIB), str(32 * GIB)))
        self.assertEqual(h.limit, 32 * GIB)

    def test_no_cgroup_limit_and_no_change_are_fine(self):
        h, d = self.host(mx="max", high="max")
        self.assertTrue(self.move(h, 26 * GIB))
        self.assertEqual(self.read(d, "memory.max"), "max")
        self.assertTrue(self.move(h, 0))
        h, d = self.host(mx=None, high=None)               # no files at all (the root cgroup)
        self.assertTrue(self.move(h, -8 * GIB))

    def test_a_limit_that_cant_be_raised_stops_the_growth_and_changes_nothing(self):
        h, d = self.host()
        real = Path.write_text

        def write(path, text, *a, **kw):
            if path.name == "memory.high":
                raise PermissionError("read-only")
            return real(path, text, *a, **kw)
        with mock.patch.object(Path, "write_text", write):
            self.assertFalse(self.move(h, 26 * GIB))
        self.assertEqual((self.read(d, "memory.max"), h.limit), (str(53 * GIB), int(53 * GIB - 1.5e9)))


class FakeNv:
    """What server.RamWorker uses of nv2's engine: one chunk (or a smaller step) per grow step."""
    def __init__(self, nram, fail_at=None):
        self.nram, self.slot, self.ram, self.freed, self.fail_at = nram, REC, types.SimpleNamespace(cs=128), [], fail_at
        self.steps, self.free_times = [], []

    def grow_ram_step(self, target, step=None):
        if self.fail_at is not None and self.nram >= self.fail_at:
            raise RuntimeError("cudaHostAlloc failed")
        n = min(step or self.ram.cs, self.ram.cs, target - self.nram)
        time.sleep(0.005)
        self.nram += n
        self.steps.append((time.monotonic(), n))
        return n

    def free_ram_chunks(self, chunks):
        self.freed.extend(c[0] for c in chunks)
        self.free_times.append(time.monotonic())

    def reattach_ram_chunks(self, chunks):
        self.reattached = getattr(self, "reattached", []) + [c[0] for c in chunks]
        n = sum(c[2] for c in chunks)
        self.nram += n
        return n


def chunks(*names, k=128):
    """What a shrink takes out, in the arena's order: (buffer, address, slots)."""
    return [(n, i, k) for i, n in enumerate(names)]


class RamWorkerTests(unittest.TestCase):
    """server.RamWorker: the RAM cache grown in the background and freed after a shrink, the limits following
    what it really adds or frees."""
    @classmethod
    def setUpClass(cls):
        ServerCheckTests.setUpClass()
        cls.qs = ServerCheckTests.module

    def worker(self, nram=3456, available=lambda: 100 * GIB, busy=lambda: False, **kw):
        h, d = RamLimitTests.host(None)
        nv = FakeNv(nram, **kw)
        return self.qs.RamWorker(nv, h, available=available, busy=busy), nv, h, d

    def test_a_grow_runs_in_the_background_to_its_target(self):
        w, nv, h, d = self.worker()
        limit0, target = h.limit, 15291
        self.assertTrue(w.prepare_grow(target))
        w.grow(target)
        self.assertTrue(w.info()["growing"] or nv.nram == target)     # the call returns at once
        w.wait()
        self.assertEqual((nv.nram, w.info()["growing"], w.info()["stopped"]), (target, False, None))
        self.assertEqual(h.limit, limit0 + (target - 3456) * REC)     # raised by what it added, exactly

    def test_it_stops_when_this_system_runs_short_and_lowers_the_limits_to_what_it_got(self):
        reads = iter([100 * GIB] * 3 + [3 * GIB] * 100)
        w, nv, h, d = self.worker(available=lambda: next(reads))
        limit0 = h.limit
        w.prepare_grow(15291)
        w.grow(15291)
        w.wait()
        self.assertEqual(nv.nram, 3456 + 3 * 128)
        self.assertEqual(w.info()["stopped"], "this system is short of RAM")
        self.assertEqual(h.limit, limit0 + 3 * 128 * REC)

    def test_a_cancel_or_a_failure_keeps_what_it_has(self):
        w, nv, h, d = self.worker()
        limit0 = h.limit
        w.prepare_grow(15291)
        w.grow(15291)
        time.sleep(0.02)
        w.wait(cancel=True)
        self.assertLess(nv.nram, 15291)
        self.assertEqual((w.info()["stopped"], w.info()["growing"]), ("stopped", False))
        self.assertEqual(h.limit, limit0 + (nv.nram - 3456) * REC)
        w, nv, h, d = self.worker(fail_at=4000)
        w.prepare_grow(15291)
        w.grow(15291)
        w.wait()
        self.assertTrue(w.info()["stopped"].startswith("error: cudaHostAlloc failed"))
        self.assertEqual(nv.nram, 3456 + 5 * 128)

    def test_a_shrinks_chunks_are_freed_in_the_background_then_the_limits_lowered(self):
        w, nv, h, d = self.worker(nram=3456)
        limit0 = h.limit
        with mock.patch.dict(os.environ, {"QWEN_RAM_COMPACT": "0"}):
            w.settle(3456, chunks("c1", "c2", k=6000))
            w.wait()
        self.assertEqual(nv.freed, ["c2", "c1"])                       # the highest first
        self.assertEqual(h.limit, max(limit0 - 12000 * REC, 20 * GIB + 2 * GIB))
        self.assertEqual((w.info()["freeing"], w.info()["pending_bytes"]), (False, 0))

    def test_while_requests_run_a_grow_adds_small_steps_now_and_then_and_speeds_up_between_them(self):
        running = threading.Event()
        running.set()
        w, nv, h, d = self.worker(busy=running.is_set)
        w.BUSY_GROW_S = 0.1
        w.prepare_grow(15291)
        w.grow(15291)
        time.sleep(0.35)
        running.clear()                                                # the request ends: whole chunks, at once
        w.wait()
        self.assertEqual((nv.nram, w.info()["stopped"]), (15291, None))
        sizes = [n for _, n in nv.steps]
        during = nv.steps[:sizes.index(128)]                           # small steps while it ran, then whole chunks
        self.assertTrue(2 <= len(during) <= 6, sizes[:8])              # one per BUSY_GROW_S, not back to back
        self.assertTrue(all(n == w.BUSY_STEP for _, n in during) and w.BUSY_STEP not in sizes[len(during):-1])
        self.assertTrue(all(b - a >= 0.09 for (a, _), (b, _) in zip(during, during[1:], strict=False)), during)
        self.assertEqual(w.info()["grow_busy_steps"], len(during))

    def test_a_cancel_reaches_a_grow_waiting_for_a_gap(self):
        w, nv, h, d = self.worker(busy=lambda: True)
        w.prepare_grow(15291)
        w.grow(15291)
        time.sleep(0.05)
        t0 = time.monotonic()
        w.wait(cancel=True)
        self.assertLess(time.monotonic() - t0, 0.2)                    # not BUSY_GROW_S
        self.assertEqual((nv.nram, w.info()["stopped"]), (3456 + w.BUSY_STEP, "stopped"))

    def test_while_requests_run_a_free_hands_back_a_chunk_every_busy_free_s(self):
        w, nv, h, d = self.worker(nram=3456, busy=lambda: True)
        w.BUSY_FREE_S = 0.05
        with mock.patch.dict(os.environ, {"QWEN_RAM_COMPACT": "0"}):
            w.settle(3456, chunks("c1", "c2", "c3"))
            w.wait()
        self.assertEqual(nv.freed, ["c3", "c2", "c1"])
        t = nv.free_times
        self.assertTrue(all(b - a >= 0.045 for a, b in zip(t, t[1:], strict=False)), t)

    # ---- memory reused: a grow right after a shrink takes back what is not freed yet
    def shrunk_and_partly_freed(self, names=("c0", "c1", "c2", "c3", "c4")):
        """From 3456 + 640 experts shrunk to 3456 (5 chunks out), the free stopped after one or two chunks."""
        w, nv, h, d = self.worker(nram=3456, busy=lambda: True)
        w.BUSY_FREE_S = 0.05
        limit0 = h.limit
        with mock.patch.dict(os.environ, {"QWEN_RAM_COMPACT": "0"}):
            w.settle(3456, chunks(*names))
            time.sleep(0.07)
            w.wait(cancel=True)
        return w, nv, h, limit0

    def test_a_grow_right_after_a_shrink_takes_back_what_is_not_freed_yet(self):
        w, nv, h, limit0 = self.shrunk_and_partly_freed()
        freed = list(nv.freed)
        self.assertTrue(1 <= len(freed) <= 3, freed)                   # the highest went first
        self.assertEqual(freed, ["c4", "c3", "c2"][:len(freed)])
        self.assertEqual(h.limit, limit0 - len(freed) * 128 * REC)    # down by what was freed only
        self.assertEqual(w.info()["pending_bytes"], (5 - len(freed)) * 128 * REC)
        target = 3456 + 640
        self.assertTrue(w.prepare(target))
        kept = 5 - len(freed)
        self.assertEqual(nv.reattached, ["c0", "c1", "c2", "c3"][:kept])  # back where they were, in order
        self.assertEqual((nv.nram, w.info()["reused_slots"], w.info()["pending_bytes"]), (3456 + kept * 128,
                                                                                           kept * 128, 0))
        w.BUSY_GROW_S = 0.0
        w.settle(target)                                               # the freed part is pinned again
        w.wait()
        self.assertEqual((nv.nram, w.info()["stopped"]), (target, None))
        self.assertEqual(h.limit, limit0)                              # back where it started, exactly

    def test_a_grow_smaller_than_what_is_pending_takes_back_enough_and_frees_the_rest(self):
        w, nv, h, limit0 = self.shrunk_and_partly_freed(("c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"))
        self.assertTrue(w.prepare(3456 + 200))                         # two chunks: the last one partly
        self.assertEqual((nv.reattached, nv.nram), (["c0", "c1"], 3456 + 256))
        w.BUSY_FREE_S = 0.0
        with mock.patch.dict(os.environ, {"QWEN_RAM_COMPACT": "0"}):
            w.settle(3456 + 200)
            w.wait()
        self.assertEqual(set(nv.freed), {"c2", "c3", "c4", "c5", "c6", "c7"})
        self.assertEqual((w.info()["pending_bytes"], w.info()["growing"]), (0, False))
        self.assertEqual(h.limit, limit0 - 6 * 128 * REC)

    def test_a_second_shrink_adds_its_chunks_below_the_pending_ones_and_frees_them_all(self):
        w, nv, h, limit0 = self.shrunk_and_partly_freed(("a0", "a1", "a2", "a3"))
        nv.nram = 3456 - 256
        w.BUSY_FREE_S = 0.0
        with mock.patch.dict(os.environ, {"QWEN_RAM_COMPACT": "0"}):
            self.assertTrue(w.prepare(3200))
            w.settle(3200, chunks("b0", "b1"))
            w.wait()
        self.assertEqual(nv.freed[-2:], ["b1", "b0"])                   # the old ones first, the highest first
        self.assertEqual(w.info()["pending_bytes"], 0)

    def test_the_brake_stops_a_grow_but_a_free_it_interrupts_goes_on(self):
        w, nv, h, limit0 = self.shrunk_and_partly_freed()
        w.BUSY_FREE_S = 0.0
        with mock.patch.dict(os.environ, {"QWEN_RAM_COMPACT": "0"}):
            w.free_pending()                                           # what POST /expert_cache/ram/stop does
            w.wait()
        self.assertEqual(sorted(nv.freed), ["c0", "c1", "c2", "c3", "c4"])
        self.assertEqual(h.limit, limit0 - 5 * 128 * REC)


class ManagerTests(unittest.TestCase):
    """service.Service's switch, on a stand-in that has the methods it uses and fakes for the rest."""
    TOTAL, EXT = 34.19e9, 2.0e9

    def svc(self, state="running", tier="full", ram_gib=34, staging=True):
        from repertory import service
        cfg = dict(config.DEFAULTS, quant="4.05", context=262144, live_resize=True, live_ram_resize=True)
        be = service.Backend()
        be.set(state)
        be.since = 1000.0
        be.tier, be.quant, be.context, be.ram_gib = tier, "4.05", 262144, ram_gib
        be.plan = dict(sizing.plan("4.05", 29.6e9, 262144), staging=staging)
        m = {"t": 1, "vram_total": self.TOTAL, "vram_used": 31e9, "ext_vram_abs": self.EXT, "perproc_ded": 0.0,
             "ram_avail": 20e9}
        s = types.SimpleNamespace(cfg=cfg, be=be, observe=False, live_unsupported=False, live_ram_unsupported=False,
                                  ram_growing=None, ram_watch=False, learned={}, freed=None,
                                  ram_soft_cap=None, gpu_display=True, gpu_name="Test GPU", events=[], calls=[],
                                  backend_auth={"X-Repertory-Key": "k"},
                                  ext_at_start=None, perproc_at_start=None, last_transition=0.0,
                                  bench=types.SimpleNamespace(base=None), tlock=asyncio.Lock(),
                                  host=types.SimpleNamespace(kind="windows-wsl", engine_dir="/root/repertory"),
                                  mon=types.SimpleNamespace(get=lambda: m))
        s.event = lambda k, msg: s.events.append((k, msg))

        async def fresh_sample(after):
            return m

        async def stop(reason, final_state="stopped"):
            s.calls.append(("stop", reason))

        async def start(t, reason):
            s.calls.append(("start", t))
            return True
        s.fresh_sample, s._stop_locked, s._start_locked = fresh_sample, stop, start
        for name in ("plan_for", "full_budget", "lean_budget", "headroom", "ext_vram", "learned_key", "shift_request",
                     "_shift_locked", "change_tier", "keeps_ram", "full_ram_gib", "ram_for_start", "soft_ram_cap",
                     "own_ram_estimate", "ram_overhead", "save_backend_meta", "ram_resizable", "to_full",
                     "watch_ram_growth"):
            setattr(s, name, types.MethodType(getattr(service.Service, name), s))
        return s

    def run_change(self, s, tier, reply, in_place=True, manual=False, ram_gib=None, coro=None):
        """Runs change_tier (or coro(s)) with the engine's reply faked; returns (result, the bodies sent)."""
        from repertory import service
        sent = []

        def post(url, body, timeout, headers):
            sent.append((url, body, headers))
            if isinstance(reply, Exception):
                raise reply
            return reply
        with mock.patch.object(service, "http_post_json", post):
            c = coro(s) if coro else s.change_tier(tier, "test", in_place=in_place, manual=manual, ram_gib=ram_gib)
            return asyncio.run(c), sent

    def test_vram_pressure_resizes_in_place(self):
        s = self.svc()
        ok, sent = self.run_change(s, "lean", (200, {"pools": [{"slots_before": 7488, "slots": 2272,
                                                                 "arena_bytes": 5.58e9}], "chunk": 8192,
                                                      "seconds": 0.9}))
        self.assertTrue(ok)
        self.assertEqual(s.calls, [])                              # no stop, no start
        self.assertEqual((s.be.state, s.be.tier, s.be.since), ("running", "lean", 1000.0))
        url, body, _ = sent[0]
        self.assertTrue(url.endswith("/expert_cache/resize"))
        lean = s.plan_for("lean", s.mon.get())
        self.assertEqual(body["max_slots"], lean["slots"])
        self.assertEqual(body["chunk"], lean["chunk"])
        self.assertEqual(body["vram_limit_bytes"], int(self.EXT + lean["budget"]))
        self.assertEqual(body["torch_limit_bytes"], int(lean["budget"] + 0.5 * GB))
        self.assertEqual(s.be.budget, lean["budget"])
        meta = config.load_json("backend.json", {})
        self.assertEqual((meta["tier"], meta["env"]["GLM53_EC_MAX_SLOTS"]), ("lean", str(lean["slots"])))
        self.assertEqual(s.events[-1][0], "shift")

    def test_back_to_full_lets_the_engine_size_the_cache(self):
        s = self.svc(tier="lean")
        ok, sent = self.run_change(s, "full", (200, {"pools": [{}], "chunk": 24576, "seconds": 0.4}))
        self.assertTrue(ok)
        body = sent[0][1]
        self.assertIsNone(body["max_slots"])
        self.assertEqual(body["vram_limit_bytes"], int(self.TOTAL - s.headroom(s.mon.get())))

    def test_an_engine_without_the_route_restarts_and_is_remembered(self):
        s = self.svc()
        ok, _ = self.run_change(s, "lean", (404, None))
        self.assertTrue(ok)
        self.assertEqual(s.calls, [("stop", "test"), ("start", "lean")])
        self.assertTrue(s.live_unsupported)
        s.calls.clear()
        s.be.tier = "full"
        ok, sent = self.run_change(s, "lean", (200, {}))
        self.assertEqual((sent, s.calls[-1]), ([], ("start", "lean")))    # no second try

    def test_a_request_that_just_started_puts_it_off(self):
        s = self.svc()
        ok, _ = self.run_change(s, "lean", (409, {"error": "a request is still running"}))
        self.assertFalse(ok)
        self.assertEqual((s.calls, s.be.state, s.be.tier), ([], "running", "full"))

    def test_a_dashboard_switch_put_off_by_a_long_request_restarts_as_before(self):
        s = self.svc()
        ok, _ = self.run_change(s, "lean", (409, {"error": "a request is still running"}), manual=True)
        self.assertTrue(ok)
        self.assertEqual(s.calls, [("stop", "test"), ("start", "lean")])
        self.assertIn("still running", s.events[-1][1])

    def test_a_broken_reply_leaves_it_running_and_restarts(self):
        import http.client
        s = self.svc()
        ok, _ = self.run_change(s, "lean", http.client.IncompleteRead(b""))
        self.assertEqual((s.be.target, s.calls[-1]), (None, ("start", "lean")))
        self.assertNotEqual(s.be.state, "resizing")

    def test_failures_and_ram_pressure_restart(self):
        for kw, reply in (({}, (500, {"error": "boom"})), ({}, OSError("refused")), ({"in_place": False}, (200, {}))):
            s = self.svc()
            ok, sent = self.run_change(s, "lean", reply, **kw)
            self.assertEqual(s.calls, [("stop", "test"), ("start", "lean")], kw)
            self.assertEqual(len(sent), 0 if kw else 1)
        s = self.svc(staging=False)                                # a staging-free plan allocates differently
        self.run_change(s, "lean", (200, {}))
        self.assertEqual(s.calls[-1], ("start", "lean"))
        s = self.svc()
        s.cfg["live_resize"] = False
        _, sent = self.run_change(s, "lean", (200, {}))
        self.assertEqual((sent, s.calls[-1]), ([], ("start", "lean")))

    def test_lean_that_kept_fulls_ram_cache_caps_the_next_full_start_when_ram_runs_short(self):
        from repertory import service
        s = self.svc(tier="lean", ram_gib=34)
        s.cap_after_ram_pressure = types.MethodType(service.Service.cap_after_ram_pressure, s)
        m = dict(s.mon.get(), ram_avail=4e9)
        s.cap_after_ram_pressure(m, ["RAM low (4.0 GB free)"])
        self.assertIsNotNone(s.ram_soft_cap)
        s = self.svc(tier="lean", ram_gib=8)                       # lean's own RAM cache: nothing to learn
        s.cap_after_ram_pressure = types.MethodType(service.Service.cap_after_ram_pressure, s)
        s.cap_after_ram_pressure(m, ["RAM low (4.0 GB free)"])
        self.assertIsNone(s.ram_soft_cap)

    def test_full_restarts_when_lean_began_with_a_smaller_ram_cache(self):
        s = self.svc(tier="lean", ram_gib=8)
        m = s.mon.get()
        self.assertFalse(s.keeps_ram("full", m))
        s.be.ram_gib = 34                                          # what full gets here
        self.assertLess(s.full_ram_gib(m, s.plan_for("full", m)) - 34, 8)
        self.assertTrue(s.keeps_ram("full", m))
        self.assertTrue(s.keeps_ram("lean", m))


    # ---- the RAM cache (live_ram_resize)
    RAM_REPLY = (200, {"pools": [{"slots_before": 7264, "slots": 1920}], "chunk": 8192, "seconds": 1.2,
                       "ram": {"ram_slots_before": 14854, "ram_slots": 3456, "ram_bytes": 3456 * REC,
                               "dropped": 9000, "moved": 2000}})

    def test_ram_pressure_shrinks_the_ram_cache_in_place(self):
        s = self.svc()
        ok, sent = self.run_change(s, "lean", self.RAM_REPLY, ram_gib=8)
        self.assertTrue(ok)
        self.assertEqual(s.calls, [])
        self.assertEqual(sent[0][1]["ram_bytes"], 8 * GIB)
        self.assertEqual((s.be.tier, s.be.ram_gib), ("lean", 8))                # 3,456 slots: 7.9 GiB
        self.assertEqual(config.load_json("backend.json", {})["env"]["RAM_GB"], "8")
        self.assertIn("RAM cache 34 -> 8 GiB", s.events[-1][1])

    def test_lean_with_the_full_ram_cache_shrinks_only_that(self):
        s = self.svc(tier="lean", ram_gib=34)
        ok, sent = self.run_change(s, "lean", self.RAM_REPLY, ram_gib=8)
        self.assertTrue(ok and sent)
        self.assertEqual((s.calls, s.be.ram_gib), ([], 8))
        self.assertTrue(s.events[-1][1].startswith("lean without a restart"))
        s = self.svc(tier="lean", ram_gib=8)                       # already the lean size: no request
        ok, sent = self.run_change(s, "lean", self.RAM_REPLY, ram_gib=8)
        self.assertEqual(sent, [])

    def test_an_engine_that_cant_resize_ram_restarts_and_is_remembered(self):
        s = self.svc()
        err = (400, {"error": "ram_bytes: the compiled kernels predate RAM resizing; compile them again"})
        self.run_change(s, "lean", err, ram_gib=8)
        self.assertEqual(s.calls, [("stop", "test"), ("start", "lean")])
        self.assertTrue(s.live_ram_unsupported)
        self.assertFalse(s.ram_resizable())
        self.assertIn("predate", s.events[-1][1])

    def test_ram_changes_restart_while_the_setting_is_off(self):
        s = self.svc()
        s.cfg["live_ram_resize"] = False
        _, sent = self.run_change(s, "lean", self.RAM_REPLY, ram_gib=8)
        self.assertEqual((sent, s.calls[-1]), ([], ("start", "lean")))

    def test_back_to_full_grows_the_ram_cache_to_what_a_start_would_get(self):
        s = self.svc(tier="lean", ram_gib=8)
        m = s.mon.get()
        want = s.full_ram_gib(m, s.plan_for("full", m))
        self.assertGreater(want, 8)
        grown = (200, {"pools": [{}], "chunk": 24576, "seconds": 6.0,
                       "ram": {"ram_slots_before": 3456, "ram_slots": int(want * GIB // REC),
                               "ram_bytes": int(want * GIB // REC) * REC, "dropped": 0, "moved": 0}})
        ok, sent = self.run_change(s, "full", grown, coro=lambda s: s.to_full("test", m))
        self.assertTrue(ok)
        self.assertEqual((sent[0][1]["ram_bytes"], s.be.ram_gib, s.calls), (int(want * GIB), want, []))
        s = self.svc(tier="lean", ram_gib=8)                       # without it, a bigger RAM cache needs a restart
        s.cfg["live_ram_resize"] = False
        ok, sent = self.run_change(s, "full", grown, coro=lambda s: s.to_full("test", m))
        self.assertEqual((sent, s.calls), ([], [("stop", "test"), ("start", "full")]))


    # ---- a RAM cache growing in the background
    def test_a_background_grow_counts_its_target_and_is_watched(self):
        s = self.svc(tier="lean", ram_gib=8)
        reply = (200, {"pools": [{}], "chunk": 24576, "seconds": 0.4,
                       "ram": {"growing": True, "target_bytes": 15291 * REC, "ram_slots": 3456, "ram_bytes": 3456 * REC,
                               "ram_slots_before": 3456, "dropped": 0, "moved": 0}})
        ok, _ = self.run_change(s, "full", reply, ram_gib=35)
        self.assertTrue(ok)
        self.assertEqual((s.be.ram_gib, s.ram_growing["from_gib"]), (35, 8))   # the higher estimate meanwhile
        self.assertIn("growing in the background", s.events[-1][1])

    def watch(self, s, m, info):
        from repertory import service
        calls = []

        def get(url, timeout, headers):
            calls.append(("get", url))
            return info

        def post(url, body, timeout, headers):
            calls.append(("post", url))
            return 200, dict(info, growing=False)
        with mock.patch.object(service, "http_get_json", get), mock.patch.object(service, "http_post_json", post):
            asyncio.run(s.watch_ram_growth(m))
        return calls

    def test_the_watch_stops_a_grow_when_free_ram_nears_the_scale_back_line(self):
        s = self.svc(tier="full", ram_gib=35)
        s.ram_growing = {"since": 0, "from_gib": 8}
        m = dict(s.mon.get(), ram_avail=6.5e9)                             # lean's line is 6 GB: stop at 7
        calls = self.watch(s, m, {"growing": True, "ram_bytes": 6000 * REC})
        self.assertEqual(calls, [("post", "http://127.0.0.1:30001/expert_cache/ram/stop")])
        self.assertEqual((s.ram_growing, s.be.ram_gib, s.ram_watch), (None, 14, False))
        self.assertIn("stopped growing the RAM cache at 14 GiB", s.events[-1][1])

    def test_the_watch_waits_while_it_grows_and_takes_the_final_size(self):
        s = self.svc(tier="full", ram_gib=35)
        s.ram_growing = {"since": 0, "from_gib": 8}
        m = dict(s.mon.get(), ram_avail=20e9)
        self.watch(s, m, {"growing": True, "ram_bytes": 9000 * REC})
        self.assertEqual((s.be.ram_gib, bool(s.ram_growing), s.events), (35, True, []))     # still growing
        self.watch(s, m, {"growing": False, "ram_bytes": 15291 * REC, "grow_s": 12.3, "stopped": None})
        self.assertEqual((s.be.ram_gib, s.ram_growing), (35, None))
        self.assertIn("RAM cache 8 -> 35 GiB in the background in 12 s", s.events[-1][1])
        self.assertEqual(config.load_json("backend.json", {})["env"]["RAM_GB"], "35")


if __name__ == "__main__":
    unittest.main()
