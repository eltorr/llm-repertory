"""Manager pieces that need no GPU: response timing, settings coercion, budget choice."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["REPERTORY_HOME"] = tempfile.mkdtemp(prefix="repertory-test-")      # never a real settings dir
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repertory import config, hardware  # noqa: E402
from repertory.httpio import StreamStats  # noqa: E402

SSE = (b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\n",
       b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n',
       b'data: {"choices":[{"delta":{"reasoning_content":"Hm"}}]}\n\n',
       b'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}],'
       b'"usage":{"prompt_tokens":12,"completion_tokens":40,"total_tokens":52}}\n\ndata: [DONE]\n\n')


class DriverTests(unittest.TestCase):
    def test_versions_come_from_setup_lib_sh(self):
        self.assertRegex(hardware.engine_cuda() or "", r"^\d+\.\d+$")
        self.assertRegex(hardware.lib_value("DRIVER_MIN") or "", r"^\d{3}$")

    def test_a_driver_older_than_the_engines_cuda(self):
        need = hardware.engine_cuda()
        major, minor = hardware.version(need)[:2]
        self.assertIsNone(hardware.driver_too_old({"driver": {"cuda": need}}))
        self.assertIsNone(hardware.driver_too_old({"driver": {"cuda": f"{major}.{minor + 1}"}}))
        self.assertIsNone(hardware.driver_too_old({"driver": {"error": "no driver"}}))
        old = hardware.driver_too_old({"driver": {"cuda": f"{major - 1}.9"}})
        self.assertEqual(old, (f"{major - 1}.9", need, hardware.lib_value("DRIVER_MIN")))


class StreamStatsTests(unittest.TestCase):
    def test_streaming_first_token_and_usage(self):
        s = StreamStats()
        for i, chunk in enumerate(SSE):
            s.feed(chunk, float(i))
        self.assertEqual(s.status, 200)
        self.assertEqual(s.t_first, 2.0)            # the role-only event is not a token
        self.assertEqual(s.usage()["completion_tokens"], 40)

    def test_plain_json_response(self):
        s = StreamStats()
        s.feed(b'HTTP/1.1 200 OK\r\n\r\n{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":7}}', 1.0)
        self.assertIsNone(s.t_first)
        self.assertEqual(s.usage()["prompt_tokens"], 5)

    def test_headers_split_across_reads(self):
        s = StreamStats()
        s.feed(b"HTTP/1.1 503 Service", 0.0)
        s.feed(b" Unavailable\r\n\r\n{}", 0.1)
        self.assertEqual(s.status, 503)


class ConfigTests(unittest.TestCase):
    def test_coerce(self):
        self.assertIs(config.coerce("auto_switch", "off"), False)
        self.assertEqual(config.coerce("context", "131072"), 131072)
        self.assertEqual(config.coerce("watchlist", "game.exe\n blender ,"), ["game.exe", "blender"])

    def test_renamed_keys_migrate(self):
        config.CONFIG_PATH.write_text('{"windows_ram_reserve_gb": 9, "bogus": 1}')
        cfg = config.load_config()
        self.assertEqual(cfg["ram_reserve_gb"], 9)
        self.assertNotIn("bogus", cfg)


if __name__ == "__main__":
    unittest.main()


class StopTests(unittest.TestCase):
    """A stop counts only once the model server is gone (under memory pressure the stop command can fail)."""

    def run_stop(self, statuses):
        import asyncio
        import types
        from repertory import service
        calls = {"stop": 0}
        host = types.SimpleNamespace(after_stop=lambda: None)
        def backend_stop():
            calls["stop"] += 1
            if calls["stop"] == 1:
                raise OSError("wsl.exe: insufficient resources")
            return "stopped"
        host.backend_stop = backend_stop
        host.backend_status = lambda: (statuses.pop(0), None)
        events = []
        be = service.Backend()
        be.set("running")
        be.pid, be.tier = 123, "full"
        svc = types.SimpleNamespace(observe=False, be=be, host=host, event=lambda k, m: events.append((k, m)),
                                    ext_at_start=None, perproc_at_start=None, freed=None,
                                    mon=types.SimpleNamespace(get=lambda: {"ram_avail": 20e9}),
                                    own_ram_estimate=lambda: 45e9)
        real_sleep = asyncio.sleep
        service.asyncio.sleep = lambda s: real_sleep(0)
        try:
            asyncio.run(service.Service._stop_locked(svc, "test"))
        finally:
            service.asyncio.sleep = real_sleep
        return be.state, events, calls["stop"]

    def test_a_failed_stop_is_retried(self):
        state, events, stops = self.run_stop(["stopped"])     # the first stop raises; the second works
        self.assertEqual(state, "stopped")
        self.assertEqual(stops, 2)

    def test_a_server_that_stays_is_still_tracked(self):
        state, events, stops = self.run_stop(["running", "running", "running"])
        self.assertEqual(state, "running")
        self.assertEqual(events[-1][0], "error")

    def test_a_host_that_doesnt_answer_is_an_error_not_running(self):
        state, events, stops = self.run_stop(["unknown", "unknown", "unknown"])     # WSL down: can't tell
        self.assertEqual(state, "error")
        self.assertEqual(events[-1][0], "error")
        self.assertIn("can't reach the model server's host", events[-1][1])


class AliveTests(unittest.TestCase):
    """A running model whose host stops answering (WSL down) is taken as lost after UNREACHABLE_S, not shown as
    running; one missed check isn't enough."""

    def svc(self, statuses):
        import types
        from repertory import service
        be = service.Backend()
        be.set("running")
        be.pid, be.tier = 123, "full"
        s = types.SimpleNamespace(be=be, task=None, unreachable_since=None, events=[], spawned=[], follower=None,
                                  host=types.SimpleNamespace(backend_status=lambda: statuses.pop(0), follower=None))
        s.event = lambda k, m: s.events.append((k, m))
        s.spawn = lambda coro: (s.spawned.append(coro), coro.close())
        s._recover = lambda *a: service.asyncio.sleep(0)
        s.check_alive = types.MethodType(service.Service.check_alive, s)
        return s

    def check(self, s, at):
        import asyncio
        from unittest import mock
        from repertory import service
        with mock.patch.object(service.time, "time", lambda: at):
            asyncio.run(s.check_alive())

    def test_a_minute_without_an_answer_loses_the_model(self):
        from repertory import service
        s = self.svc([("unknown", "E_FAIL")] * 3)
        self.check(s, 1000.0)
        self.check(s, 1000.0 + service.UNREACHABLE_S - 1)
        self.assertEqual((s.events, s.spawned), ([], []))                  # a while without an answer: wait
        self.check(s, 1000.0 + service.UNREACHABLE_S)
        self.assertEqual(len(s.spawned), 1)                                # then recover, and say why
        self.assertIn("can't reach the model server's host", s.be.error)
        self.assertIn("E_FAIL", s.events[-1][1])

    def test_an_answer_in_between_starts_the_count_again(self):
        from repertory import service
        s = self.svc([("unknown", "x"), ("running", 123), ("unknown", "x")])
        self.check(s, 1000.0)
        self.check(s, 1030.0)                                              # answered: the model is fine
        self.check(s, 1000.0 + service.UNREACHABLE_S + 5)                  # a fresh miss, not a minute of them
        self.assertEqual(s.spawned, [])
        self.assertEqual(s.unreachable_since, 1000.0 + service.UNREACHABLE_S + 5)


class RamTuningTests(unittest.TestCase):
    """How the RAM cache is sized: through WSL's slow hand-back after a stop, and after running short."""
    GIB = 2 ** 30

    def svc(self):
        import types
        from repertory import service
        s = types.SimpleNamespace(cfg=dict(config.DEFAULTS, quant="4.05", context=262144), learned={}, freed=None,
                                  ram_soft_cap=None, be=service.Backend(), events=[], ram_before_start=None,
                                  host=types.SimpleNamespace(kind="windows-wsl", gpu=None))
        s.event = lambda k, m: s.events.append((k, m))
        for name in ("ram_for_start", "soft_ram_cap", "cap_after_ram_pressure", "full_ram_gib", "own_ram_estimate",
                     "ram_overhead", "learned_key", "measure_ram_overhead"):
            setattr(s, name, types.MethodType(getattr(service.Service, name), s))
        s.gpu_name = "Test GPU"
        return s

    def plan(self):
        from engines.exl3_offload import sizing
        return sizing.plan("4.05", 28.5e9, 262144)

    def test_the_ram_cache_holds_90_percent_of_the_experts(self):
        from engines.exl3_offload import sizing
        p = self.plan()
        self.assertAlmostEqual(p["ram_target"], 0.9 * sizing.expert_bytes("4.05") - p["cache"], delta=1)
        s = self.svc()
        self.assertEqual(s.full_ram_gib({"ram_avail": 60e9}, p), int(-(-p["ram_target"] // self.GIB)))

    def test_a_restart_reuses_the_ram_wsl_has_not_handed_back_yet(self):
        import time
        s = self.svc()
        p = self.plan()
        short = s.full_ram_gib({"ram_avail": 12e9}, p)               # just after the stop, as Windows sees it
        s.freed = (time.time(), 9e9, 45e9)                           # 9 GB free while the server held 45 GB
        self.assertGreater(s.full_ram_gib({"ram_avail": 12e9}, p), short + 20)
        s.freed = (time.time() - 400, 9e9, 45e9)                     # too long ago: the figure may be stale
        self.assertEqual(s.full_ram_gib({"ram_avail": 12e9}, p), short)

    def test_running_short_of_ram_caps_the_next_full_start(self):
        s = self.svc()
        p = self.plan()
        s.be.tier, s.be.ram_gib = "full", 40
        s.cap_after_ram_pressure({"ram_avail": 4.5e9}, ["GPU busy: game.exe at 90%"])
        self.assertIsNone(s.ram_soft_cap)                            # not a RAM problem
        s.cap_after_ram_pressure({"ram_avail": 4.5e9}, ["RAM low (4.5 GB free)"])
        self.assertEqual(s.soft_ram_cap(), 40 - 4 - 2)               # 3.5 GB short of 8 GB free: 4 GiB, plus 2
        self.assertEqual(s.full_ram_gib({"ram_avail": 60e9}, p), 34)
        s.ram_soft_cap = (s.ram_soft_cap[0] - 4000, 34)              # an hour later it lapses
        self.assertIsNone(s.soft_ram_cap())

    def test_the_ram_overhead_estimate_rises_fast_and_falls_slowly(self):
        s = self.svc()
        s.be.ram_gib = 30
        key = "windows-wsl|4.05"
        for before, after, want in ((60e9, 60e9 - 30 * self.GIB - 4e9, 4e9), (60e9, 60e9 - 30 * self.GIB - 8e9, 8e9),
                                    (60e9, 60e9 - 30 * self.GIB - 3e9, 0.8 * 8e9 + 0.2 * 3e9)):
            s.ram_before_start = before
            s.measure_ram_overhead({"ram_avail": after})
            self.assertAlmostEqual(s.learned["_ram_overhead"][key], want, delta=1e6)
