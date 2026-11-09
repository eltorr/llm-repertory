"""The benchmark (repertory/bench.py) against a fake model server: runs, workloads, records and tables, no GPU."""
import asyncio
import http.server
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
import unittest.mock
from pathlib import Path

os.environ["REPERTORY_HOME"] = tempfile.mkdtemp(prefix="repertory-test-")      # never a real settings dir
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from engines.exl3_offload import sizing  # noqa: E402
from repertory import bench, config  # noqa: E402
from repertory.service import Service  # noqa: E402

bench.SETTLE_S = 0
bench.WATCH_S, bench.YIELD_AFTER_S = 0.02, 0.05
bench.WORKLOADS["resize"]["ram_timeline_s"] = (0,)       # no waiting for Windows to see the freed RAM
KEY = "test-key"


class FakeEngine(http.server.BaseHTTPRequestHandler):
    """Answers like the engine: the manager's key on every request, streamed answers with usage at the end."""
    bodies, slow = [], 0.0
    arena = 20e9                                         # the VRAM expert cache, which /expert_cache/resize changes
    ram_slots, ram_unsupported = 0, False                # the RAM cache (ram_bytes), and an engine without it
    SLOT = 1.8432e6
    # a grow runs in the background: done after this many polls of /expert_cache/ram (stop and a new resize cut it
    # short halfway); drift: a bug where the RAM guard's limit doesn't follow the cache; stall_s: one where a
    # request waits this long for its first token while the RAM cache grows
    grow_polls, polls, ram_target, ram_from, stopped, drift, limit0, growing = 2, 0, 0, 0, None, False, 0, False
    stall_s = 0.0
    pending, reused = 0, 0                               # a shrink's slots not freed yet: freed by the next request

    @classmethod
    def ram_state(cls):
        limit = cls.limit0 + (0 if cls.drift else cls.ram_slots * cls.SLOT)
        return {"growing": cls.growing, "freeing": cls.pending > 0, "ram_slots": cls.ram_slots,
                "pending_bytes": cls.pending * cls.SLOT, "reused_slots": cls.reused,
                "ram_bytes": cls.ram_slots * cls.SLOT, "target_slots": cls.ram_target,
                "target_bytes": cls.ram_target * cls.SLOT, "stopped": cls.stopped, "grow_s": 0.1, "limit_bytes": limit}

    @classmethod
    def cut_grow(cls):
        if cls.growing:                                  # as the engine: it keeps what it has, and its target
            cls.ram_slots = (cls.ram_from + cls.ram_target) // 2
            cls.stopped, cls.growing = "stopped", False

    def send_json(self, body, status=200):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass

    def send_events(self, events):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            for ev in events:
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.01 + self.slow)
            self.wfile.write(b"data: [DONE]\n\n")
        except ConnectionError:
            pass                                        # the client left after the first token (prompt reading)

    def keyed(self):
        if self.headers.get("X-Repertory-Key") != KEY:
            self.send_response(403)
            self.end_headers()
            return False
        return True

    def do_GET(self):
        if not self.keyed():
            return
        FakeEngine.pending = 0                           # any request between two switches: the free has finished
        if self.path == "/expert_cache/ram":
            FakeEngine.polls += 1
            if FakeEngine.growing and FakeEngine.polls >= FakeEngine.grow_polls:
                FakeEngine.ram_slots, FakeEngine.growing = FakeEngine.ram_target, False
            return self.send_json(FakeEngine.ram_state())
        if self.path == "/memory":
            st = FakeEngine.ram_state()
            return self.send_json({"ram": {"used": st["ram_bytes"] + 6e9, "peak": 0, "limit": st["limit_bytes"]},
                                   "vram": None})
        body = {"/v1/models": {"data": [{"id": "qwen3.8-flash-next"}]},
                "/stats": {"expert_cache": [{"slots": 1234}], "nv2": {"host": {"vram": 90, "ram": 9, "nvme": 1}}},
                "/nv_verify?n=64": {"vram_checked": 64, "vram_bad": 0, "ram_bad": 0, "table_bad": 0}}
        data = json.dumps(body.get(self.path, {})).encode()
        self.send_response(200 if self.path in body else 404)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if not self.keyed():
            return
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeEngine.bodies.append((self.path, req))
        if self.path == "/expert_cache/resize":
            if "ram_bytes" in req and FakeEngine.ram_unsupported:
                data = json.dumps({"error": "ram_bytes: the compiled kernels predate RAM resizing"}).encode()
                self.send_response(400)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            before = int(FakeEngine.arena // 1.8432e6)
            FakeEngine.arena = req["max_slots"] * 1.8432e6 if req["max_slots"] is not None else 20e9
            ram = None
            if "ram_bytes" in req:
                FakeEngine.cut_grow()                       # as the engine: a new RAM resize stops a grow first
                was, want = FakeEngine.ram_slots, int(req["ram_bytes"] // FakeEngine.SLOT)
                ram = {"ram_slots_before": was, "ram_bytes_before": was * FakeEngine.SLOT,
                       "dropped": max(0, was - want), "moved": 0, "ms": 5.0}
                FakeEngine.stopped = None
                FakeEngine.ram_target = want
                FakeEngine.reused = min(FakeEngine.pending, max(0, want - was))   # what a shrink has not freed yet
                FakeEngine.pending -= FakeEngine.reused
                FakeEngine.ram_slots += FakeEngine.reused
                if want > FakeEngine.ram_slots:             # the rest grows in the background
                    FakeEngine.ram_from, FakeEngine.polls, FakeEngine.growing = FakeEngine.ram_slots, 0, True
                else:
                    FakeEngine.pending += max(0, FakeEngine.ram_slots - want)
                    FakeEngine.ram_slots = want
                ram.update(FakeEngine.ram_state())
            data = json.dumps({"pools": [{"slots_before": before, "slots": int(FakeEngine.arena // 1.8432e6)}],
                               "chunk": req["chunk"], "ram": ram, "seconds": 0.01}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/expert_cache/ram/stop":
            FakeEngine.cut_grow()
            self.send_json(FakeEngine.ram_state())
        elif self.path == "/tokenize":                      # a token for every 4 characters
            data = json.dumps({"tokens": [0] * (len(req["prompt"]) // 4)}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            prompt = req["messages"][0]["content"]
            FakeEngine.pending = 0
            if FakeEngine.growing and FakeEngine.stall_s:
                time.sleep(FakeEngine.stall_s)
            reply = " ".join(c for _, _, c in bench.LOCKERS) if "locker" in prompt else "Heat moves uphill."
            self.send_events([{"choices": [{"delta": {"content": w + " "}}]} for w in reply.split()]
                             + [{"choices": [{"delta": {}}], "usage": {"prompt_tokens": len(prompt) // 4,
                                                                       "completion_tokens": 40}}])


class FakeHost:
    kind, follower, ram_settle_s = "linux", None, 0

    def __init__(self):
        self.envs, self.state, self.dropped = [], "stopped", []

    def backend_start(self, env, args):
        self.envs.append(env)
        FakeEngine.ram_slots = FakeEngine.ram_target = int(int(env["RAM_GB"]) * 1024 ** 3 // 1.8432e6)
        FakeEngine.limit0, FakeEngine.stopped = 18e9, None
        self.state = "running"
        return True, 4242

    def backend_stop(self, timeout=30):
        self.state = "stopped"
        return "stopped"

    def backend_status(self):
        return self.state, (4242 if self.state == "running" else None)

    def after_ready(self):
        self.dropped.append("ready")

    def after_stop(self):
        self.dropped.append("stop")


class FakeService:
    """What the benchmark uses from the manager."""
    observe, gpu_display = False, True

    def __init__(self, port):
        self.cfg = dict(config.DEFAULTS, backend_port=port, quant="3.05", context=65536, start_timeout_s=20)
        self.host, self.learned = FakeHost(), {}
        self.setup = types.SimpleNamespace(running=None, check=self._check, hw={
            "gpus": [{"name": "Test GPU", "pcie_gen_max": 5, "pcie_width_max": 16}], "ram": {"total_gb": 68.7},
            "engine": {"driver": "1", "m305": "1", "m405": "0", "torch": "2.15.0+cu132", "cuda_version": "13.2"},
            "driver": {"version": "617.14"}, "repo": {"head": "abcdef0123"}})
        self.backend_key, self.backend_auth = KEY, {"X-Repertory-Key": KEY}
        self.be = types.SimpleNamespace(state="running", tier="full")
        self.game = False                                   # another app has the GPU
        self.mon = types.SimpleNamespace(get=self.sample)
        self.started, self.stopped = [], []
        self.last_stop_t, self.ram_avail = 0.0, 50e9
        self.server_ram, self.ram_back_at = 0.0, []          # RAM a run's server takes; when runs waited for RAM

    def sample(self):
        busy = self.game() if callable(self.game) else self.game
        used = 2e9 + (FakeEngine.arena + 6e9 if self.host.state == "running" else 0)
        ram = self.ram_avail - (self.server_ram if self.host.state == "running" else 0)
        return {"t": time.time(), "vram_total": 32e9, "vram_used": used, "ram_avail": ram,
                "perproc_ded": 1e9, "running": {"game.exe"} if busy else set(), "ext_util": 90.0 if busy else 0.0,
                "ext_util_top": [["game.exe", 90]] if busy else []}

    def ext_vram(self, m):
        return 2e9

    def ram_for_start(self, m):
        return m["ram_avail"]

    async def ram_back(self, m, cancelled=None):
        self.ram_back_at.append((self.host.state, len(self.host.envs)))
        return m

    async def _check(self):
        pass

    async def fresh_sample(self, after):
        return self.mon.get()

    def headroom(self, m):
        return 3.2e9

    def lean_budget(self, m):
        return 16e9

    def full_budget(self, m):
        return 26.8e9

    def learned_key(self, quant=None, context=None):
        return f"gpu|{quant}|{context}"

    def ram_overhead(self, quant=None):
        return 9e9

    def plan_for(self, tier, m):
        return Service.plan_for(self, tier, m)

    def shift_request(self, tier, m, current=None, ram_gib=None):
        return Service.shift_request(self, tier, m, current, ram_gib)

    async def stop(self, reason, why):
        self.stopped.append(reason)
        self.be.state, self.be.tier = "stopped", None

    async def start(self, tier, reason):
        self.started.append(tier)
        self.be.state, self.be.tier = "running", tier
        return True


class BenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeEngine)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        FakeEngine.bodies.clear()
        FakeEngine.slow, FakeEngine.arena, FakeEngine.ram_unsupported = 0.0, 20e9, False
        FakeEngine.drift, FakeEngine.grow_polls, FakeEngine.growing, FakeEngine.stall_s = False, 2, False, 0.0
        FakeEngine.pending = FakeEngine.reused = 0
        self.svc = FakeService(self.server.server_address[1])

    def run_bench(self, runs, workload="quick", cancel_after=None):
        b = bench.Bench(self.svc)

        async def go():
            ok, msg = b.start(runs, workload)
            self.assertTrue(ok, msg)
            if cancel_after is not None:
                await asyncio.sleep(cancel_after)
                await b.cancel()
            await b.task
        asyncio.run(go())
        return b

    def test_a_run_measures_answers_and_prompt_reading_then_restores_the_model(self):
        b = self.run_bench(["current"])
        r = b.records[0]
        self.assertIsNone(r.get("error"), r.get("error"))
        self.assertEqual(r["label"], "3.05-card-64k")
        self.assertGreater(r["decode_tps_median"], 0)
        self.assertEqual(set(r["prefill"]), {"1024", "8192"})
        self.assertEqual(r["protocol"], bench.PROTOCOL)
        self.assertEqual(r["engine"]["slots"], 1234)
        self.assertEqual(r["engine"]["picks_vram_pct"], 90.0)
        self.assertEqual(r["stack"], {"torch": "2.15.0+cu132", "cuda": "13.2", "driver": "617.14", "commit": "abcdef0"})
        self.assertTrue((b.dir / "3.05-card-64k.json").is_file())
        self.assertEqual((self.svc.stopped, self.svc.started), (["benchmark"], ["full"]))
        env = self.svc.host.envs[0]
        self.assertEqual(env["REPERTORY_BACKEND_KEY"], KEY)
        self.assertEqual(int(env["QWEN_VRAM_LIMIT_BYTES"]), int(32e9 - 3.2e9))    # the whole card, as full runs
        self.assertEqual(self.svc.host.state, "stopped")                       # no benchmark server left behind

    def test_answers_are_greedy_without_thinking_and_never_capped(self):
        self.run_bench(["current"], "standard")
        chats = [body for path, body in FakeEngine.bodies if path == "/v1/chat/completions"]
        self.assertEqual(chats[0]["messages"][0]["content"], bench.WARMUP[0])
        for body in chats:
            self.assertNotIn("max_tokens", body)
            self.assertEqual(body["temperature"], 0)                            # greedy
            self.assertIs(body["chat_template_kwargs"]["enable_thinking"], False)
        docs = [b["messages"][0]["content"] for b in chats if "-8192: " in b["messages"][0]["content"][:20]]
        self.assertEqual(len(docs), 3)                                          # one per trial
        self.assertTrue(docs[0].startswith(bench.DOC_HEAD.format(t=0, size=8192) + "\nEntry 000001: "))
        self.assertTrue(docs[0].endswith(" hPa.\n" + bench.DOC_TAIL))
        self.assertNotEqual(docs[0][:10], docs[1][:10])                         # nothing from the prefix cache
        self.assertAlmostEqual(len(docs[0]) // 4, 8192 - 48, delta=40)          # sized in tokens

    def test_the_standard_workload_finds_the_codes_in_the_long_document(self):
        r = self.run_bench(["current"], "standard").records[0]
        self.assertIsNone(r.get("error"), r.get("error"))
        self.assertTrue(r["long"]["codes_found"])
        self.assertAlmostEqual(r["long"]["prompt_tokens"], 65536 - 2048 - 48, delta=60)  # sized to the context
        self.assertEqual(len(r["decode_runs"]), 3)
        self.assertEqual(len(r["prefill"]["8192"]["runs"]), 3)
        self.assertEqual(set(r["prefill"]), {"1024", "8192", "32768"})          # 64K doesn't fit in a 64K context

    def test_a_first_token_faster_than_a_millisecond_still_counts(self):
        import unittest.mock
        clock = (100.0 + i * 1e-4 for i in range(1000))                 # 0.1 ms between readings
        with unittest.mock.patch.object(bench.time, "perf_counter", lambda: next(clock)):
            client = bench.Client(self.server.server_address[1], KEY)
            a = client.answer(bench.ANSWER.format(t=0))
        self.assertEqual(a["ttft_s"], 0.0)
        self.assertGreater(a["prefill_tps"], 0)                                 # not dropped as "no measurement"

    def test_an_emulated_card_gets_its_limit_guard_and_settings(self):
        self.run_bench(["3.05:10.5:131072:nostage:ram=12"])
        env = self.svc.host.envs[0]
        self.assertEqual(int(env["QWEN_VRAM_LIMIT_BYTES"]), int(2e9 + 10.5e9))
        self.assertEqual(int(env["QWEN_VRAM_GUARD_BYTES"]), int(2e9 + 10.5e9 + 0.25e9))
        self.assertEqual((env["QWEN_CHUNK"], env["RAM_GB"], env["CONTEXT"]), ("1536", "12", "131072"))
        self.assertEqual((env["MODEL_DIR"], env["STORE"]), ("./model", "./store/qwen.bin"))     # from the catalog
        for k in ("QWEN_MTP", "QWEN_CPU_LANE", "GLM53_NV_PREFETCH"):
            self.assertNotIn(k, env)                                  # engine features stay off unless named

    def test_a_glm_run_starts_glm_from_its_catalog_entry(self):
        self.svc.setup.hw["engine"]["mglm305"] = "0"
        b = self.run_bench(["glm-3.05:card:131072"])
        self.assertIn("glm-3.05 bpw model isn't downloaded", b.records[0]["error"])        # its own setup step
        self.assertEqual(self.svc.host.envs, [])
        self.svc.setup.hw["engine"]["mglm305"] = "1"
        b = self.run_bench(["glm-3.05:22:65536", "glm-3.05:16:65536:nostage:ram=24"])
        full, small = self.svc.host.envs
        self.assertEqual((full["QUANT"], full["MODEL_DIR"], full["STORE"], full["CONTEXT"]),
                         ("glm-3.05", "./model-glm-3.05", "./store/glm305.bin", "65536"))
        plan = b.records[0]["plan"]
        self.assertTrue(plan["ok"] and plan["staging"], plan)
        self.assertEqual(full["QWEN_CHUNK"], str(plan["chunk"]))
        # the engine may lend prefill its two staging buffers (5.4 GB) and the chunk's activations
        self.assertGreater(float(full["GLM53_EC_ELASTIC_GB"]), 6.0)
        self.assertEqual((small["QWEN_CHUNK"], small["RAM_GB"], small["GLM53_EC_STAGE_GB"]), ("1792", "24", "0"))

    def test_engine_features_reach_the_engine(self):
        self.run_bench(["3.05:card:65536:mtp=2:cpu=2-15:prefetch"])
        env = self.svc.host.envs[0]
        self.assertEqual((env["QWEN_MTP"], env["QWEN_CPU_LANE"], env["GLM53_NV_CPU_CPUS"], env["GLM53_NV_PREFETCH"]),
                         ("2", "1", "2-15", "1"))

    def test_the_resize_workload_switches_in_place_and_back(self):
        b = self.run_bench(["current"], workload="resize")
        r = b.records[0]
        self.assertIsNone(r.get("error"), r.get("error"))
        c = r["resize"]["checks"]
        self.assertTrue(c["same_answers"] and c["caches_consistent"] and c["deterministic"])
        lean = sizing.plan("3.05", 16e9, 65536)
        self.assertAlmostEqual(c["vram_freed_gb"], (20e9 - lean["slots"] * 1.8432e6) / 1e9, places=1)
        self.assertEqual(c["regrown_vs_first_gb"], [0.0] * (4 + 1 + bench.WORKLOADS["resize"]["ram_cycles"]))
        wl = bench.WORKLOADS["resize"]
        sent = [body for path, body in FakeEngine.bodies if path == "/expert_cache/resize"]
        vram_only = [b for b in sent if "ram_bytes" not in b]
        self.assertEqual(len(vram_only), 2 + 2 * wl["cycles"])
        self.assertEqual((sent[0]["max_slots"], sent[0]["chunk"]), (lean["slots"], lean["chunk"]))
        self.assertEqual(sent[0]["vram_limit_bytes"], int(2e9 + 16e9))          # other apps' VRAM + lean's budget
        self.assertIsNone(sent[1]["max_slots"])                                  # full: the engine sizes it
        self.assertEqual(sent[1]["vram_limit_bytes"], int(32e9 - 3.2e9))
        self.assertEqual([p["at"] for p in r["resize"]["phases"]],
                         ["full", "lean", "full again", "full after cycles", "lean with lean's RAM cache",
                          "full, RAM cache regrown", "full after RAM cycles"])
        docs = [body["messages"][0]["content"][:40] for path, body in FakeEngine.bodies
                if path == "/v1/chat/completions" and body["messages"][0]["content"].startswith("Log ")]
        self.assertEqual(len(set(docs)), 7)                 # its own at each size: nothing from the prompt cache
        # then the RAM cache: lean's, the run's again, and cycles
        ram = [b["ram_bytes"] for b in sent if "ram_bytes" in b]
        full_ram, lean_ram = r["ram_gib"] * 1024 ** 3, self.svc.cfg["lean_ram_gb"] * 1024 ** 3
        self.assertEqual(ram, [lean_ram, full_ram] * (1 + wl["ram_cycles"]))
        self.assertTrue(c["ram_slots_regrown"])
        self.assertAlmostEqual(c["ram_shrunk_gib"], r["ram_gib"] - self.svc.cfg["lean_ram_gb"], delta=0.01)
        self.assertEqual(len(r["resize"]["shifts"][2 + 2 * wl["cycles"]]["ram_timeline"]), len(wl["ram_timeline_s"]))
        self.assertEqual((self.svc.stopped, self.svc.started), (["benchmark"], ["full"]))

    def test_the_resize_workload_waits_for_the_background_grow_and_answers_meanwhile(self):
        r = self.run_bench(["current"], workload="resize").records[0]
        c, ram = r["resize"]["checks"], r["resize"]["ram"]
        self.assertTrue(c["ram_slots_regrown"] and c["same_answers"])
        self.assertEqual(len(ram["during_growth"]), 1)                     # an answer while it grew
        grows = [s for s in r["resize"]["shifts"] if (s["reply"].get("ram") or {}).get("growing")]
        self.assertEqual(len(grows), 1 + bench.WORKLOADS["resize"]["ram_cycles"])
        self.assertTrue(all(not s["ram_after"]["growing"] for s in grows))

    def test_the_stress_workload_runs_every_variation(self):
        r = self.run_bench(["current"], workload="resize-stress").records[0]
        self.assertIsNone(r.get("error"), r.get("error"))
        st = r["stress"]
        self.assertTrue(st["checks"]["all_ok"], st["checks"])
        self.assertEqual(len(st["variations"]), 5)
        names = " / ".join(v["name"] for v in st["variations"])
        for what in ("while the RAM cache grows", "a shrink while", "stopped partway", "quick switches", "around",
                     "while its memory goes back"):
            self.assertIn(what, names)
        v1, v4, v5 = st["variations"][0], st["variations"][3], st["variations"][4]
        self.assertGreater(v4["reused_slots"][0], 0)                       # back to full took memory back
        self.assertTrue(v1["still_growing_after_answer"])                  # the answer ran while it grew
        self.assertLess(v1["ttft_while_growing_s"], 1.0)
        self.assertEqual(len(v5["answers"]), 4)                             # running, arriving, while freeing, last
        v3 = st["variations"][2]
        self.assertTrue(v3["partial"] and v3["stop_reason"] == "stopped")
        self.assertEqual(v3["ram"]["ram_slots"], v3["target"])             # then grown the rest of the way
        self.assertIn(("/expert_cache/ram/stop", {}), [(p, b) for p, b in FakeEngine.bodies])
        self.assertEqual(FakeEngine.ram_slots, FakeEngine.ram_target)        # left full, its RAM cache grown
        self.assertEqual((self.svc.stopped, self.svc.started), (["benchmark"], ["full"]))

    def test_the_stress_workload_catches_limits_that_dont_follow(self):
        FakeEngine.drift = True
        r = self.run_bench(["current"], workload="resize-stress").records[0]
        self.assertIn("stress check failed", r["error"])
        self.assertFalse(r["stress"]["checks"]["all_ok"])
        self.assertTrue(any(not v["limits_follow"] for v in r["stress"]["variations"]))

    def test_the_stress_workload_catches_a_grow_that_holds_up_requests(self):
        FakeEngine.stall_s, FakeEngine.grow_polls = 0.4, 3
        with unittest.mock.patch.dict(bench.WORKLOADS["resize-stress"], {"latency_max_s": 0.3}):
            r = self.run_bench(["current"], workload="resize-stress").records[0]
        self.assertIn("stress check failed", r["error"])
        v1 = r["stress"]["variations"][0]
        self.assertFalse(v1["ok"])
        self.assertGreater(v1["ttft_while_growing_s"], 0.3)
        self.assertTrue(v1["same_answers"] and v1["cache_ok"] and v1["limits_follow"])   # only the wait failed it

    def test_an_engine_that_cant_resize_its_ram_cache_is_recorded_not_failed(self):
        FakeEngine.ram_unsupported = True
        r = self.run_bench(["current"], workload="resize").records[0]
        self.assertIsNone(r.get("error"), r.get("error"))
        self.assertIn("predate", r["resize"]["ram"]["unsupported"])
        self.assertEqual(len(r["resize"]["phases"]), 4)
        self.assertNotIn("ram_slots_regrown", r["resize"]["checks"])

    def test_the_resize_workload_needs_a_whole_card_run(self):
        r = self.run_bench(["3.05:16:65536"], workload="resize").records[0]
        self.assertIn("whole-card run", r["error"])

    def test_a_model_that_isnt_downloaded_is_skipped(self):
        r = self.run_bench(["4.05:card:65536"]).records[0]
        self.assertIn("isn't downloaded", r["error"])
        self.assertEqual(self.svc.host.envs, [])

    def test_a_ram_cache_the_pc_cant_spare_is_never_started(self):
        # 50 GB free, less the 9 GB kept free and the server's own 9 GB: 29 GiB to spare
        r = self.run_bench(["3.05:card:65536:ram=30"]).records[0]
        self.assertEqual(r["error"], "not enough free RAM for a 30 GiB RAM cache: the PC can spare 29 GiB now")
        self.assertTrue(r["skipped"])                                   # not run: curation keeps the earlier result
        self.assertEqual(self.svc.host.envs, [])

    def test_a_set_ram_cache_waits_for_wsl_to_hand_the_last_servers_ram_back(self):
        self.svc.host.ram_settle_s = 30
        self.svc.last_stop_t, self.svc.ram_avail = time.time(), 30e9         # just stopped: WSL still holds it
        threading.Timer(1.0, setattr, (self.svc, "ram_avail", 50e9)).start()
        r = self.run_bench(["3.05:card:65536:ram=20"]).records[0]
        self.assertIsNone(r.get("error"), r.get("error"))
        self.assertEqual(self.svc.host.envs[0]["RAM_GB"], "20")

    def test_each_run_waits_for_the_last_servers_ram_and_records_what_its_own_held(self):
        self.svc.server_ram = 20e9
        self.run_bench(["current", "current"])
        self.assertEqual(self.svc.ram_back_at, [("stopped", 0), ("stopped", 1)])     # before each start
        t, free, held = self.svc.freed
        self.assertEqual((free, held), (30e9, 20e9))       # free with the server up, and what it took

    def test_page_cache_is_dropped_once_a_server_is_up_and_after_it_stops(self):
        r = self.run_bench(["current"]).records[0]
        self.assertEqual(self.svc.host.dropped, ["ready", "stop"])
        self.assertEqual(r["ram_free_min_gb"], 50.0)                     # the least free RAM while it ran

    def test_the_pc_needs_ram_back_where_the_manager_would_pause(self):
        m = dict(self.svc.sample(), ram_avail=2.5e9)                     # ram_pause_gb: 3
        self.assertEqual(bench.Bench(self.svc).disturbed(m), ["RAM nearly out (2.5 GB free)"])

    def test_a_second_run_is_compared_with_the_first(self):
        self.run_bench(["current"])
        r = self.run_bench(["current"]).records[0]
        self.assertIsNotNone(r["previous"])
        self.assertIsInstance(r["previous"]["decode_pct"], float)

    def test_a_benchmark_started_while_the_last_one_starts_the_model_again_waits_for_it(self):
        log, start, host_start = [], self.svc.start, self.svc.host.backend_start

        async def slow_start(tier, reason):
            log.append("model starting")
            await asyncio.sleep(0.5)                       # as the manager waits for WSL to hand RAM back
            log.append("model started")
            return await start(tier, reason)
        self.svc.start = slow_start
        self.svc.host.backend_start = lambda env, args: (log.append("run's server"), host_start(env, args))[1]
        b = bench.Bench(self.svc)

        async def go():
            self.assertTrue(b.start(["current"], "quick")[0])
            while b.running:
                await asyncio.sleep(0.01)
            first = b.task
            ok, msg = b.start(["current"], "quick")        # the first is still starting the model again
            self.assertTrue(ok, msg)
            self.assertFalse(first.done())
            await b.task
        asyncio.run(go())
        self.assertEqual(log, ["run's server", "model starting", "model started"] * 2)
        self.assertEqual(self.svc.stopped, ["benchmark", "benchmark"])   # the second stopped the model it started
        self.assertIsNone(b.records[0].get("error"), b.records[0].get("error"))

    def test_a_run_whose_server_was_replaced_on_its_port_is_an_error(self):
        self.svc.host.backend_status = lambda: ("running", 999)          # not the pid the run's launch gave
        r = self.run_bench(["current"]).records[0]
        self.assertEqual(r["error"], bench.REPLACED)
        self.assertNotIn("decode_tps_median", r)

    def test_a_server_replaced_during_the_workload_spoils_the_run(self):
        calls = []
        self.svc.host.backend_status = lambda: (calls.append(1), ("running", 4242 if len(calls) == 1 else 999))[1]
        r = self.run_bench(["current"]).records[0]
        self.assertEqual(r["error"], bench.REPLACED)

    def test_cancel(self):
        FakeEngine.slow = 0.2
        b = self.run_bench(["current", "3.05:22:65536"], cancel_after=0.5)
        self.assertEqual([r.get("error") for r in b.records], ["cancelled"])
        self.assertEqual(self.svc.started, ["full"])

    def test_a_run_steps_aside_for_the_pc_and_starts_again_once_it_is_free(self):
        FakeEngine.slow = 0.1
        self.svc.cfg["recover_after_s"] = 0.2
        b = bench.Bench(self.svc)

        async def go():
            ok, msg = b.start(["current"], "quick")
            self.assertTrue(ok, msg)
            await asyncio.sleep(0.3)                        # mid-run, a game takes the GPU
            self.svc.game = True
            for _ in range(200):
                if b.waiting:
                    break
                await asyncio.sleep(0.02)
            self.assertEqual(b.waiting, ["GPU busy: game.exe at 90%"])
            await asyncio.sleep(0.3)                        # still waiting while the game runs
            self.assertEqual(len(self.svc.host.envs), 1)
            self.svc.game = False
            await b.task
        asyncio.run(go())
        lines = [x for _, x in b.out]
        self.assertIn("  stepping aside: GPU busy: game.exe at 90%", lines)
        self.assertIn("  the PC is free again", lines)
        self.assertEqual(len(self.svc.host.envs), 2)                            # the run started again
        r = b.records[0]
        self.assertIsNone(r.get("error"), r.get("error"))
        self.assertTrue(bench.counts(r))
        self.assertEqual(self.svc.started, ["full"])
        self.assertEqual(self.svc.host.state, "stopped")

    def test_a_watched_app_counts_and_a_run_interrupted_too_often_is_recorded(self):
        FakeEngine.slow = 0.1
        self.svc.cfg.update(recover_after_s=0, watchlist=["Game.exe"], ext_util_pct=101)
        self.svc.game = lambda: len(self.svc.host.envs) > 0 and self.svc.host.state == "running"   # every start
        b = self.run_bench(["current"])
        self.assertEqual(len(self.svc.host.envs), bench.MAX_YIELDS + 1)
        r = b.records[0]
        self.assertEqual(r["error"], "interrupted: watched app running: game.exe")
        self.assertTrue(r["interrupted"])
        self.assertFalse(bench.counts(r))                                       # resuming redoes it
        self.assertEqual(bench.unfinished()[0], b.dir)

    def test_a_benchmark_cut_short_resumes_where_it_stopped(self):
        FakeEngine.slow = 0.2
        first = self.run_bench(["current", "3.05:22:65536"], cancel_after=0.5)
        self.assertEqual([r.get("error") for r in first.records], ["cancelled"])
        self.assertEqual([r["label"] for r in bench.unfinished()[2]], ["3.05-card-64k", "3.05-22gb-64k"])
        FakeEngine.slow = 0.0
        b = bench.Bench(self.svc)

        async def go():
            ok, msg = b.start([], "standard", resume=True)
            self.assertTrue(ok, msg)
            self.assertIn("2 runs left, quick workload", msg)               # the benchmark's own workload
            await b.task
        asyncio.run(go())
        self.assertEqual(b.dir, first.dir)
        records = bench.load_records(b.dir)                                     # _plan.json isn't a result
        self.assertEqual(sorted(r["label"] for r in records), ["3.05-22gb-64k", "3.05-card-64k"])
        self.assertTrue(all(bench.counts(r) and not r.get("error") for r in records))
        self.assertIsNone(bench.unfinished())
        self.assertEqual(b.start([], resume=True), (False, "no unfinished benchmark to resume"))

    def test_requests_are_refused_while_it_runs(self):
        b = bench.Bench(self.svc)
        b.running = True
        self.assertEqual(b.start(["current"], "quick"), (False, "a benchmark is already running"))


class RunSpecTests(unittest.TestCase):
    cfg = dict(config.DEFAULTS, quant="4.05", context=262144)

    def test_presets(self):
        runs = bench.expand(["current", "quants", "lean"], self.cfg, lean_gb=16.0)
        self.assertEqual([r["label"] for r in runs],
                         ["4.05-card-256k", "3.05-card-256k", "4.05-card-256k", "4.05-16gb-256k-ram8"])
        matrix = bench.expand("matrix", self.cfg)
        self.assertEqual(len(matrix), len(bench.MATRIX))
        self.assertEqual({r["ram"] for r in matrix}, {38})                      # runs differ only in VRAM

    def test_glm_runs(self):
        r = bench.parse_run("glm-3.05:card:131072")
        self.assertEqual((r["quant"], r["vram_gb"], r["context"], r["label"], r["spec"]),
                         ("glm-3.05", None, 131072, "glm-3.05-card-128k", "glm-3.05:card:131072"))
        r = bench.parse_run("glm-3.05:24:65536:nostage:ram=40")
        self.assertEqual((r["vram_gb"], r["nostage"], r["ram"], r["label"]),
                         (24.0, True, 40, "glm-3.05-24gb-64k-nostage-ram40"))
        # "quants" is the configured model's own family: GLM's two, Qwen's two
        glm = dict(self.cfg, quant="glm-3.05", context=131072)
        self.assertEqual([x["label"] for x in bench.expand(["current", "quants"], glm)],
                         ["glm-3.05-card-128k", "glm-3.05-card-128k", "glm-2.05-card-128k"])
        self.assertEqual(bench.family_quants("4.05"), ["3.05", "4.05"])
        with self.assertRaises(ValueError):
            bench.parse_run("glm-4.05:card:65536")
        # the matrix stays Qwen's
        self.assertEqual({x["quant"] for x in bench.expand("matrix", glm)}, {"3.05", "4.05"})
        self.assertEqual((bench.nostage_chunk("3.05"), bench.nostage_chunk("glm-3.05"), bench.nostage_chunk("x")),
                         (1536, 1792, 1536))

    def test_engine_features(self):
        r = bench.parse_run("3.05:card:262144:ram=26:mtp=2:cpu=2-15:prefetch")
        self.assertEqual((r["mtp"], r["cpu"], r["prefetch"]), (2, "2-15", True))
        self.assertEqual(r["label"], "3.05-card-256k-ram26-mtp2-cpu2-15-prefetch")
        r = bench.parse_run("3.05:card:262144:cpu")
        self.assertEqual((r["mtp"], r["cpu"], r["prefetch"], r["label"]), (0, "", False, "3.05-card-256k-cpu"))
        plain = bench.parse_run("4.05:card:262144")
        self.assertEqual((plain["mtp"], plain["cpu"], plain["prefetch"]), (0, None, False))

    def test_bad_specs(self):
        for bad in ("5.0:card:65536", "3.05:2:65536", "3.05:card:999999999", "3.05:card:64k", "3.05:nan:65536",
                    "3.05:card:65536:turbo", "3.05:card:65536:ram=0", "3.05:card", "", "3.05:card:65536:mtp=0",
                    "3.05:card:65536:mtp=9", "3.05:card:65536:cpu=all", "3.05:card:65536:cpu=2-15;rm"):
            with self.assertRaises(ValueError, msg=bad):
                bench.expand([bad], self.cfg)
        with self.assertRaises(ValueError):
            bench.expand(["matrix"] * 3 + ["current"] * 5, self.cfg)        # too many runs


class TableTests(unittest.TestCase):
    def test_results_from_before_this_tool_still_render(self):
        records = bench.load_records(REPO / "research" / "vram-scaling" / "results")
        rows = bench.table(records).splitlines()
        self.assertEqual(len(rows), 2 + len(records))
        self.assertIn("**fails**: ran out of VRAM", "\n".join(rows))
        self.assertEqual(len(bench.measured_rows(records, "research/x").splitlines()), len(records) - 1)


class ApiTests(unittest.TestCase):
    def test_other_devices_cannot_start_or_cancel_a_benchmark(self):
        class Writer:
            data = b""

            def write(self, b):
                Writer.data += b

            async def drain(self):
                pass
        svc = types.SimpleNamespace()
        for route in ("bench", "bench/cancel"):
            Writer.data = b""
            path = "/_switcher/api/" + route
            asyncio.run(Service.api(svc, "POST", path, path, b'{"runs": ["current"]}', Writer(), local=False))
            self.assertTrue(Writer.data.startswith(b"HTTP/1.1 403"), Writer.data[:40])


if __name__ == "__main__":
    unittest.main()
