"""Curation (repertory/curate.py): the compatibility checks against fakes for Hugging Face, the hardware and the
engine's files; the regression check through the benchmark (bench.py's "reference" workload) against a fake engine,
recording a reference and comparing with it token by token; the comparison's figures on made-up distributions; and
the result written into the catalog."""
import asyncio
import copy
import hashlib
import http.server
import io
import json
import math
import os
import random
import re
import sys
import tempfile
import threading
import time
import types
import unittest
import zlib
from pathlib import Path
from unittest import mock

os.environ["REPERTORY_HOME"] = tempfile.mkdtemp(prefix="repertory-test-")      # never a real settings dir
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from repertory import bench, catalog, config, curate  # noqa: E402

bench.SETTLE_S = 0
bench.WATCH_S, bench.YIELD_AFTER_S = 0.02, 0.05
KEY = "test-key"
GIB = 1024 ** 3
DATA = json.loads(catalog.CATALOG.read_text(encoding="utf-8"))
M305, M405 = "qwen3.8-flash-next-exl3-3.05", "qwen3.8-flash-next-exl3-4.05"
REPO_ID = "turboderp/Qwen3.8-Flash-Next-exl3"
REV = {m["id"]: m["source"]["revision"] for m in DATA["models"]}


# --------------------------------------------------------------------------------------------------------- fakes
def blob_id(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def config_json(bits=3.05, archs=("Qwen4ExpForConditionalGeneration",), experts=512):
    return json.dumps({"architectures": list(archs), "model_type": "qwen4_exp",
                       "text_config": {"num_experts": experts, "num_experts_per_tok": 10},
                       "quantization_config": {"quant_method": "exl3", "bits": bits}}).encode()


class FakeHub:
    """Hugging Face: each model's pinned revision with its files (config.json and a weights file), and its branch."""

    def __init__(self, offline=False):
        self.offline, self.asked = offline, []
        self.files = {}
        self.revisions = {}
        for m in DATA["models"]:
            cfg, weights = config_json(m["bpw"]), b"weights of " + m["id"].encode()
            files = {"config.json": cfg, "model.safetensors": weights}
            self.files.update({(REV[m["id"]], n): b for n, b in files.items()})
            info = {"sha": REV[m["id"]], "siblings": [
                {"rfilename": "config.json", "blobId": blob_id(cfg)},
                {"rfilename": "model.safetensors", "blobId": "x",
                 "lfs": {"sha256": hashlib.sha256(weights).hexdigest()}}]}
            self.revisions[REV[m["id"]]] = info
            self.revisions[m["source"]["branch"]] = info

    def revision(self, repo, rev):
        self.asked.append(("revision", repo, rev))
        if self.offline:
            raise OSError("no network")
        return self.revisions.get(rev) if repo == REPO_ID else None

    def file(self, repo, rev, name):
        self.asked.append(("file", repo, rev, name))
        if self.offline:
            raise OSError("no network")
        return self.files.get((rev, name))


class Files(curate.LocalFiles):
    def commit(self):
        return "0123456789abcdef0123456789abcdef01234567"


def rtx5090(ram_gib=64, platform="windows-wsl", **gpu):
    return curate.Machine(platform, dict({"name": "NVIDIA GeForce RTX 5090", "vram_total": 31.84 * GIB,
                                          "compute_capability": "12.0", "display_active": None}, **gpu),
                          "617.14", ram_gib * GIB * 0.98, "Windows 11")


def download(root, model, hub, verified=True):
    """The model as download.py and the pack step leave it: its files, .verified.json and the expert store."""
    folder = Path(root) / model["paths"]["model_dir"]
    folder.mkdir(parents=True)
    rec = {}
    for s in hub.revisions[REV[model["id"]]]["siblings"]:
        (folder / s["rfilename"]).write_bytes(hub.files[(REV[model["id"]], s["rfilename"])])
        rec[s["rfilename"]] = (s.get("lfs") or {}).get("sha256") or s["blobId"]
    if verified:
        (folder / ".verified.json").write_text(json.dumps(rec))
    store = Path(root) / model["paths"]["store"]
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_bytes(b"store")
    store.with_suffix(".json").write_text("{}")
    return folder


def by_name(result):
    return {i["name"]: i for i in result["items"]}


# ------------------------------------------------------------------------------------------------- compatibility
class CompatTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="repertory-engine-")
        self.files, self.hub = Files(self.root), FakeHub()
        self.cfg = dict(config.DEFAULTS)

    def check(self, model_id=M405, machine=None, data=None, hub=None):
        return curate.compat(data or DATA, model_id, machine or rtx5090(), self.files, hub or self.hub, self.cfg)

    def test_an_entry_that_is_not_downloaded_yet(self):
        r = self.check()
        self.assertTrue(r["ok"], r)
        items = by_name(r)
        self.assertEqual(list(items), ["entry", "engine", "pin", "architecture", "fits", "files"])
        self.assertIn("from Hugging Face at the pinned revision", items["architecture"]["detail"])
        self.assertIn(("file", REPO_ID, REV[M405], "config.json"), self.hub.asked)          # at the pin
        self.assertIn("not downloaded", items["files"]["detail"])
        self.assertEqual(r["numbers"]["context"], 262144)
        self.assertEqual(r["machine"], {"gpu": "NVIDIA GeForce RTX 5090", "vram_gb": 34.2, "ram_gb": 67.3,
                                        "platform": "windows-wsl", "os": "Windows 11"})
        self.assertEqual(r["engine_commit"], Files(self.root).commit())

    def test_a_downloaded_entry_reads_its_own_config_and_trusts_downloads_record(self):
        folder = download(self.root, catalog.load().model(M405), self.hub)
        nope = AssertionError("hashed again")
        with mock.patch.object(hashlib, "sha256", side_effect=nope), \
                mock.patch.object(hashlib, "sha1", side_effect=nope):
            r = self.check()
        self.assertTrue(r["ok"], r)
        items = by_name(r)
        self.assertIn("from the downloaded folder", items["architecture"]["detail"])
        self.assertIn("2 files, each checked against its hash (the pinned revision's list)", items["files"]["detail"])
        (folder / "notes.txt").write_text("mine")                       # a file download.py never checked
        (folder / "x.safetensors.part").write_bytes(b"")                # one still downloading
        verified = json.loads((folder / ".verified.json").read_text())
        verified["config.json"] = "0" * 40                              # checked against another upload
        (folder / ".verified.json").write_text(json.dumps(verified))
        (Path(self.root) / "store" / "qwen405.json").unlink()           # the store isn't packed
        d = by_name(self.check())["files"]
        self.assertFalse(d["ok"])
        for what in ("still downloading", "never checked", "notes.txt", "another revision", "isn't packed"):
            self.assertIn(what, d["detail"])

    def test_the_earlier_downloaders_state_files_are_not_model_files(self):
        folder = download(self.root, catalog.load().model(M405), self.hub)
        (folder / "config.json.http-state.json").write_text("[0]")      # QwenFlash's downloader left these
        d = by_name(self.check())["files"]
        self.assertTrue(d["ok"], d)

    def test_a_download_without_a_record_is_not_taken_as_checked(self):
        download(self.root, catalog.load().model(M305), self.hub, verified=False)
        d = by_name(self.check(M305))["files"]
        self.assertFalse(d["ok"])
        self.assertIn("2 file(s) never checked", d["detail"])
        self.assertIn("run the model's setup step", d["detail"])

    def test_a_pin_that_doesnt_resolve(self):
        data = copy.deepcopy(DATA)
        data["models"][1]["source"]["revision"] = "f" * 40
        r = self.check(data=data)
        self.assertFalse(r["ok"])
        self.assertFalse(by_name(r)["pin"]["ok"])
        self.assertIn("has no revision ffff", by_name(r)["pin"]["detail"])

    def test_a_branch_that_moved_on_is_reported_not_failed(self):
        self.hub.revisions["4.05bpw_h6_ng6"] = {"sha": "a" * 40}
        p = by_name(self.check())["pin"]
        self.assertTrue(p["ok"])
        self.assertIn("has moved on to aaaa", p["detail"])

    def test_offline_the_check_is_not_decided(self):
        r = self.check(hub=FakeHub(offline=True))
        self.assertIsNone(r["ok"])
        self.assertIsNone(by_name(r)["pin"]["ok"])
        self.assertIsNone(by_name(r)["architecture"]["ok"])
        self.assertIn("not decided: pin, architecture", r["summary"])
        download(self.root, catalog.load().model(M405), self.hub)       # downloaded: its own config.json answers
        r = self.check(hub=FakeHub(offline=True))
        self.assertTrue(by_name(r)["architecture"]["ok"])
        self.assertIn("the Hub's list unavailable", by_name(r)["files"]["detail"])

    def test_an_architecture_or_format_the_engine_doesnt_run(self):
        self.hub.files[(REV[M405], "config.json")] = config_json(4.05, archs=("LlamaForCausalLM",))
        a = by_name(self.check())["architecture"]
        self.assertFalse(a["ok"])
        self.assertIn("doesn't run LlamaForCausalLM", a["detail"])
        self.hub.files[(REV[M405], "config.json")] = config_json(3.05, experts=256)
        a = by_name(self.check())["architecture"]
        self.assertIn("3.05 bits, the entry 4.05 bpw", a["detail"])
        self.assertIn("num_experts is 256, the entry's experts_per_layer 512", a["detail"])

    def test_it_has_to_fit_this_pc(self):
        f = by_name(self.check(machine=rtx5090(vram_total=8 * GIB * 0.995)))["fits"]
        self.assertFalse(f["ok"])                                       # 8 GB with a desktop: too little VRAM
        f = by_name(self.check(machine=rtx5090(ram_gib=16)))["fits"]
        self.assertFalse(f["ok"])
        self.assertIn("not enough RAM", f["detail"])
        self.assertTrue(by_name(self.check(M305, machine=rtx5090(ram_gib=32, platform="linux")))["fits"]["ok"])

    def test_the_engine_must_run_on_this_platform_gpu_and_driver(self):
        e = by_name(self.check(machine=rtx5090(compute_capability="7.0")))["engine"]
        self.assertFalse(e["ok"])
        self.assertIn("compute capability 7.0 is below 7.5", e["detail"])
        m = rtx5090()
        m.driver = "581.42"
        self.assertIn("older than 595", by_name(self.check(machine=m))["engine"]["detail"])
        m = curate.Machine("macos", None, None, 64 * GIB)
        e = by_name(self.check(machine=m))["engine"]
        self.assertIn("not on macos", e["detail"])
        self.assertIn("no NVIDIA GPU", e["detail"])

    def test_an_invalid_entry_stops_there(self):
        data = copy.deepcopy(DATA)
        data["models"][1]["engines"] = ["vllm"]
        r = self.check(data=data)
        self.assertFalse(r["ok"])
        self.assertEqual([i["name"] for i in r["items"]], ["entry"])
        self.assertIn("engine vllm is not in the catalog", r["items"][0]["detail"])
        self.assertEqual(self.hub.asked, [])


class HubTests(unittest.TestCase):
    def test_urls_and_missing_revisions(self):
        asked = []

        class Reply:
            def __init__(self, body):
                self.body = body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return self.body

        def urlopen(req, timeout):
            asked.append(req.full_url)
            if "ffff" in req.full_url:
                raise curate.urllib.error.HTTPError(req.full_url, 404, "not found", {}, io.BytesIO(b"{}"))
            return Reply(b'{"sha": "x"}')
        with mock.patch.object(curate.urllib.request, "urlopen", urlopen):
            hub = curate.Hub()
            self.assertEqual(hub.revision(REPO_ID, REV[M305]), {"sha": "x"})
            self.assertIsNone(hub.revision(REPO_ID, "f" * 40))
            hub.file(REPO_ID, REV[M305], "config.json")
        self.assertEqual(asked, [f"https://huggingface.co/api/models/{REPO_ID}/revision/{REV[M305]}?blobs=true",
                                 f"https://huggingface.co/api/models/{REPO_ID}/revision/{'f' * 40}?blobs=true",
                                 f"https://huggingface.co/{REPO_ID}/resolve/{REV[M305]}/config.json"])


# ------------------------------------------------------------------------------------------------- the engine
EOS = 2


def tid(word):
    """The fake engine's tokenizer: a token a word."""
    return 100 + zlib.crc32(word.encode()) % 50000


class FakeEngine(http.server.BaseHTTPRequestHandler):
    """Answers like the engine, deterministically: each reference request gets its own answer (a tool call in two
    pieces, reasoning before the answer with thinking on), streamed a token at a time with usage at the end. The same
    answers as token ids (/prepare, /generate), and a model for teacher forcing that is sure of whatever token comes
    next."""
    token_s, changed, forgetful = 0.004, None, False       # seconds a token; a case whose answer changed; misses facts
    shift, template = 0.0, None                            # a damaged model: every logit moved; a new chat template
    prompts, seen = {}, []                                 # the prompts /prepare gave, by their ids; POSTs as sent

    def log_message(self, *a):
        pass

    def send_json(self, body, status=200):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.headers.get("X-Repertory-Key") != KEY:
            return self.send_json({}, 403)
        body = {"/v1/models": {"data": [{"id": "qwen3.8-flash-next"}]},
                "/stats": {"expert_cache": [{"slots": 8736}], "nv2": {"host": {"vram": 90, "ram": 9, "nvme": 1}}}}
        self.send_json(body.get(self.path, {}), 200 if self.path in body else 404)

    def answer(self, req):
        prompt = req["messages"][-1]["content"]
        thinking = (req.get("chat_template_kwargs") or {}).get("enable_thinking")
        if req.get("tools"):
            return None, [{"index": 0, "id": "call_1", "type": "function",
                           "function": {"name": "get_weather", "arguments": '{"city": '}},
                          {"index": 0, "function": {"arguments": '"Lisbon"}'}}], "tool_calls"
        if "backup generator" in prompt:
            text = ("I don't know." if FakeEngine.forgetful else
                    "The backup generator was serviced by technician Ilsa Varga, and the code word is PERIWINKLE.")
        elif "Fibonacci" in prompt:
            text = 'def fib(n):\n    """The n-th Fibonacci number."""\n    a, b = 0, 1\n    return a'
        elif thinking:
            return "221 is 13 times 17.", "No; 221 = 13 × 17.", "stop"
        else:
            text = "A heat pump moves heat from a cold place to a warm one with a refrigerant loop that it compresses."
        if FakeEngine.changed and FakeEngine.changed in prompt:
            text += " Also, it is efficient."
        return None, text, "stop"

    @staticmethod
    def prompt_ids(req):
        kw = req.get("chat_template_kwargs") or {}
        return ([tid(w) for w in req["messages"][-1]["content"].split()] + [11 if kw.get("enable_thinking") else 10]
                + ([12] if req.get("tools") else []) + ([13] if FakeEngine.template else []))

    def answer_ids(self, req):
        reasoning, out, _ = self.answer(req)
        words = (reasoning or "").split() + (["</think>"] if reasoning else [])
        words += ["<tool_call>", "get_weather", "Lisbon"] if isinstance(out, list) else out.split()
        return [tid(w) for w in words] + [EOS]

    @staticmethod
    def dist(token, t):
        """The model's logprobs at position t where `token` comes next, likeliest first: that token, then 30 others;
        a damaged model (shift) moves every logit by up to shift."""
        logits = {token: 6.0, **{60000 + k: 3.0 - 0.2 * k for k in range(30)}}       # tid() never gives these
        if FakeEngine.shift:
            logits = {i: v + FakeEngine.shift * math.sin(i * 7.1 + t) for i, v in logits.items()}
        z = curate.logsumexp(list(logits.values()))
        return sorted(((v - z, i) for i, v in logits.items()), reverse=True)

    def generate(self, req):
        ids = req["input_ids"]
        if not req.get("return_logprob"):                   # the greedy answer to a prompt /prepare gave
            out = self.answer_ids(FakeEngine.prompts[tuple(ids)])
            return {"text": "...", "output_ids": out, "meta_info": {"prompt_tokens": len(ids),
                                                                     "completion_tokens": len(out)}}
        own, top = [], []
        for t in range(req["logprob_start_len"], len(ids)):
            d = self.dist(ids[t], t)
            own.append([next(lp for lp, i in d if i == ids[t]), ids[t], None])
            top.append([[lp, i, None] for lp, i in d[:req["top_logprobs_num"]]])
        return {"text": "", "output_ids": [EOS], "meta_info": {"input_token_logprobs": own, "input_top_logprobs": top}}

    def do_POST(self):
        if self.headers.get("X-Repertory-Key") != KEY:
            return self.send_json({}, 403)
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeEngine.seen.append((self.path, req))
        if self.path == "/tokenize":
            return self.send_json({"tokens": [0] * (len(req["prompt"]) // 4)})
        if self.path == "/prepare":
            ids = self.prompt_ids(req)
            FakeEngine.prompts[tuple(ids)] = req
            return self.send_json({"ids": ids})
        if self.path == "/generate":
            return self.send_json(self.generate(req))
        reasoning, out, finish = self.answer(req)
        events = [{"choices": [{"delta": {"reasoning_content": w + " "}}]} for w in (reasoning or "").split()]
        if isinstance(out, list):
            events += [{"choices": [{"delta": {"tool_calls": [piece]}}]} for piece in out]
            n = 28
        else:
            events += [{"choices": [{"delta": {"content": w + " "}}]} for w in out.split(" ")]
            n = len(events)
        events.append({"choices": [{"delta": {}, "finish_reason": finish}],
                       "usage": {"prompt_tokens": len(req["messages"][-1]["content"]) // 4, "completion_tokens": n}})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for ev in events:
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
            self.wfile.flush()
            time.sleep(FakeEngine.token_s)
        self.wfile.write(b"data: [DONE]\n\n")


class FakeHost:
    kind, follower = "windows-wsl", None

    def __init__(self):
        self.envs, self.state = [], "stopped"

    def backend_start(self, env, args):
        self.envs.append(env)
        self.state = "running"
        return True, 4242

    def backend_stop(self, timeout=30):
        self.state = "stopped"
        return "stopped"

    def backend_status(self):
        return self.state, (4242 if self.state == "running" else None)


class FakeService:
    """What the benchmark uses from the manager (as tests/test_bench.py's)."""
    observe, gpu_display = False, True

    def __init__(self, port, gpu="NVIDIA GeForce RTX 5090"):
        self.cfg = dict(config.DEFAULTS, backend_port=port, quant="4.05", context=262144, start_timeout_s=20)
        self.host, self.learned = FakeHost(), {}
        self.setup = types.SimpleNamespace(running=None, check=self._check, hw={
            "gpus": [{"name": gpu, "pcie_gen_max": 5, "pcie_width_max": 16}], "ram": {"total_gb": 68.4},
            "engine": {"driver": "1", "m305": "1", "m405": "1", "torch": "2.15.0+cu132", "cuda_version": "13.2",
                       "engine_head": "fedcba9876543210fedcba9876543210fedcba98"},
            "driver": {"version": "617.14"}, "repo": {"head": "abcdef0123"}})
        self.backend_key, self.backend_auth = KEY, {"X-Repertory-Key": KEY}
        self.be = types.SimpleNamespace(state="running", tier="full")
        self.mon = types.SimpleNamespace(get=self.sample)
        self.started, self.stopped = [], []

    def sample(self):
        return {"t": time.time(), "vram_total": 34.2e9, "vram_used": 2e9 + (26e9 if self.host.state == "running"
                                                                               else 0),
                "ram_avail": 50e9, "perproc_ded": 1e9, "running": set(), "ext_util": 0.0, "ext_util_top": []}

    def ext_vram(self, m):
        return 2e9

    def ram_for_start(self, m):
        return m["ram_avail"]

    async def _check(self):
        pass

    async def fresh_sample(self, after):
        return self.mon.get()

    def headroom(self, m):
        return 2.4e9

    def lean_budget(self, m):
        return 16e9

    def full_budget(self, m):
        return 29.8e9

    def learned_key(self, quant=None, context=None):
        return f"gpu|{quant}|{context}"

    def ram_overhead(self, quant=None):
        return 9e9

    async def stop(self, reason, why):
        self.stopped.append(reason)
        self.be.state, self.be.tier = "stopped", None

    async def start(self, tier, reason):
        self.started.append(tier)
        self.be.state, self.be.tier = "running", tier
        return True


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeEngine)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        FakeEngine.token_s, FakeEngine.changed, FakeEngine.forgetful = 0.004, None, False
        FakeEngine.shift, FakeEngine.template, FakeEngine.seen = 0.0, None, []
        self.refs = tempfile.mkdtemp(prefix="repertory-refs-")
        self.svc = FakeService(self.server.server_address[1])
        self.specs = []
        self.model = catalog.load().model(M405)
        # timing on a test machine is noisy (the fake's first token comes in milliseconds): speeds are checked in
        # SpeedTests, and here only where a test sets a tolerance (100%: any speed passes)
        patcher = mock.patch.multiple(curate, DECODE_TOLERANCE=1.0, PREFILL_TOLERANCE=1.0)
        patcher.start()
        self.addCleanup(patcher.stop)
        # the benchmark finds the reference it replays where curate.py keeps it
        patcher = mock.patch.object(catalog, "REFERENCE_DIR", Path(self.refs))
        patcher.start()
        self.addCleanup(patcher.stop)

    def reference(self):
        return json.loads((Path(self.refs) / f"{M405}.json").read_text(encoding="utf-8"))

    def runner(self, specs, workload):
        """The benchmark, as `repertory curate` runs it: it sets the model aside, runs, and brings it back."""
        self.specs.append(list(specs))
        b = bench.Bench(self.svc)

        async def go():
            ok, msg = b.start(specs, workload)
            self.assertTrue(ok, msg)
            await b.task
        asyncio.run(go())
        return b.records

    def regression(self, record=False):
        return curate.regression(self.model, self.runner, record, self.refs)

    def test_record_then_compare(self):
        r = self.regression()
        self.assertIsNone(r["ok"])                                          # nothing to compare with yet
        self.assertIn("--record", r["summary"])
        self.assertEqual(self.specs, [])                                    # and the GPU isn't taken for it
        r = self.regression(record=True)
        self.assertIsNone(r["ok"], r)                                       # recorded, not yet compared
        self.assertTrue(all(i["ok"] for i in r["items"]), r["items"])       # the answers are fit to be a reference
        self.assertEqual(self.specs, [["4.05:card:262144"]])                # the whole card at its full context
        ref = json.loads((Path(self.refs) / f"{M405}.json").read_text(encoding="utf-8"))
        self.assertEqual(ref["protocol"], bench.REFERENCE_PROTOCOL)
        self.assertEqual([c["name"] for c in ref["cases"]], ["short", "code", "tool", "document", "reasoning"])
        tool = ref["cases"][2]
        self.assertEqual((tool["tool_calls"], tool["finish"]),
                         ([{"name": "get_weather", "arguments": '{"city": "Lisbon"}'}], "tool_calls"))
        self.assertEqual(ref["cases"][4]["reasoning"], "221 is 13 times 17. ")
        self.assertEqual(ref["machine"], {"gpu": "NVIDIA GeForce RTX 5090", "vram_gb": 34.2, "ram_gb": 68.4,
                                          "platform": "windows-wsl"})
        self.assertEqual(ref["engine_commit"], "fedcba9876543210fedcba9876543210fedcba98")
        self.assertGreater(ref["speed"]["decode_tps"], 0)
        self.assertGreater(ref["speed"]["prefill_tps"], 0)
        self.assertGreater(ref["run"]["ram_gib"], 0)
        # the token data: every case's but the document's, which keeps the facts its answer must name
        doc = ref["cases"][3]
        self.assertEqual((doc["expect"], "tokens" in doc), (["Ilsa Varga", "PERIWINKLE"], False))
        for c in ref["cases"][:3] + ref["cases"][4:]:
            t, case = c["tokens"], dict(bench.REFERENCE_CASES)[c["name"]]
            self.assertEqual(t["prompt"], FakeEngine.prompt_ids(bench.Client(0, "").reference_body(case)))
            self.assertEqual(t["answer"][-1], EOS)                        # to its natural end, the stop token too
            self.assertEqual(len(t["logprobs"]), len(t["answer"]))
            self.assertEqual([len(t[k]) for k in ("top_ids", "top_logprobs")], [len(t["answer"])] * 2)
            self.assertTrue(all(len(row) == bench.REFERENCE_TOP for row in t["top_ids"] + t["top_logprobs"]))
            self.assertEqual([row[0] for row in t["top_ids"]], t["answer"])  # the fake is sure of each token
            self.assertTrue(all(round(x, 3) == x for row in t["top_logprobs"] for x in row))
        self.assertEqual(ref["cases"][2]["tokens"]["prompt"][-1], 12)        # the tool case's prompt has the tools
        self.assertEqual(ref["cases"][4]["tokens"]["prompt"][-1], 11)        # and the last has thinking on
        # compact: a line a position, each list of numbers on one line
        text = (Path(self.refs) / f"{M405}.json").read_text(encoding="utf-8")
        rows = sum(len(c["tokens"]["answer"]) for c in ref["cases"] if "tokens" in c)
        self.assertEqual(len(re.findall(r"^ +\[\d+(, \d+){19}\],?$", text, re.M)), rows)
        self.assertLess(text.count("\n"), 2 * rows + 200)
        # the next run starts the same configuration, with the reference's RAM cache, and compares
        r = self.regression()
        self.assertEqual(self.specs[-1], [f"4.05:card:262144:ram={ref['run']['ram_gib']}"])
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["numbers"]["identical"], "5/5")
        self.assertEqual((r["numbers"]["short_top1"], r["numbers"]["short_kl_mean"], r["numbers"]["code_kl_max"]),
                         (1.0, 0.0, 0.0))
        self.assertEqual((r["numbers"]["kl_mean_limit"], r["numbers"]["top1_limit"]), (0.002, 0.98))
        items = by_name(r)
        self.assertIn("top-1 agreement", items["short"]["detail"])
        self.assertIn("text identical", items["short"]["detail"])
        self.assertIn("names Ilsa Varga and PERIWINKLE", items["document"]["detail"])
        self.assertEqual(r["summary"], "answers within the tolerance (5/5 identical as text), speed within the "
                                       "tolerance")
        self.assertEqual(self.svc.started, ["full", "full"])                # the model came back after each run
        self.assertEqual(self.svc.host.envs[-1]["RAM_GB"], str(ref["run"]["ram_gib"]))
        self.assertTrue(all(e["QUANT"] == "4.05" and e["MODEL_DIR"] == "./model-4.05" for e in self.svc.host.envs))
        # the reference's own answers were what this run teacher-forced, not its own
        forced = [p["input_ids"] for path, p in FakeEngine.seen if path == "/generate" and p.get("return_logprob")]
        for c in ref["cases"]:
            if "tokens" in c:
                self.assertIn(c["tokens"]["prompt"] + c["tokens"]["answer"], forced)

    def test_new_text_passes_when_its_tokens_replay_and_says_where_it_parted(self):
        self.regression(record=True)
        FakeEngine.changed = "heat pump moves heat."                     # as when a near tie flips the answer
        r = self.regression()
        self.assertTrue(r["ok"], r)
        items = by_name(r)
        self.assertTrue(items["short"]["ok"])
        self.assertIn("text not identical (content, completion_tokens differ", items["short"]["detail"])
        self.assertIn("became", items["short"]["detail"])
        self.assertIn("which alone doesn't fail", items["short"]["detail"])
        self.assertEqual(r["numbers"]["identical"], "4/5")

    def test_a_damaged_model_fails(self):
        self.regression(record=True)
        FakeEngine.shift = 1.5                                           # every logit moved
        r = self.regression()
        self.assertFalse(r["ok"])
        self.assertEqual(r["summary"], "regressed: short, code, tool, reasoning")
        self.assertGreater(r["numbers"]["short_kl_mean"], 10 * curate.KL_MEAN_MAX)
        self.assertTrue(by_name(r)["document"]["ok"])                     # its facts are still named
        self.assertIn("text identical", by_name(r)["short"]["detail"])    # the same text doesn't save it

    def test_a_document_answer_that_misses_its_facts_fails(self):
        self.regression(record=True)
        FakeEngine.forgetful = True
        r = self.regression()
        self.assertEqual(r["summary"], "regressed: document")
        self.assertIn("doesn't name Ilsa Varga or PERIWINKLE", by_name(r)["document"]["detail"])

    def test_a_prompt_whose_tokens_changed_fails(self):
        self.regression(record=True)
        FakeEngine.template = "v2"
        r = self.regression()
        self.assertFalse(by_name(r)["short"]["ok"])
        self.assertIn("prompt's tokens differ from the reference's", by_name(r)["short"]["detail"])
        self.assertEqual(r["numbers"]["short_kl_mean"], 0.0)              # the reference's own prompt replays fine

    def test_a_reference_from_before_the_token_comparison_needs_recording_again(self):
        self.regression(record=True)
        path = Path(self.refs) / f"{M405}.json"
        old = self.reference()
        for c in old["cases"]:
            c.pop("tokens", None)
            c.pop("expect", None)
        path.write_text(json.dumps(old, indent=1), encoding="utf-8")
        r = self.regression()
        self.assertIsNone(r["ok"])                                       # untested, not failed
        self.assertEqual(curate.status_of({"compat": {"ok": True}, "regression": r}), "untested")
        self.assertIn("predates the token-by-token comparison: record it again with --record", r["summary"])
        self.assertEqual([(i["name"], i["ok"]) for i in r["items"]], [("reference", None)])
        self.assertEqual(len(self.specs), 1)                              # and the GPU isn't taken for it
        # recording replaces it, and the run after compares
        self.assertTrue(all(i["ok"] for i in self.regression(record=True)["items"]))
        self.assertTrue(self.regression()["ok"])

    def test_slower_decode_fails(self):
        self.regression(record=True)
        FakeEngine.token_s = 0.03                                          # several times slower a token
        with mock.patch.object(curate, "DECODE_TOLERANCE", 0.15):
            r = self.regression()
        items = by_name(r)
        self.assertTrue(items["short"]["ok"])                               # same answers...
        self.assertFalse(items["decode"]["ok"])                             # ...too slowly
        self.assertLess(r["numbers"]["decode_change_pct"], -15)

    def test_another_machine_compares_answers_but_not_speed(self):
        self.regression(record=True)
        self.svc.setup.hw["gpus"][0]["name"] = "NVIDIA GeForce RTX 4090"
        r = self.regression()
        self.assertIsNone(r["ok"])
        self.assertIsNone(by_name(r)["decode"]["ok"])
        self.assertIn("measured on NVIDIA GeForce RTX 5090", by_name(r)["decode"]["detail"])
        self.assertEqual(r["summary"], "answers within the tolerance (5/5 identical as text); not decided: decode, "
                                       "prefill")

    def test_answers_unfit_to_be_a_reference_are_not_recorded(self):
        FakeEngine.forgetful = True
        r = self.regression(record=True)
        self.assertFalse(r["ok"])
        self.assertFalse(by_name(r)["document"]["ok"])
        self.assertFalse((Path(self.refs) / f"{M405}.json").exists())

    def test_a_run_that_fails_or_is_cut_short(self):
        self.regression(record=True)
        r = curate.regression(self.model, lambda specs, wl: [{"error": "does not fit: 6 GB"}], folder=self.refs)
        self.assertFalse(r["ok"])
        r = curate.regression(self.model, lambda specs, wl: [{"error": "interrupted: GPU busy", "interrupted": True}],
                              folder=self.refs)
        self.assertIsNone(r["ok"])
        self.assertIsNone(curate.regression(self.model, lambda specs, wl: None, folder=self.refs)["ok"])

    def test_requests_are_greedy_never_capped_and_think_only_once(self):
        seen = []
        real = bench.Client.events

        def events(client, path, payload, timeout=7200):
            seen.append(payload)
            return real(client, path, payload, timeout)
        with mock.patch.object(bench.Client, "events", events):
            self.regression(record=True)
        cases = [p for p in seen if p.get("stream_options")]
        self.assertEqual(len(cases), 5)
        for p in seen:
            self.assertNotIn("max_tokens", p)
            self.assertEqual(p["temperature"], 0)
            self.assertEqual(p["model"], "qwen3.8-flash-next")
        self.assertEqual([p["chat_template_kwargs"]["enable_thinking"] for p in cases],
                         [False, False, False, False, True])
        # the prompts as token ids: the very requests the chat route ran (not the document's), and greedy answers
        prepared = [p for path, p in FakeEngine.seen if path == "/prepare"]
        streamed = [{k: v for k, v in p.items() if k not in ("stream", "stream_options")} for p in cases]
        self.assertEqual(prepared, streamed[:3] + streamed[4:])
        greedy = [p for path, p in FakeEngine.seen if path == "/generate" and not p.get("return_logprob")]
        self.assertEqual([p["sampling_params"] for p in greedy], [{"temperature": 0}] * 4)   # no max_new_tokens


def made_up(n=60, seed=0):
    """A reference answer's token data: n positions, at each its own token the likeliest (about 0.9) and 19 others."""
    rng = random.Random(seed)                                          # noqa: S311 - made-up data, seeded
    t = {"prompt": [1, 2, 3], "answer": [], "logprobs": [], "top_ids": [], "top_logprobs": []}
    for j in range(n):
        logits = [5.0] + sorted((rng.uniform(-3.0, 2.0) for _ in range(19)), reverse=True)
        z = curate.logsumexp(logits)
        t["answer"].append(500 + j)
        t["top_ids"].append([500 + j] + rng.sample(range(1000, 5000), 19))
        t["top_logprobs"].append([round(x - z, 3) + 0.0 for x in logits])
        t["logprobs"].append(t["top_logprobs"][-1][0])
    return t


def replay_of(tokens):
    """A run that replays the reference exactly."""
    return copy.deepcopy({k: tokens[k] for k in ("logprobs", "top_ids", "top_logprobs")})


class SpeedTests(unittest.TestCase):
    """The comparison on its own, with made-up runs: the tolerance and what a different machine means."""
    machine = {"gpu": "G", "vram_gb": 34.2, "ram_gb": 68.4, "platform": "linux"}

    def record(self, decode, prefill, gpu="G"):
        cases = []
        for n, _ in bench.REFERENCE_CASES:
            c = dict({k: None for k in bench.REFERENCE_ANSWER}, name=n)
            if n in bench.REFERENCE_FACTS:
                c["content"] = "Ilsa Varga serviced it; the code word is PERIWINKLE."
            else:
                c["tokens"] = made_up(8)
                c["replay"] = replay_of(c["tokens"])
            cases.append(c)
        return {"host": {"gpu": gpu, "ram_total_gb": 68.4, "platform": "linux"}, "card_total_gb": 34.2,
                "reference": {"cases": cases, "decode_tps": decode, "prefill_tps": prefill}}

    def ref(self):
        rec = self.record(60.0, 5000.0)
        cases = [curate.reference_case(c) for c in rec["reference"]["cases"]]
        return {"machine": self.machine, "speed": {"decode_tps": 60.0, "prefill_tps": 5000.0}, "cases": cases}

    def test_within_the_tolerance(self):
        items, numbers = curate.compare(self.ref(), self.record(51.6, 4510.0))     # 14% and 9.8% slower
        self.assertEqual(curate.outcome(items), True)
        self.assertEqual((numbers["decode_change_pct"], numbers["prefill_change_pct"]), (-14.0, -9.8))
        items, _ = curate.compare(self.ref(), self.record(80.0, 9000.0))            # faster never fails
        self.assertEqual(curate.outcome(items), True)

    def test_beyond_it(self):
        items, _ = curate.compare(self.ref(), self.record(50.4, 5000.0))           # 16% slower
        self.assertFalse(by_name({"items": items})["decode"]["ok"])
        items, _ = curate.compare(self.ref(), self.record(60.0, 4400.0))           # 12% slower
        self.assertFalse(by_name({"items": items})["prefill"]["ok"])

    def test_the_tolerances(self):
        self.assertEqual((curate.DECODE_TOLERANCE, curate.PREFILL_TOLERANCE), (0.15, 0.10))
        self.assertEqual((curate.KL_MEAN_MAX, curate.TOP1_MIN, curate.MISSING_MARGIN), (0.002, 0.98, 1.0))


class ReplayTests(unittest.TestCase):
    """The token comparison's figures on made-up distributions: what last-bit noise and a near tie do to them, and
    what a damaged model does."""

    def test_the_same_distributions(self):
        t = made_up()
        m = curate.replay_metrics(t, replay_of(t))
        self.assertEqual((m["positions"], m["agreed"], m["top1"]), (60, 60, 1.0))
        self.assertEqual((m["kl_mean"], m["kl_max"], m["logprob_delta_mean"], m["logprob_delta_max"]),
                         (0.0, 0.0, 0.0, 0.0))
        self.assertTrue(curate.replay_passes(m))

    def test_last_bit_noise_and_a_flipped_near_tie_pass(self):
        t = made_up()
        a, b = t["top_ids"][10][:2]
        t["top_logprobs"][10][:2] = [-0.700, -0.701]                   # two tokens nearly tied in the reference
        t["logprobs"][10] = -0.700
        run = replay_of(t)
        rng = random.Random(1)                                         # noqa: S311 - made-up noise, seeded
        for row in run["top_logprobs"]:                                # every logprob a little off
            row[:] = [round(x + rng.uniform(-0.002, 0.002), 3) for x in row]
        run["top_ids"][10][:2] = [b, a]                                # and the tie goes the other way
        run["top_logprobs"][10][:2] = [-0.700, -0.702]
        run["logprobs"] = [row[0] for row in run["top_logprobs"]]
        run["logprobs"][10] = -0.702                                   # the reference's token, now second
        m = curate.replay_metrics(t, run)
        self.assertEqual(m["agreed"], 59)
        self.assertGreaterEqual(m["top1"], curate.TOP1_MIN)
        self.assertLess(m["kl_mean"], 1e-4)
        self.assertGreater(m["kl_max"], 0.0)
        self.assertAlmostEqual(m["logprob_delta_max"], 0.002, places=6)
        self.assertTrue(curate.replay_passes(m))

    def test_a_damaged_model_fails(self):
        t = made_up()
        flat = replay_of(t)                                            # every distribution flatter
        for row in flat["top_logprobs"]:
            z = curate.logsumexp([x / 2 for x in row])
            row[:] = [round(x / 2 - z, 3) for x in row]
        flat["logprobs"] = [row[0] for row in flat["top_logprobs"]]
        m = curate.replay_metrics(t, flat)
        self.assertEqual(m["top1"], 1.0)                               # the likeliest tokens stay...
        self.assertGreater(m["kl_mean"], 10 * curate.KL_MEAN_MAX)      # ...the distributions don't
        self.assertFalse(curate.replay_passes(m))
        wrong = replay_of(t)                                           # another token wins one position in ten
        for j in range(0, 60, 10):
            wrong["top_ids"][j][:2] = wrong["top_ids"][j][1::-1]
        m = curate.replay_metrics(t, wrong)
        self.assertEqual(m["agreed"], 54)
        self.assertFalse(curate.replay_passes(m))

    def test_a_token_missing_from_one_list(self):
        p, q = [math.log(0.9), math.log(0.1)], [math.log(0.8), math.log(0.2)]
        got = curate.position_kl([1, 2], p, [1, 3], q)
        # token 3 isn't in the reference's list, token 2 isn't in the run's: each at its list's smallest, less 1 nat
        ref = {1: 0.9, 2: 0.1, 3: 0.1 / math.e}
        run = {1: 0.8, 2: 0.2 / math.e, 3: 0.2}
        zr, zq = sum(ref.values()), sum(run.values())
        want = sum(ref[t] / zr * math.log((ref[t] / zr) / (run[t] / zq)) for t in ref)
        self.assertAlmostEqual(got, want, places=12)
        self.assertEqual(curate.position_kl([1, 2], p, [1, 2], p), 0.0)

    def test_a_replay_missing_from_the_run_is_undecided(self):
        t = made_up(8)
        ok, detail = curate.replay_case({"name": "short", "tokens": t}, {"tokens": t}, {})
        self.assertIsNone(ok)
        self.assertIn("wasn't replayed", detail)
        short = dict(replay_of(t), top_ids=replay_of(t)["top_ids"][:-1])
        self.assertIsNone(curate.replay_case({"name": "short", "tokens": t}, {"replay": short}, {})[0])


# ------------------------------------------------------------------------------------------------- the result
class ResultTests(unittest.TestCase):
    def test_status_from_the_checks(self):
        ok, bad, open_ = {"ok": True}, {"ok": False}, {"ok": None}
        self.assertEqual(curate.status_of({"compat": ok, "regression": ok}), "curated")
        self.assertEqual(curate.status_of({"compat": ok, "regression": bad}), "failed")
        self.assertEqual(curate.status_of({"compat": bad}), "failed")
        self.assertEqual(curate.status_of({"compat": ok, "regression": open_}), "untested")
        self.assertEqual(curate.status_of({"compat": ok}), "untested")

    def test_exit_codes(self):
        self.assertEqual(curate.exit_code({"compat": {"ok": True}}), 0)
        self.assertEqual(curate.exit_code({"compat": {"ok": True}, "regression": {"ok": None}}), 1)
        self.assertEqual(curate.exit_code({"compat": {"ok": True}, "regression": {"ok": None,
                                                                                  "numbers": {"recorded": True}}},
                                          record=True), 0)
        self.assertEqual(curate.exit_code({"compat": {"ok": False}, "regression": None}), 1)

    def test_run_writes_the_result_into_the_entry_and_keeps_checks_it_didnt_run(self):
        root = tempfile.mkdtemp(prefix="repertory-cat-")
        path = Path(root) / "catalog.json"
        path.write_text(catalog.CATALOG.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        engine = tempfile.mkdtemp(prefix="repertory-engine-")
        kw = dict(cfg=dict(config.DEFAULTS), machine=rtx5090(), files=Files(engine), hub=FakeHub(),
                  catalog_path=path, reference_dir=root)
        regression = {"ok": True, "date": "2026-11-07", "items": [{"name": "short", "ok": True}],
                      "engine_commit": "abc", "machine": {"gpu": "G", "vram_gb": 34.2, "ram_gb": 68.4,
                                                          "platform": "linux"},
                      "numbers": {"identical": "5/5", "decode_tps": 61.2}}
        with mock.patch.object(curate, "regression", lambda *a, **k: copy.deepcopy(regression)):
            rec, results = curate.run(M405, checks="all", runner=None, **kw)
        self.assertEqual(rec["status"], "curated")
        entry = catalog.load(path).model(M405)
        self.assertEqual(entry["curation"], rec)
        self.assertEqual(entry["curation"]["machine"], regression["machine"])
        self.assertTrue(catalog.offered(entry))
        # later, compatibility alone: the regression's result stands
        rec, _ = curate.run("4.05", checks="compat", **kw)
        entry = catalog.load(path).model(M405)
        self.assertEqual(rec["status"], "curated")
        self.assertEqual(entry["curation"]["checks"]["regression"], regression)
        self.assertEqual(entry["curation"]["checks"]["compat"]["machine"]["gpu"], "NVIDIA GeForce RTX 5090")
        # a check that fails: failed, and no longer offered; the GPU isn't taken when compatibility fails
        kw["machine"] = rtx5090(ram_gib=16)
        rec, results = curate.run(M405, checks="all", runner=lambda *a: self.fail("the GPU was taken"), **kw)
        self.assertEqual((rec["status"], results["regression"]), ("failed", None))
        self.assertFalse(catalog.offered(catalog.load(path).model(M405)))
        # and nothing else in the catalog changed
        got = json.loads(path.read_text(encoding="utf-8"))
        got["models"][1].pop("curation")
        self.assertEqual(got, DATA)
        # --no-write
        before = path.read_text(encoding="utf-8")
        curate.run(M305, checks="compat", write=False, **kw)
        self.assertEqual(path.read_text(encoding="utf-8"), before)


class CliTests(unittest.TestCase):
    def cli(self, *args):
        import contextlib
        from repertory import __main__ as cli
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = cli.main(["curate", *args])
        return rc, out.getvalue()

    def test_the_listing_and_what_it_refuses(self):
        rc, out = self.cli()
        self.assertEqual(rc, 0)
        self.assertIn(f"{M305:<34} 3.05   untested", out)
        self.assertIn("(no curation yet)", out)
        self.assertEqual(self.cli("9.9")[0], 2)
        rc, out = self.cli("4.05", "--checks", "compat", "--record")
        self.assertEqual(rc, 2)
        self.assertIn("--record needs the regression check", out)


if __name__ == "__main__":
    unittest.main()
