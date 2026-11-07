"""engines/exl3_offload/loader.py without a GPU: the store checks, the GLM53_* environment a load sets, and the
exllamav3 arguments it builds (checked against exllamav3's own add_args, read from the repo's exllamav3 source, so no
exllamav3 import and no CUDA). Real stores / models are checked too when listed:
  GLM53_TEST_STORES=/path/qwen.bin:/path/qwen405.bin  GLM53_TEST_MODELS=/path/model:/path/model-4.05  (same order)
Only manifests, the model index and one safetensors header are read."""
import argparse
import ast
import json
import os
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engines" / "exl3_offload"))
import loader  # noqa: E402

E, HID, INT = 4, 256, 128
KEY0 = "model.language_model.layers.0.mlp"


def manifest(bits=3, layers=3):
    def shape(k, n):
        return [k // 16, n // 16, 16 * bits]

    def nbytes(s):
        return s[0] * s[1] * s[2] * 2
    shapes = {"gate_proj.trellis": shape(HID, INT), "up_proj.trellis": shape(HID, INT),
              "down_proj.trellis": shape(INT, HID)}
    fields, off = [], 0
    for name, s in shapes.items():
        fields.append({"name": name, "offset": off, "bytes": nbytes(s), "dtype": "I16", "shape": s})
        off += nbytes(s)
    trellis = off
    rec = (trellis + 2304 + 4095) // 4096 * 4096
    keys = [f"model.language_model.layers.{i}.mlp" for i in range(layers - 1)] + ["mtp.layers.0.mlp"]
    return {"format": "glm53-expert-store-v1", "record_bytes": rec, "trellis_bytes": trellis, "align": 4096,
            "experts_per_layer": E, "n_layers": layers,
            "layers": [{"li": i, "model_layer": 0, "key": k} for i, k in enumerate(keys)],
            "mtp_layer_li": layers - 1, "fields": fields, "total_bytes": layers * E * rec,
            "vram_slot": {"off_u": fields[1]["offset"], "off_d": fields[2]["offset"], "slot_bytes": trellis}}


def write_store(d, man, name="qwen"):
    b = Path(d) / f"{name}.bin"
    with open(b, "wb") as f:
        f.truncate(man["total_bytes"])
    (Path(d) / f"{name}.json").write_text(json.dumps(man))
    return b


def write_model(d, bits):
    head = {}
    off = 0
    for name, (k, n) in (("gate_proj", (HID, INT)), ("up_proj", (HID, INT)), ("down_proj", (INT, HID))):
        shape = [k // 16, n // 16, 16 * bits]
        nb = shape[0] * shape[1] * shape[2] * 2
        head[f"{KEY0}.experts.0.{name}.trellis"] = {"dtype": "I16", "shape": shape, "data_offsets": [off, off + nb]}
        off += nb
    hb = json.dumps(head).encode()
    with open(Path(d) / "model-00001-of-00001.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb)
        f.truncate(8 + len(hb) + off)
    (Path(d) / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in head}}))
    (Path(d) / "config.json").write_text("{}")


def exllamav3_stub():
    """exllamav3.model_init with the real add_args (from the repo's source), nothing else."""
    src = (ROOT / "exllamav3" / "exllamav3" / "model_init.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "add_args")
    ns = {"ArgumentParser": argparse.ArgumentParser, "SimpleNamespace": types.SimpleNamespace}
    # only add_args, taken from the repo's own exllamav3 source: the rest of model_init needs a GPU
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "exllamav3/model_init.py:add_args", "exec"), ns)  # noqa: S102
    mi = types.ModuleType("exllamav3.model_init")
    mi.add_args = ns["add_args"]
    pkg = types.ModuleType("exllamav3")
    pkg.__path__ = []
    pkg.model_init = mi
    return {"exllamav3": pkg, "exllamav3.model_init": mi}


class Store(unittest.TestCase):
    def test_read_and_bits(self):
        with tempfile.TemporaryDirectory() as d:
            for bits in (3, 4):
                man = manifest(bits)
                b = write_store(d, man, f"s{bits}")
                self.assertEqual(loader.read_store(b)["n_layers"], 3)
                self.assertEqual(loader.expert_bits(man), bits)

    def test_bad_stores(self):
        with tempfile.TemporaryDirectory() as d:
            man = manifest()
            b = write_store(d, man)
            with open(b, "r+b") as f:
                f.truncate(man["total_bytes"] - 4096)
            with self.assertRaises(ValueError):
                loader.read_store(b)
            with self.assertRaises(FileNotFoundError):
                loader.read_store(Path(d) / "missing.bin")
            (Path(d) / "x.bin").write_bytes(b"")
            with self.assertRaises(FileNotFoundError):
                loader.read_store(Path(d) / "x.bin")

    def test_store_matches_model(self):
        with tempfile.TemporaryDirectory() as d:
            write_model(d, 3)
            loader.check_store_matches(d, manifest(3))
            with self.assertRaises(ValueError):
                loader.check_store_matches(d, manifest(4))   # a 4-bit store next to a 3-bit model

    def test_real_stores(self):
        stores = [p for p in os.environ.get("GLM53_TEST_STORES", "").split(":") if p]
        models = [p for p in os.environ.get("GLM53_TEST_MODELS", "").split(":") if p]
        if not stores:
            self.skipTest("GLM53_TEST_STORES not set")
        for i, s in enumerate(stores):
            man = loader.read_store(s)
            self.assertIn(loader.expert_bits(man), (3, 4))
            if i < len(models):
                loader.check_store_matches(models[i], man)
                for j, other in enumerate(stores):
                    if j != i and loader.expert_bits(loader.read_store(other)) != loader.expert_bits(man):
                        with self.assertRaises(ValueError):
                            loader.check_store_matches(models[i], loader.read_store(other))


@unittest.skipUnless(hasattr(os, "sched_getaffinity"), "Linux CPU sets")
class Env(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.man = manifest(3)
        self.store = write_store(self.d.name, self.man)

    def tearDown(self):
        self.d.cleanup()

    def test_nvme_mode(self):
        out = loader.plan_env(self.store, self.man, 6.0, False, {})
        for k, v in {"GLM53_NV": "2", "GLM53_EC": "1", "GLM53_NV_STORE": str(self.store), "GLM53_NV_CPU": "0",
                     "GLM53_NV_RAM_GB": "6.0", "GLM53_EC_STAGE_GB": "auto", "GLM53_EC_STAGE_MIN": "512"}.items():
            self.assertEqual(out[k], v, k)
        have = os.sched_getaffinity(0)
        for k in ("GLM53_NV_CTL_CPU", "GLM53_NV_READER_CPUS"):
            self.assertTrue(set(loader_cpus(out[k])) <= have, (k, out[k]))
        self.assertNotIn("GLM53_NV_CPU_CPUS", out)       # no CPU lane, no lane cores
        self.assertNotIn("GLM53_NV_PREFETCH", out)       # tuning flags keep the engine's defaults

    def test_caller_values_are_kept(self):
        env = {"GLM53_EC_STAGE_MIN": "1024", "GLM53_NV_CTL_CPU": str(min(os.sched_getaffinity(0)))}
        out = loader.plan_env(self.store, self.man, None, False, env)
        self.assertNotIn("GLM53_EC_STAGE_MIN", out)
        self.assertNotIn("GLM53_NV_CTL_CPU", out)
        self.assertNotIn("GLM53_NV_RAM_GB", out)         # None: the engine sizes the tier from the memory cap

    def test_cpu_lane(self):
        out = loader.plan_env(self.store, self.man, 6.0, True, {})
        self.assertEqual(out["GLM53_NV_CPU"], "1")
        self.assertTrue(set(loader_cpus(out["GLM53_NV_CPU_CPUS"])) <= os.sched_getaffinity(0))
        man4 = manifest(4)
        with self.assertRaises(ValueError):
            loader.plan_env(write_store(self.d.name, man4, "q4"), man4, 6.0, True, {})

    def test_all_ram_flags_refused(self):
        for env in ({"GLM53_ZC_VRAM": "0"}, {"GLM53_CPU_TIER": "1"}):
            with self.assertRaises(ValueError):
                loader.plan_env(self.store, self.man, 6.0, False, env)
        loader.plan_env(self.store, self.man, 6.0, False, {"GLM53_CPU_TIER": "0", "GLM53_ZC_VRAM": ""})


def loader_cpus(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out


class Args(unittest.TestCase):
    def args(self, **kw):
        a = dict(model_dir="/m", chunk=2048, cache_tokens=16384, mtp=0, embed_disk=True, cache_quant=4)
        a.update(kw)
        with mock.patch.dict(sys.modules, exllamav3_stub()):
            return loader.model_args(**a)

    def test_defaults(self):
        a = self.args()
        self.assertEqual((a.model_dir, a.cache_size, a.chunk_size, a.cache_quant), ("/m", 16384, 2048, "4"))
        self.assertEqual((a.mtp, a.num_draft_tokens, a.ngram_ram, a.embed_disk), (False, None, False, False))
        self.assertEqual(a.autosplit_max_batch_size, 1)

    def test_options(self):
        a = self.args(cache_tokens=16385, mtp=3, embed_disk=False, cache_quant=None, chunk=512)
        self.assertEqual((a.cache_size, a.chunk_size, a.cache_quant), (16640, 512, None))   # whole 256-token pages
        self.assertEqual((a.mtp, a.num_draft_tokens, a.ngram_ram), (True, 3, True))
        with mock.patch.dict(os.environ, {"GLM53_MAX_BATCH": "2"}):
            self.assertEqual(self.args().autosplit_max_batch_size, 2)

    def test_load_checks_arguments_first(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                loader.load(Path(d) / "nomodel", Path(d) / "s.bin")
            write_model(d, 3)
            with self.assertRaises(FileNotFoundError):
                loader.load(d, Path(d) / "s.bin")
            with self.assertRaises(ValueError):
                loader.load(d, write_store(d, manifest(4)))   # 4-bit store, 3-bit model


if __name__ == "__main__":
    unittest.main()
