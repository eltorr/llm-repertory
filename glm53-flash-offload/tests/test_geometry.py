"""Engine geometry without a GPU, for every supported model:
  - nv2's compile-time caps (kernels/nv2/nv2_shared.h) hold each model's experts per layer and 8 tokens x top-k picks
  - the built nv2 extension (build/nv2, or GLM53_NV2_SO) reports the same caps and the struct sizes they imply
  - the expert cache's VRAM slot (expert_cache.slot_layout) is the store record's trellis image, i.e. what nv2 and
    nv_tier check at attach (nv_tier.check_pool_layout), for each model and quant
  - the B70 ring constants in nv2_host.cpp are b70tier/ring.py's
Model facts: the published configs below; GLM53_TEST_STORES (store .json paths, ':'-separated) adds real manifests.
Needs torch (CPU) and numpy for the engine modules; the extension check is skipped when it is not built."""
import glob
import importlib.util
import json
import os
import re
import sys
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KER = os.path.join(ROOT, "kernels", "nv2")
sys.path.insert(0, os.path.join(ROOT, "glm53"))

# routed-expert geometry: experts per layer, top-k, hidden, expert width, MoE layers, trellis bits of the quant(s)
MODELS = {
    "GLM-5.3-Flash 3.05bpw": dict(E=288, topk=8, H=4096, I=2048, L=42, K=3),
    "Qwen3.8-Flash-Next 3.05bpw": dict(E=512, topk=10, H=2560, I=640, L=48, K=3),
    "Qwen3.8-Flash-Next 4.05bpw": dict(E=512, topk=10, H=2560, I=640, L=48, K=4),
}


def src(name):
    with open(os.path.join(KER, name)) as f:
        return f.read()


def define(text, name):
    return int(re.search(rf"#define {name} (\d+)", text).group(1))


def constexpr(text, name):
    return int(re.search(rf"constexpr int {name} = (\d+);", text).group(1))


def trellis_bytes(k, n, K):
    return (k // 16) * (n // 16) * 16 * K * 2


def manifest_for(g):
    """The record layout the packer writes for this geometry (gate, up, down trellis; then suh/svh per matrix)."""
    sizes = [trellis_bytes(g["H"], g["I"], g["K"]), trellis_bytes(g["H"], g["I"], g["K"]), trellis_bytes(g["I"], g["H"], g["K"])]
    fields, off = [], 0
    for m, n in zip(("gate_proj", "up_proj", "down_proj"), sizes):
        fields.append({"name": f"{m}.trellis", "offset": off, "bytes": n}); off += n
    return {"fields": fields, "experts_per_layer": g["E"],
            "vram_slot": {"off_u": fields[1]["offset"], "off_d": fields[2]["offset"], "slot_bytes": off}}


class Caps(unittest.TestCase):
    def test_caps_cover_models(self):
        sh = src("nv2_shared.h")
        maxu, maxp, maxb = define(sh, "NV2_MAXU"), define(sh, "NV2_MAXP"), constexpr(sh, "MAXB")
        self.assertLessEqual(maxu, 1024)          # nv_step's admission mask
        for name, g in MODELS.items():
            self.assertLessEqual(g["E"], maxu, name)
            self.assertLessEqual(maxb * g["topk"], maxp, name)   # decode: 8 tokens x top-k picks with payload

    def test_pub_fast_threads(self):
        dev = src("nv2_dev.cu")
        self.assertIn("pubf_threads((int) E)", dev)               # one thread per expert, at least 320
        self.assertIn("__launch_bounds__(PUBF_TMAX)", dev)

    def test_b70_ring_matches_ring_py(self):
        host = src("nv2_host.cpp")
        bm = int(re.search(r"constexpr int B70_MAXP = (\d+), B70_H = (\d+);", host).group(1))
        bh = int(re.search(r"constexpr int B70_MAXP = (\d+), B70_H = (\d+);", host).group(2))
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy")
        spec = importlib.util.spec_from_file_location("b70ring", os.path.join(ROOT, "b70tier", "ring.py"))
        rg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rg)
        self.assertEqual((bm, bh), (rg.MAXP, rg.H))
        off_x = int(re.search(r"B70_OFF_X = (\d+)", host).group(1))
        self.assertEqual(off_x, rg.OFF_X)
        self.assertEqual(off_x + bm * bh * 2, rg.OFF_OUT)
        self.assertEqual((off_x + 2 * bm * bh * 2 + 4095) // 4096 * 4096, rg.SIZE)


class Extension(unittest.TestCase):
    def test_sizes(self):
        so = os.environ.get("GLM53_NV2_SO") or next(iter(glob.glob(os.path.join(ROOT, "build", "nv2", "glm53_nv2*.so"))), None)
        if not so:
            self.skipTest("nv2 extension not built (scripts/install.sh)")
        import torch  # noqa: F401  (the extension links against torch's libraries)
        spec = importlib.util.spec_from_file_location("glm53_nv2", so)
        ext = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ext)
        sreq, srep, slog, maxu, maxp, rq, lg, lr, maxb = ext.sizes()[:9]
        sh = src("nv2_shared.h")
        self.assertEqual((maxu, maxp, maxb), (define(sh, "NV2_MAXU"), define(sh, "NV2_MAXP"), constexpr(sh, "MAXB")))
        self.assertEqual(sreq, 48 + 12 * (maxu + maxp))   # seq, tpub, 6 ints | key, cls, cnt [MAXU] | pe, pw, pf [MAXP]
        self.assertEqual(srep, 16 + 4 * maxu)
        self.assertEqual(slog, 16)


class SlotLayout(unittest.TestCase):
    def setUp(self):
        try:
            import expert_cache, nv_tier  # noqa: F401
        except ImportError as e:
            self.skipTest(f"engine modules need torch + numpy: {e}")
        self.ec, self.nt = expert_cache, nv_tier

    def pool(self, man):
        f = {x["name"]: x["bytes"] for x in man["fields"]}
        sg, su, sd = f["gate_proj.trellis"], f["up_proj.trellis"], f["down_proj.trellis"]
        off_u, off_d, rec = self.ec.slot_layout(sg, su, sd)
        return types.SimpleNamespace(sg=sg, su=su, sd=sd, off_u=off_u, off_d=off_d, rec=rec, first=0,
                                     E=man["experts_per_layer"])

    def test_models(self):
        glm = manifest_for(MODELS["GLM-5.3-Flash 3.05bpw"])
        self.assertEqual(glm["vram_slot"], {"off_u": 3145728, "off_d": 6291456, "slot_bytes": 9437184})
        for name, g in MODELS.items():
            man = manifest_for(g)
            self.nt.check_pool_layout(self.pool(man), man)      # raises on a mismatch
            self.assertEqual(man["vram_slot"]["slot_bytes"] % 4096, 0, name)   # O_DIRECT reads of the trellis part

    def test_real_manifests(self):
        paths = [p for p in os.environ.get("GLM53_TEST_STORES", "").split(":") if p]
        if not paths:
            self.skipTest("GLM53_TEST_STORES not set")
        for p in paths:
            with open(p) as f:
                man = json.load(f)
            self.nt.check_pool_layout(self.pool(man), man)
            self.assertLessEqual(man["experts_per_layer"], define(src("nv2_shared.h"), "NV2_MAXU"))

    def test_mismatch_is_refused(self):
        man = manifest_for(MODELS["Qwen3.8-Flash-Next 3.05bpw"])
        other = manifest_for(MODELS["Qwen3.8-Flash-Next 4.05bpw"])
        with self.assertRaises(AssertionError):
            self.nt.check_pool_layout(self.pool(other), man)


if __name__ == "__main__":
    unittest.main()
