"""scripts/pack_glm53_store.py on small synthetic EXL3 checkpoints in both layouts it packs:
  GLM-5.3-Flash style  MoE layers model.layers.3 / .4 in the index, the MTP layer (model.layers.5) in mtp.safetensors
  Qwen3.8 style        model.language_model.layers.0 / .1, the MTP layer mtp.layers.0 in the main shards (its layer
                       number 0 is also a trunk layer's: they must stay two layers)
Checks the record layout byte for byte against the source tensors, the layer order, the manifest, verify / cmp / diff,
and partial packs. CPU only, stdlib only; needs a filesystem with O_DIRECT (the packer's and the engine's reads)."""
import hashlib
import importlib.util
import json
import mmap
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PACKER = os.path.join(os.path.dirname(HERE), "scripts", "pack_glm53_store.py")
spec = importlib.util.spec_from_file_location("pack_glm53_store", PACKER)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

HID, INT, K, E = 256, 128, 3, 4          # hidden, expert width, bits, experts per layer
MUL1 = struct.pack("<I", 0x83DCD12D)


def odirect_ok(d):
    try:
        fd = os.open(os.path.join(d, ".odirect"), os.O_RDWR | os.O_CREAT | os.O_DIRECT, 0o600)
    except OSError:
        return False
    buf = mmap.mmap(-1, 4096)
    try:
        return os.pwritev(fd, [buf], 0) == 4096
    except OSError:
        return False
    finally:
        os.close(fd)
        os.unlink(os.path.join(d, ".odirect"))


def tmpdir():
    for base in (os.environ.get("GLM53_TEST_TMP"), None, HERE):
        d = tempfile.mkdtemp(prefix="pack_", dir=base)
        if odirect_ok(d):
            return d
        shutil.rmtree(d)
    raise unittest.SkipTest("no O_DIRECT-capable temp directory (set GLM53_TEST_TMP)")


def tensors_of(prefix, e):
    """The tensors of one expert, in a fixed order, with deterministic contents: name -> (dtype, shape, bytes)."""
    rng = random.Random(f"{prefix}.{e}")
    blob = lambda n: bytes(rng.getrandbits(8) for _ in range(n))
    out = {}
    for m, (k, n) in (("gate_proj", (HID, INT)), ("up_proj", (HID, INT)), ("down_proj", (INT, HID))):
        shape = [k // 16, n // 16, 16 * K]
        out[f"{prefix}.experts.{e}.{m}.trellis"] = ("I16", shape, blob(k // 16 * n // 16 * 16 * K * 2))
        out[f"{prefix}.experts.{e}.{m}.suh"] = ("F16", [k], blob(2 * k))
        out[f"{prefix}.experts.{e}.{m}.svh"] = ("F16", [n], blob(2 * n))
        out[f"{prefix}.experts.{e}.{m}.mul1"] = ("I32", [], MUL1)
    return out


def write_st(path, tensors):
    head, blobs, off = {}, [], 0
    for k, (dt, shape, data) in tensors.items():
        head[k] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    hb = json.dumps(head).encode()
    hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb + b"".join(blobs))


def make_model(d, style):
    """Writes the checkpoint; returns the expected store layers [(key prefix, model layer)] and the MTP index."""
    if style == "glm":
        trunk, mtp, mtp_file = ["model.layers.3.mlp", "model.layers.4.mlp"], "model.layers.5.mlp", True
    else:
        trunk, mtp, mtp_file = ["model.language_model.layers.0.mlp", "model.language_model.layers.1.mlp"], "mtp.layers.0.mlp", False
    shards = {"model-00001-of-00002.safetensors": {}, "model-00002-of-00002.safetensors": {}}
    names = list(shards)
    for i, pfx in enumerate(trunk):
        for e in range(E):
            shards[names[i]].update(tensors_of(pfx, e))
    shards[names[0]]["model.embed_tokens.weight"] = ("F16", [8, HID], bytes(16 * HID))
    mtp_t = {}
    for e in range(E):
        mtp_t.update(tensors_of(mtp, e))
    if mtp_file:
        write_st(os.path.join(d, "mtp.safetensors"), mtp_t)
    else:
        shards[names[1]].update(mtp_t)
    wm = {}
    for fn, t in shards.items():
        write_st(os.path.join(d, fn), t)
        wm.update({k: fn for k in t})
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": wm}, f)
    return [(p, int(p.split(".layers.")[1].split(".")[0])) for p in trunk + [mtp]], len(trunk)


def expected_record(prefix, e, rec):
    t = tensors_of(prefix, e)
    parts = [t[f"{prefix}.experts.{e}.{m}.trellis"][2] for m in P.MATS]
    parts += [t[f"{prefix}.experts.{e}.{m}.{s}"][2] for m in P.MATS for s in ("suh", "svh")]
    b = b"".join(parts)
    return b + bytes(rec - len(b))


class PackStore(unittest.TestCase):
    def setUp(self):
        self.d = tmpdir()
        self.model, self.out = os.path.join(self.d, "model"), os.path.join(self.d, "store")
        os.makedirs(self.model); os.makedirs(self.out)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def pack(self, *args, name="s", mode="pack"):
        r = subprocess.run([sys.executable, PACKER, mode, "--model", self.model, "--out-dir", self.out, "--name", name,
                            "--threads", "2", *args], capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr

    def meta(self, name="s"):
        with open(os.path.join(self.out, name + ".json")) as f:
            return json.load(f)

    def read(self, name="s"):
        with open(os.path.join(self.out, name + ".bin"), "rb") as f:
            return f.read()

    def check_full(self, style, mtp=1):
        layers, ntrunk = make_model(self.model, style)
        if not mtp:
            layers = layers[:ntrunk]
        rc, log = self.pack("--mtp", str(mtp))
        self.assertEqual(rc, 0, log)
        m = self.meta()
        trellis = 3 * HID // 16 * INT // 16 * 16 * K * 2
        rec = P.al(trellis + 2 * 3 * (HID + INT))
        self.assertEqual((m["format"], m["experts_per_layer"], m["n_layers"]), (P.FORMAT, E, len(layers)))
        self.assertEqual((m["trellis_bytes"], m["record_bytes"], m["align"]), (trellis, rec, 4096))
        self.assertEqual([(L["key"], L["model_layer"]) for L in m["layers"]], layers)
        self.assertEqual([L["li"] for L in m["layers"]], list(range(len(layers))))
        self.assertEqual(m["mtp_layer_li"], ntrunk if mtp else None)
        self.assertEqual(m["vram_slot"], {"off_u": trellis // 3, "off_d": 2 * trellis // 3, "slot_bytes": trellis})
        self.assertEqual([f["name"] for f in m["fields"]], [f"{x}.trellis" for x in P.MATS] +
                         [f"{x}.{s}" for x in P.MATS for s in ("suh", "svh")])
        self.assertEqual(m["mul1_markers_hex"], [MUL1.hex()])
        data = self.read()
        self.assertEqual(len(data), m["total_bytes"])
        self.assertEqual(m["total_bytes"], len(layers) * E * rec)
        for li, (pfx, _) in enumerate(layers):
            for e in range(E):
                r = li * E + e
                want = expected_record(pfx, e, rec)
                self.assertEqual(data[r * rec:(r + 1) * rec], want, (pfx, e))
                self.assertEqual(m["sha256"][r], hashlib.sha256(want).hexdigest())
        for mode in ("verify", "cmp"):
            rc, log = self.pack(mode=mode)
            self.assertEqual(rc, 0, log)
            self.assertIn("OK", log)
        return layers, rec

    def test_glm53_layout(self):
        self.check_full("glm")

    def test_glm53_without_mtp(self):
        self.check_full("glm", mtp=0)

    def test_qwen38_layout(self):
        layers, _ = self.check_full("qwen")
        # mtp.layers.0 and model.language_model.layers.0 share a layer number but are separate store layers
        self.assertEqual([p for p, _ in layers], ["model.language_model.layers.0.mlp", "model.language_model.layers.1.mlp",
                                                   "mtp.layers.0.mlp"])

    def test_qwen38_without_mtp(self):
        self.check_full("qwen", mtp=0)

    def test_partial_pack_and_diff(self):
        layers, rec = self.check_full("qwen")
        rc, log = self.pack("--layers", "0,-1", "--limit", "2", name="part")
        self.assertEqual(rc, 0, log)
        self.assertIn("PARTIAL", log)
        m, full = self.meta("part"), self.meta()
        self.assertEqual(m["partial"], {"layers": [0, len(layers) - 1], "experts": 2})
        held = [r for r, h in enumerate(m["sha256"]) if h]
        self.assertEqual(held, [0, 1, (len(layers) - 1) * E, (len(layers) - 1) * E + 1])
        self.assertEqual([m["sha256"][r] for r in held], [full["sha256"][r] for r in held])
        self.assertEqual(os.path.getsize(os.path.join(self.out, "part.bin")), full["total_bytes"])
        for k in P.LAYOUT_KEYS:
            self.assertEqual(m[k], full[k], k)
        for mode in ("verify", "cmp"):
            rc, log = self.pack(mode=mode, name="part")
            self.assertEqual(rc, 0, log)
        ref = os.path.join(self.out, "s.bin")
        rc, log = self.pack("--ref", ref, mode="diff", name="part")
        self.assertEqual(rc, 0, log)
        self.assertIn("DIFF OK", log)
        self.assertIn("4 identical, 0 differ", log)
        # one flipped byte in a record the partial store holds: diff must fail
        with open(ref, "r+b") as f:
            f.seek(rec + 100)
            b = f.read(1)
            f.seek(rec + 100)
            f.write(bytes([b[0] ^ 0xFF]))
        rc, log = self.pack("--ref", ref, mode="diff", name="part")
        self.assertEqual(rc, 1, log)
        self.assertIn("1 differ [1]", log)

    def test_parse_layers(self):
        self.assertEqual(P.parse_layers("", 5), [0, 1, 2, 3, 4])
        self.assertEqual(P.parse_layers("0,-1", 5), [0, 4])
        self.assertEqual(P.parse_layers("1-3,0", 5), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
