#!/usr/bin/env python3
"""Pack the routed experts of an EXL3 MoE checkpoint into one O_DIRECT-friendly store on NVMe (GLM53_MODE=nvme /
nvme-exact). Made for GLM-5.3-Flash (campaign N116; the measured store sits on a 4-drive NVMe RAID0 at /mnt/nvx/glm53);
any checkpoint with uniform per-expert EXL3 tensors packs the same way (Qwen3.8-Flash-Next: --name qwen).

Python stdlib only, CPU only. Reads the checkpoint read-only (safetensors headers + pread), writes only under --out-dir.

Store layout (little-endian, every record 4096-aligned, layer-major):
  record index r = li * E + e     li = 0..L-1 over the MoE layers in model order, then the MTP layer when --mtp is on
                                  (GLM-5.3-Flash: model layers 3..44, then model layer 45 from mtp.safetensors as li 42;
                                  Qwen3.8-Flash-Next: model layers 0..47, then mtp.layers.0 as li 48)
  record bytes  [gate.trellis][up.trellis][down.trellis] [gate.suh][gate.svh][up.suh][up.svh][down.suh][down.svh] [pad]
  The trellis part (first `trellis_bytes`: 9,437,184 B for GLM-5.3-Flash 3.05bpw, 1,843,200 B for Qwen3.8-Flash-Next
  3.05bpw) is byte-identical to one VRAM slot of expert_cache.py (slot = gate | up @ off_u | down @ off_d, each 256-
  aligned; in both models every matrix is a multiple of 256 B, so the slot has no padding), so an NVMe -> RAM slot ->
  VRAM slot fill is one contiguous copy and the pointer tables become (a, a + off_u, a + off_d). A reader that keeps
  suh/svh resident reads only the trellis part (also 4096-aligned).
  The .mul1 codebook markers (one I32 scalar per matrix) are checked to be identical and recorded in the manifest.

Outputs in --out-dir: <name>.bin (records), <name>.json (layout + per-record sha256 of the source bytes).

Modes:
  pack            write the store (O_DIRECT, preallocated), hash every record from the SOURCE buffer before the write;
                  --layers / --limit pack only part of it (smoke test): the .bin is sparse at full size and the manifest
                  marks the part ("partial"; sha256 null for the records not written)
  verify          O_DIRECT re-read of every record, sha256 vs the manifest (independent of the page cache)
  cmp             byte-exact compare of --sample N random records (+ first/last of every layer) against fresh source reads
  diff            byte-exact compare with another store (--ref REF.bin, manifest next to it): the manifests' layout fields
                  and every record this store holds (all, or the packed part of a partial pack), O_DIRECT on both sides
"""
import argparse, concurrent.futures as cf, hashlib, json, mmap, os, random, re, struct, sys, time

ALIGN = 4096
MATS = ("gate_proj", "up_proj", "down_proj")
MTP_FILE = "mtp.safetensors"      # GLM-5.3-Flash keeps its MTP layer in a file of its own
KEY_RE = re.compile(r"(.*\.layers\.(\d+)\.mlp)\.experts\.(\d+)\.")
FORMAT = "glm53-expert-store-v1"
# manifest fields that define the store layout (diff compares them; format, model_dir and created are provenance)
LAYOUT_KEYS = ("record_bytes", "trellis_bytes", "align", "experts_per_layer", "n_layers", "order", "layers",
               "mtp_layer_li", "fields", "vram_slot", "mul1_markers_hex", "total_bytes")


def al(x, a=ALIGN):
    return (x + a - 1) // a * a


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return 8 + n, h


def is_mtp(fn, key):
    """MTP experts: GLM-5.3-Flash's mtp.safetensors, or keys under mtp. (Qwen3.8-Flash-Next, in the main shards)."""
    return fn == MTP_FILE or key.startswith("mtp.")


class Src:
    """Tensor locator over the checkpoint shards (read-only). layers = [(model layer, key prefix, experts)] in store
    order: the trunk's MoE layers by layer number, then the MTP layer(s) when use_mtp."""

    def __init__(self, model_dir, use_mtp):
        self.dir = model_dir
        idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        files = sorted(set(idx.values()))
        if use_mtp and MTP_FILE not in files and os.path.exists(os.path.join(model_dir, MTP_FILE)):
            files.append(MTP_FILE)
        self.loc = {}
        self.fds = {}
        mtp = set()
        for fn in files:
            base, h = st_header(os.path.join(model_dir, fn))
            for k, v in h.items():
                m = KEY_RE.match(k) if ".mlp.experts." in k else None
                if m is None or (is_mtp(fn, k) and not use_mtp):
                    continue
                a, b = v["data_offsets"]
                self.loc[k] = (fn, base + a, b - a, v["dtype"], tuple(v["shape"]))
                if is_mtp(fn, k):
                    mtp.add(m.group(1))
            fd = os.open(os.path.join(model_dir, fn), os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)   # tensor-sized preads: no readahead into the page cache
            self.fds[fn] = fd
        layers = {}
        for k in self.loc:
            m = KEY_RE.match(k)
            layers.setdefault(m.group(1), [int(m.group(2)), set()])[1].add(int(m.group(3)))
        order = sorted(layers, key=lambda p: (p in mtp, layers[p][0], p))
        self.layers = [(layers[p][0], p, len(layers[p][1])) for p in order]
        self.mtp_li = next((i for i, (_, p, _) in enumerate(self.layers) if p in mtp), None)

    def read(self, key, mv):
        fn, off, n, _, _ = self.loc[key]
        got = 0
        while got < n:
            r = os.preadv(self.fds[fn], [mv[got:n]], off + got)
            assert r > 0, (key, got, n)
            got += r
        return n

    def drop_cache(self, keys=None):
        """POSIX_FADV_DONTNEED on what was read: the byte ranges of `keys`, or every shard file."""
        if keys is None:
            for fd in self.fds.values():
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            return
        for k in keys:
            fn, off, n, _, _ = self.loc[k]
            os.posix_fadvise(self.fds[fn], off, n, os.POSIX_FADV_DONTNEED)


def layout(src):
    """Field list (key suffix, offset, bytes) for one record, from layer-0 expert-0 shapes (asserted for all)."""
    _, pfx, _ = src.layers[0]
    fields, off = [], 0
    for m in MATS:
        k = f"{pfx}.experts.0.{m}.trellis"
        n = src.loc[k][2]
        fields.append((f"{m}.trellis", off, n, src.loc[k][3], src.loc[k][4])); off += n
    trellis = off
    assert trellis % ALIGN == 0, trellis
    for m in MATS:
        for s in ("suh", "svh"):
            k = f"{pfx}.experts.0.{m}.{s}"
            n = src.loc[k][2]
            fields.append((f"{m}.{s}", off, n, src.loc[k][3], src.loc[k][4])); off += n
    return fields, trellis, al(off)


def check_uniform(src, fields, layers, experts):
    mul1 = set()
    for li in layers:
        _, pfx, _ = src.layers[li]
        for e in experts:
            for name, _, n, dt, shp in fields:
                v = src.loc[f"{pfx}.experts.{e}.{name}"]
                assert v[2] == n and v[3] == dt and v[4] == shp, (pfx, e, name, v, n, dt, shp)
            for m in MATS:
                k = f"{pfx}.experts.{e}.{m}.mul1"
                if k in src.loc:
                    fn, off, n, dt, _ = src.loc[k]
                    mul1.add(os.pread(src.fds[fn], n, off).hex())
    return sorted(mul1)


def fill_record(src, pfx, e, fields, buf):
    mv = memoryview(buf)
    for name, off, n, _, _ in fields:
        src.read(f"{pfx}.experts.{e}.{name}", mv[off:off + n])
    return mv


def build_record(src, li, e, fields, rec, buf=None):
    """Record (rec bytes, zero padded) of store layer li, expert e; buf: page-aligned scratch (O_DIRECT source)."""
    buf = buf if buf is not None else mmap.mmap(-1, rec)
    mv = fill_record(src, src.layers[li][1], e, fields, buf)
    pad0 = fields[-1][1] + fields[-1][2]
    if rec > pad0:
        mv[pad0:rec] = bytes(rec - pad0)
    return mv


def parse_layers(spec, L):
    """'0,47,-1' | '0-3' | '' -> sorted store layer indices (negative: from the end)."""
    if not spec or spec == "all":
        return list(range(L))
    out = set()
    for part in spec.split(","):
        a, sep, b = part.partition("-") if not part.startswith("-") else (part, "", "")
        out.update(range(int(a), int(b) + 1) if sep else [int(a) % L])
    assert all(0 <= x < L for x in out), (spec, L)
    return sorted(out)


def manifest(a, src, fields, trellis, rec, L, E, mul1, hashes, partial=None):
    meta = dict(format=FORMAT, model_dir=a.model, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                record_bytes=rec, trellis_bytes=trellis, align=ALIGN, experts_per_layer=E, n_layers=L,
                order="record = li * experts_per_layer + e (layer-major)",
                layers=[dict(li=i, model_layer=l, key=p) for i, (l, p, _) in enumerate(src.layers)],
                mtp_layer_li=src.mtp_li if a.mtp else None,
                fields=[dict(name=n, offset=o, bytes=b, dtype=d, shape=list(s)) for n, o, b, d, s in fields],
                vram_slot=dict(off_u=fields[1][1], off_d=fields[2][1], slot_bytes=trellis),
                mul1_markers_hex=mul1, sha256=hashes, total_bytes=L * E * rec)
    if partial is not None:
        meta["partial"] = partial
    return meta


def odirect_reader(path, rec):
    import threading
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    tl = threading.local()

    def rd(r):
        if not hasattr(tl, "buf"):
            tl.buf = mmap.mmap(-1, rec)
        n = os.preadv(fd, [tl.buf], r * rec)
        assert n == rec, (path, r, n)
        return tl.buf
    return rd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pack", "verify", "cmp", "diff"])
    ap.add_argument("--model", default=os.environ.get("GLM53_MODEL_DIR", "/models"), help="EXL3 checkpoint directory")
    ap.add_argument("--out-dir", default=os.environ.get("GLM53_NV_STORE_DIR", "/nvx"), help="store directory (NVMe, O_DIRECT capable)")
    ap.add_argument("--name", default="glm53_flash_exl3_3.05bpw_experts")
    ap.add_argument("--mtp", type=int, default=1, help="append the MTP layer's experts as the last layer")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=116)
    ap.add_argument("--layers", default="", help="pack: only these store layers, e.g. 0,47,-1 (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="pack: only the first N experts of each packed layer")
    ap.add_argument("--ref", default="", help="diff: the other store's .bin (its .json next to it)")
    a = ap.parse_args()
    out_bin = os.path.join(a.out_dir, a.name + ".bin")
    out_meta = os.path.join(a.out_dir, a.name + ".json")
    assert os.path.isdir(a.out_dir), f"--out-dir {a.out_dir} does not exist"
    assert os.path.realpath(a.out_dir) != os.path.realpath(a.model), "--out-dir must not be the checkpoint directory"
    t0 = time.time()
    if a.mode == "diff":
        sys.exit(diff(a, out_bin, out_meta, t0))
    src = Src(a.model, bool(a.mtp))

    if a.mode == "pack":
        fields, trellis, rec = layout(src)
        L = len(src.layers)
        E = src.layers[0][2]
        assert all(x[2] == E for x in src.layers)
        pl = parse_layers(a.layers, L)
        pe = list(range(min(a.limit, E) if a.limit > 0 else E))
        partial = dict(layers=pl, experts=len(pe)) if len(pl) < L or len(pe) < E else None
        mul1 = check_uniform(src, fields, pl, pe)
        total = L * E * rec
        size = len(pl) * len(pe) * rec
        os.makedirs(a.out_dir, exist_ok=True)
        st = os.statvfs(a.out_dir)
        free = st.f_bavail * st.f_frsize
        print(f"layers {L} ({src.layers[0][1]} .. {src.layers[-1][1]}), E {E}, rec {rec} B (trellis {trellis}), total "
              f"{total / 1e9:.2f} GB, free {free / 1e12:.2f} TB, mul1 markers {mul1}"
              + (f"; PARTIAL: layers {pl}, first {len(pe)} experts ({size / 1e9:.2f} GB)" if partial else ""), flush=True)
        assert size < free * 0.9, "not enough free space"
        fd = os.open(out_bin, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o600)   # owner only
        if partial:
            os.ftruncate(fd, total)                  # sparse: only the packed records take space
        else:
            os.posix_fallocate(fd, 0, total)
        hashes = [None] * (L * E)
        import threading
        tl = threading.local()

        def job(r):
            li, e = divmod(r, E)
            if not hasattr(tl, "buf"):
                tl.buf = mmap.mmap(-1, rec)          # page-aligned (O_DIRECT source)
            mv = build_record(src, li, e, fields, rec, tl.buf)
            h = hashlib.sha256(mv).hexdigest()
            n = os.pwrite(fd, mv, r * rec)
            assert n == rec, (r, n)
            hashes[r] = h
            return r

        done = 0
        with cf.ThreadPoolExecutor(a.threads) as ex:
            for li in pl:
                tl0 = time.time()
                list(ex.map(job, [li * E + e for e in pe]))
                done += len(pe)
                dt = time.time() - tl0
                print(f"  layer {li:2d} ({src.layers[li][1]}) {len(pe)} records {len(pe) * rec / 1e9:.2f} GB in {dt:.1f} s "
                      f"({len(pe) * rec / 1e9 / max(dt, 1e-6):.2f} GB/s), total {done * rec / 1e9:.1f} GB, {time.time() - t0:.0f} s",
                      flush=True)
        os.fsync(fd); os.close(fd)
        src.drop_cache([f"{src.layers[li][1]}.experts.{e}.{f[0]}" for li in pl for e in pe for f in fields] if partial else None)
        meta = manifest(a, src, fields, trellis, rec, L, E, mul1, hashes, partial)
        tmp = out_meta + ".tmp"
        json.dump(meta, open(tmp, "w"))
        os.replace(tmp, out_meta)
        print(f"PACKED {done} of {L * E} records, {done * rec / 1e9:.2f} GB in {time.time() - t0:.0f} s -> {out_bin}", flush=True)
        return

    meta = json.load(open(out_meta))
    rec, E, L = meta["record_bytes"], meta["experts_per_layer"], meta["n_layers"]
    assert os.path.getsize(out_bin) == meta["total_bytes"] == rec * E * L
    have = [r for r, h in enumerate(meta["sha256"]) if h is not None]   # every record, unless a partial pack
    rd = odirect_reader(out_bin, rec)

    if a.mode == "verify":
        bad = []

        def job(r):
            h = hashlib.sha256(rd(r)).hexdigest()
            if h != meta["sha256"][r]:
                bad.append(r)

        with cf.ThreadPoolExecutor(a.threads) as ex:
            list(ex.map(job, have))
        dt = time.time() - t0
        print(f"VERIFY {'OK' if not bad else 'FAIL'}: {len(have)} records, {len(have) * rec / 1e9:.2f} GB O_DIRECT re-read + "
              f"sha256 in {dt:.0f} s ({len(have) * rec / 1e9 / max(dt, 1e-6):.2f} GB/s); bad {len(bad)} {bad[:20]}", flush=True)
        sys.exit(1 if bad else 0)

    # cmp: byte-exact against fresh source reads (different code path from the packer's buffer reuse)
    fields = [(f["name"], f["offset"], f["bytes"]) for f in meta["fields"]]
    held = set(have)
    rng = random.Random(a.seed)
    pick = set(rng.sample(have, min(a.sample, len(have))))
    for li in range(L):
        pick.update(r for r in (li * E, li * E + E - 1) if r in held)
    pick = sorted(pick)
    bad = []
    for r in pick:
        li, e = divmod(r, E)
        pfx = meta["layers"][li]["key"]
        got = rd(r)
        for name, off, n in fields:
            fn, soff, sn, _, _ = src.loc[f"{pfx}.experts.{e}.{name}"]
            assert sn == n
            want = os.pread(src.fds[fn], n, soff)
            if got[off:off + n] != want:
                bad.append((r, name))
        tail = fields[-1][1] + fields[-1][2]
        if any(got[tail:rec]):
            bad.append((r, "pad"))
    print(f"CMP {'OK' if not bad else 'FAIL'}: {len(pick)} records byte-compared against the checkpoint (every field + "
          f"zero pad), {time.time() - t0:.0f} s; mismatches {len(bad)} {bad[:20]}", flush=True)
    sys.exit(1 if bad else 0)


def diff(a, out_bin, out_meta, t0):
    """This store vs --ref: the manifests' layout fields, then every record this store holds (O_DIRECT, both sides)."""
    assert a.ref, "diff needs --ref REF.bin"
    mine = json.load(open(out_meta))
    ref = json.load(open(a.ref[:-4] + ".json" if a.ref.endswith(".bin") else a.ref + ".json"))
    same = {k: mine.get(k) == ref.get(k) for k in LAYOUT_KEYS}
    for k in ("format", "model_dir", "created"):
        if mine.get(k) != ref.get(k):
            print(f"  provenance (not layout) differs: {k}: {mine.get(k)!r} vs ref {ref.get(k)!r}")
    for k, ok in same.items():
        if not ok:
            print(f"  LAYOUT DIFFERS: {k}: {json.dumps(mine.get(k))[:300]} vs ref {json.dumps(ref.get(k))[:300]}")
    rec = mine["record_bytes"]
    have = [r for r, h in enumerate(mine["sha256"]) if h is not None]
    sha_bad = [r for r in have if r >= len(ref["sha256"]) or ref["sha256"][r] != mine["sha256"][r]]
    if same["record_bytes"] and same["experts_per_layer"] and os.path.getsize(a.ref) >= os.path.getsize(out_bin):
        r_mine, r_ref = odirect_reader(out_bin, rec), odirect_reader(a.ref, rec)
        bad = [r for r in have if bytes(r_mine(r)) != bytes(r_ref(r))]
    else:
        bad = list(have)
    ok = all(same.values()) and not sha_bad and not bad
    print(f"DIFF {'OK' if ok else 'FAIL'}: layout fields {sum(same.values())}/{len(same)} equal; {len(have)} records "
          f"({len(have) * rec / 1e9:.2f} GB) byte-compared with {a.ref}: {len(have) - len(bad)} identical, {len(bad)} differ "
          f"{bad[:20]}; manifest sha256 equal for {len(have) - len(sha_bad)}/{len(have)}; {time.time() - t0:.0f} s", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    main()
