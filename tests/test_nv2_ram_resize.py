"""nv2's live RAM resize in its real host engine (kernels/nv2/nv2_host.cpp: ram_shrink, ram_grow) without a GPU: a
test engine over plain host memory. Needs PyTorch and the compiled nv2 extension, i.e. the engine's environment
(GLM53_NV2_SO names another build of it); skipped elsewhere. The engine is one per process, so the checks run in a
child process."""
import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SO = Path(os.environ.get("GLM53_NV2_SO") or ROOT / "glm53-flash-offload" / "build" / "nv2" / "glm53_nv2.so")
SLOT, L, E = 4096, 4, 16


def load():
    import torch  # noqa: F401  (the extension needs torch's libraries loaded first)
    spec = importlib.util.spec_from_file_location("glm53_nv2", SO)
    ext = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ext)
    return ext


def child():
    import torch
    ext = load()
    nram = 40
    mem = torch.zeros(nram * SLOT, dtype=torch.uint8)
    base = mem.data_ptr()
    addrs = torch.tensor([base + i * SLOT for i in range(nram)], dtype=torch.int64)
    ext.ram_test_engine(L, E, SLOT, addrs[:24])

    def land(keys):
        for k in keys:
            s = ext.ram_test_land(k)
            o = s * SLOT
            mem[o:o + SLOT] = k % 251 + 1                  # each expert's bytes say which it is

    def check(kept, dropped, n):
        so, ks, pinc, owner = ext.state()
        lru, home, res = ext.ram_test_lists()
        assert owner.numel() == n and pinc.numel() == n, (owner.numel(), n)
        assert lru.tolist() == kept, (lru.tolist(), kept)     # same order, oldest first
        for k in kept:
            s = int(so[k])
            assert 0 <= s < n and int(ks[k]) == 2 and int(owner[s]) == k, (k, s)
            assert int(home[k]) == base + s * SLOT and int(res[k]) == 1, k
            o = s * SLOT
            assert bool((mem[o:o + SLOT] == k % 251 + 1).all()), f"key {k}: wrong bytes in slot {s}"
        for k in dropped:
            assert int(so[k]) == -1 and int(ks[k]) == 0 and int(home[k]) == 0 and int(res[k]) == 0, k
        assert len(set(int(so[k]) for k in kept)) == len(kept)

    land(range(20))                                   # 20 of 24 slots, keys 0..19, oldest first
    dropped, moved, resident = ext.ram_shrink(8, 1.0, 4)
    assert (dropped, resident) == (12, 8), (dropped, moved, resident)
    assert moved == 8, moved                          # landed in slots 0..19 in order: the newest 8 sat at 12..19
    check(list(range(12, 20)), list(range(12)), 8)
    assert ext.ram_shrink(8, 1.0, 4) == [0, 0, 8]     # no change
    assert ext.ram_grow(addrs[8:nram]) == nram        # the slots after the kept ones, as new chunks would be
    check(list(range(12, 20)), list(range(12)), nram)
    land(range(30, 52))                               # 22 more, into the new space: nothing evicted
    kept = list(range(12, 20)) + list(range(30, 52))
    check(kept, list(range(12)), nram)
    dropped, moved, resident = ext.ram_shrink(17, 1.0, 3)
    assert (dropped, resident) == (13, 17), (dropped, moved, resident)
    check(kept[13:], kept[:13] + list(range(12)), 17)
    dropped, moved, resident = ext.ram_shrink(0, 1.0, 2)
    assert (dropped, moved, resident) == (17, 0, 0)
    check([], kept, 0)
    # growing from another thread while experts keep landing (requests running): every landing gets its own slot
    import threading
    cap = ext.ram_capacity()
    assert cap == L * E + 1024, cap
    big = torch.zeros(400 * SLOT, dtype=torch.uint8)
    b2 = big.data_ptr()
    grower = threading.Thread(target=lambda: [ext.ram_grow(torch.tensor([b2 + (i * 8 + j) * SLOT for j in range(8)],
                                                                          dtype=torch.int64)) for i in range(50)])
    landed = []
    grower.start()
    while grower.is_alive() or len(landed) < 60:
        k = len(landed)
        if k >= L * E:
            break
        so, ks, _, owner = ext.state()
        if owner.numel() > k:                     # a free slot exists: land the next key
            ext.ram_test_land(k)
            landed.append(k)
    grower.join()
    so, ks, pinc, owner = ext.state()
    assert owner.numel() == 400, owner.numel()
    slots = [int(so[k]) for k in landed]
    assert all(int(ks[k]) == 2 for k in landed) and len(set(slots)) == len(slots) and max(slots) < 400
    try:                                         # more slots than experts: refused, nothing changed
        ext.ram_grow(torch.zeros(cap, dtype=torch.int64))
        raise AssertionError("ram_grow past the capacity was accepted")
    except RuntimeError as e:
        assert "more RAM slots than experts" in str(e), e
    assert ext.state()[3].numel() == 400
    print("ok")


def runnable():
    if not SO.is_file():
        return False
    try:
        return hasattr(load(), "ram_test_engine")
    except Exception:  # noqa: BLE001 - no torch here, or a build for another PyTorch
        return False


@unittest.skipUnless(runnable(), f"needs PyTorch and an nv2 build with the RAM test engine ({SO})")
class Nv2RamResizeTests(unittest.TestCase):
    def test_shrink_keeps_the_newest_with_their_bytes_and_grow_adds_space(self):
        r = subprocess.run([sys.executable, __file__, "--child"], capture_output=True, text=True, timeout=120,
                           env={**os.environ, "GLM53_NV2_SO": str(SO)})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stdout.strip().splitlines()[-1], "ok")


def has_torch():
    try:
        import numpy  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(has_torch(), "needs PyTorch (the engine's environment)")
class HostArenaTests(unittest.TestCase):
    """nv_tier.HostArena's chunks as the live RAM resize changes them; plain memory instead of pinned."""

    def arena(self, n, fail_after=None):
        import ctypes
        import mmap
        sys.path.insert(0, str(ROOT / "glm53-flash-offload" / "glm53"))
        import nv_tier

        class Plain(nv_tier.HostArena):
            def _alloc(self, k):
                if fail_after is not None and len(self.maps) >= fail_after:
                    raise AssertionError("cudaHostAlloc failed")
                mm = mmap.mmap(-1, k * self.slot)
                return mm, ctypes.addressof(ctypes.c_char.from_buffer(mm))

            def _free(self, mm, a):
                self.freed.append(a)
        Plain.freed = []
        return Plain(n, SLOT, 128, 0, "ram")

    def test_chunks_boundaries_shrink_and_grow(self):
        a = self.arena(300)
        self.assertEqual((a.n, a.sizes, a.starts, len(a.addr)), (300, [128, 128, 44], [0, 128, 256], 300))
        for s in (0, 127, 128, 299):
            self.assertEqual(a.view(s).data_ptr(), int(a.addr[s]))
        self.assertEqual([a.boundary_at_or_below(x) for x in (0, 5, 128, 200, 256, 299, 300, 1000)],
                         [0, 0, 128, 128, 256, 256, 300, 300])
        with self.assertRaises(AssertionError):
            a.shrink_to(200)                               # not a chunk boundary
        last = a.bases[1:]
        a.shrink_to(128)
        self.assertEqual((a.n, a.sizes, len(a.addr), a.freed), (128, [128], 128, last[::-1]))
        new = a.grow(200)
        self.assertEqual((a.n, a.sizes, a.starts, len(new)), (328, [128, 128, 72], [0, 128, 256], 200))
        self.assertEqual(list(a.addr[128:]), new)
        self.assertEqual(a.view(300).data_ptr(), int(a.addr[300]))

    def test_a_shrinks_chunks_can_be_freed_later(self):
        a = self.arena(300)
        b1, b2 = a.bases[1], a.bases[2]
        chunks = a.detach_above(128)
        self.assertEqual((a.n, a.sizes, a.freed), (128, [128], []))       # out of the arena, not freed yet
        a.free_chunks(chunks)
        self.assertEqual(a.freed, [b2, b1])                                # the highest first

    def test_chunks_taken_out_go_back_where_they_were(self):
        a = self.arena(300)
        addr0 = list(a.addr)
        chunks = a.detach_above(128)
        self.assertEqual([(b, k) for _, b, k in chunks], [(addr0[128], 128), (addr0[256], 44)])   # in order
        new = a.attach(chunks)
        self.assertEqual((a.n, a.sizes, a.starts, list(a.addr), new), (300, [128, 128, 44], [0, 128, 256], addr0,
                                                                         addr0[128:]))
        self.assertEqual((a.view(299).data_ptr(), a.freed), (int(a.addr[299]), []))     # nothing allocated or freed

    def test_a_grow_that_runs_out_of_pinned_memory_keeps_what_it_got(self):
        a = self.arena(100, fail_after=2)
        self.assertEqual(len(a.grow(400, strict=False)), 128)
        self.assertEqual((a.n, a.sizes), (228, [100, 128]))
        with self.assertRaises(AssertionError):
            a.grow(10)


if __name__ == "__main__":
    if sys.argv[1:] == ["--child"]:
        child()
    else:
        unittest.main()
