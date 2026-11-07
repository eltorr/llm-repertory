"""Hold GPU memory, so the engine sees a smaller card (benchmarks) or the manager sees "another app" (pressure tests).

Uses the CUDA driver API through ctypes (nvcuda.dll on Windows, libcuda.so.1 on Linux/WSL): no toolkit or torch.

  python vram_ballast.py --hold-gb 10              hold 10 GB (grabs it as it becomes free, 1 GB at a time)
  python vram_ballast.py --leave-gb 11             hold everything except 11 GB of the card
  python vram_ballast.py --leave-gb 11 --for 600   ... and release after 600 s
Prints BALLAST_READY held=<GB> free=<GB> once the target is reached; Ctrl+C (or the timeout) releases.

Note: under WSL, Windows' per-process GPU counters do not see Linux processes, so a ballast meant to look like
"another app" to the Windows manager must run on Windows.
"""
import argparse
import ctypes
import sys
import time

GB = 1e9
CHUNK = 1 << 30


def load():
    if sys.platform == "win32":
        return ctypes.WinDLL("nvcuda.dll")
    for name in ("libcuda.so.1", "/usr/lib/wsl/lib/libcuda.so.1"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise SystemExit("CUDA driver library not found")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--hold-gb", type=float)
    g.add_argument("--leave-gb", type=float, help="leave this much of the card's total memory to everything else")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--for", dest="seconds", type=float, default=0, help="release after this many seconds")
    ap.add_argument("--wait", type=float, default=600, help="give up growing after this many seconds")
    a = ap.parse_args()
    cu = load()
    assert cu.cuInit(0) == 0, "cuInit failed"
    dev, ctx = ctypes.c_int(), ctypes.c_void_p()
    assert cu.cuDeviceGet(ctypes.byref(dev), a.device) == 0
    assert cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev) == 0, "cuCtxCreate failed"
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    cu.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
    if a.hold_gb is not None:
        target = a.hold_gb * GB
    else:
        # everything the card has beyond leave_gb, counting what other processes (and our context) already use
        target = total.value - a.leave_gb * GB - (total.value - free.value)
    target = max(0.0, target)
    ptrs, held, t0 = [], 0, time.time()
    while held + CHUNK <= target + CHUNK / 2 and time.time() - t0 < a.wait:
        p = ctypes.c_uint64()
        size = int(min(CHUNK, max(64 << 20, target - held)))
        if cu.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(size)) == 0:
            ptrs.append(p)
            held += size
        else:
            time.sleep(1)           # not free yet (e.g. the model is scaling down): keep trying
    cu.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
    print(f"BALLAST_READY held={held / GB:.1f} free={free.value / GB:.1f} total={total.value / GB:.1f}", flush=True)
    try:
        end = time.time() + a.seconds if a.seconds else None
        while end is None or time.time() < end:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    for p in ptrs:
        cu.cuMemFree_v2(p)
    print("BALLAST_RELEASED", flush=True)


if __name__ == "__main__":
    main()
