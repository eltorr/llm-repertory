"""Host-to-GPU copy speed under WSL2, for the engine's RAM -> VRAM expert copies: 2.46 MB experts, one at a time
(as nv2 copies them) and batched, from torch-pinned memory and from cudaHostAlloc(Portable|Mapped) memory (what
glm53/wsl_pinned.py hands the RAM tier under WSL2). Uses ~0.5 GB of host memory and ~0.25 GB of VRAM."""
import ctypes
import time

import torch

EXPERT = 2_457_600
N = 96
dev = torch.device("cuda:0")
dst = torch.empty(N * EXPERT, dtype=torch.uint8, device=dev)
s = torch.cuda.Stream()


def timed(fn, reps=20):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


def run(src, name):
    def one_by_one():
        with torch.cuda.stream(s):
            for i in range(N):
                dst[i * EXPERT:(i + 1) * EXPERT].copy_(src[i * EXPERT:(i + 1) * EXPERT], non_blocking=True)

    def batched():
        with torch.cuda.stream(s):
            dst.copy_(src, non_blocking=True)

    for label, fn in (("one expert at a time", one_by_one), ("all at once", batched)):
        t = timed(fn)
        print(f"{name:34s} {label:22s} {N * EXPERT / t / 1e9:6.1f} GB/s   ({t / N * 1e6:6.1f} us per expert)")


pinned = torch.empty(N * EXPERT, dtype=torch.uint8).pin_memory()
run(pinned, "torch pin_memory")

rt = ctypes.CDLL(next(line.split()[-1] for line in open("/proc/self/maps") if "/libcudart.so" in line))
p = ctypes.c_void_p()
assert rt.cudaHostAlloc(ctypes.byref(p), ctypes.c_size_t(N * EXPERT), 1 | 2) == 0
buf = (ctypes.c_uint8 * (N * EXPERT)).from_address(p.value)
mapped = torch.frombuffer(buf, dtype=torch.uint8)
mapped.fill_(1)
run(mapped, "cudaHostAlloc(Portable|Mapped)")
rt.cudaFreeHost(p)
print("GPU:", torch.cuda.get_device_name(), "| PCIe from nvidia-smi below")
