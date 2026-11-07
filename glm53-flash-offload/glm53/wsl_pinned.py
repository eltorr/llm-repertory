"""Local patch (WSL2 support): pinned + GPU-mapped host memory whose device address equals its host address.

The engine writes raw host addresses into tables that GPU kernels dereference (zero-copy over PCIe), so it needs
UVA alias == host address. On bare-metal Linux, anon mmap + cudaHostRegister(Mapped) gives that. Under WSL2 the
driver reports cudaDevAttrCanUseHostPointerForRegisteredMem == 0 and the registered alias differs, but memory from
cudaHostAlloc(Portable|Mapped) is still identity-mapped. needed() decides; alloc() returns a buffer-protocol object
(usable with torch.frombuffer exactly like the mmap it replaces) plus its address.

GLM53_PINNED_HOSTALLOC=1/0 forces the choice.
"""
import ctypes
import os

import torch

_RT = None
_KEEP = {}          # page-aligned address -> what cudaHostAlloc returned; live until free(), like the mmaps they replace
_NEEDED = {}
_ATTR_CAN_USE_HOST_PTR_FOR_REGISTERED = 91
_HOST_ALLOC_PORTABLE, _HOST_ALLOC_MAPPED = 1, 2


def _rt():
    """The CUDA runtime torch loaded (libcudart.so.13 for CUDA 13 builds): a hard-coded libcudart.so.12 loaded a second
    runtime, or failed where only CUDA 13 is installed."""
    global _RT
    if _RT is None:
        torch.cuda.init()
        path = None
        try:
            with open("/proc/self/maps") as f:
                path = next((line.split()[-1] for line in f if "/libcudart.so" in line), None)
        except OSError:
            pass
        _RT = ctypes.CDLL(path or f"libcudart.so.{torch.version.cuda.split('.')[0]}")
    return _RT


def needed(dev_idx=0):
    forced = os.environ.get("GLM53_PINNED_HOSTALLOC")
    if forced in ("0", "1"):
        return forced == "1"
    if dev_idx not in _NEEDED:
        v = ctypes.c_int(0)
        r = _rt().cudaDeviceGetAttribute(ctypes.byref(v), _ATTR_CAN_USE_HOST_PTR_FOR_REGISTERED, int(dev_idx))
        _NEEDED[dev_idx] = (r == 0 and v.value == 0)
    return _NEEDED[dev_idx]


def alloc(nbytes, dev_idx, tag, zero=True):
    """Zero-filled (zero=False: as the driver hands it over), page-aligned, pinned+mapped host region. Returns
    (buffer, address)."""
    nbytes = (int(nbytes) + 4095) // 4096 * 4096

    def host_alloc(n):
        p = ctypes.c_void_p()
        with torch.cuda.device(dev_idx):
            r = _rt().cudaHostAlloc(ctypes.byref(p), ctypes.c_size_t(n), _HOST_ALLOC_PORTABLE | _HOST_ALLOC_MAPPED)
        assert r == 0 and p.value, f"cudaHostAlloc failed ({tag}, {n} B): {r}"
        return p.value
    base = host_alloc(nbytes)
    if base % 4096:
        # cudaHostAlloc doesn't promise page alignment: small requests can come from a shared pool. Ask for a page
        # more and use the first page boundary in it (the whole allocation is identity-mapped, so is that).
        _free(base, dev_idx)
        base = host_alloc(nbytes + 4096)
    a = (base + 4095) // 4096 * 4096
    if zero:
        ctypes.memset(a, 0, nbytes)   # anon mmap is zero-filled; keep that guarantee
    _KEEP[a] = base
    return (ctypes.c_uint8 * nbytes).from_address(a), a


def _free(base, dev_idx):
    with torch.cuda.device(dev_idx):
        r = _rt().cudaFreeHost(ctypes.c_void_p(base))
    assert r == 0, f"cudaFreeHost failed: {r}"


def free(addr, dev_idx):
    """Give back a region from alloc() (nv2's live RAM resize). Nothing may use it any more."""
    _free(_KEEP.pop(addr), dev_idx)
