"""The model server's RAM and VRAM watchdogs, and the VRAM figures its loader needs.

Two watchdogs run beside the model, each a thread that reads one figure every POLL_S seconds:
  RamGuard    the memory cgroup's RAM (memory.current, page cache included), against QWEN_RAM_LIMIT_BYTES
  VramGuard   the GPU's VRAM as NVML counts it (every process: under WSL2 also what Windows apps hold, which CUDA
              doesn't see), against QWEN_VRAM_GUARD_BYTES
Past its limit a watchdog prints a "GUARD ram" or "GUARD vram" line, which the manager reads as running out of that
memory, and ends the server before the kernel or the driver ends it in the middle of a request. Without the variable
a watchdog only measures (RAM) or holds the server to the whole card (VRAM).

For loading (server.py): unseen_vram() is what CUDA can't see of the VRAM in use, which the expert cache counts
against its VRAM limit, and torch_limit_bytes() the cap for this process's torch allocator.
"""
import ctypes
import os
import sys
import threading
from pathlib import Path

POLL_S = 0.02
EXIT_RAM, EXIT_VRAM = 75, 76          # the server's exit status when a watchdog stops it


class _MemoryInfo(ctypes.Structure):          # nvmlMemory_t
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


def _nvml_device(index):
    """(NVML library, device handle), or None where there is no NVIDIA driver (or no Linux)."""
    try:
        lib = ctypes.CDLL("libnvidia-ml.so.1")
    except OSError:
        return None
    handle = ctypes.c_void_p()
    if lib.nvmlInit_v2() != 0 or lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) != 0:
        return None
    return lib, handle


def _nvml_memory(device):
    info = _MemoryInfo()
    lib, handle = device
    if lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(info)) != 0:
        raise OSError("NVML could not read the device's memory")
    return info


def device_used(index=0):
    """VRAM in use on the device as NVML counts it (every process: what VramGuard watches), or None without NVML."""
    device = _nvml_device(index)
    if device is None:
        return None
    try:
        return _nvml_memory(device).used
    except OSError:
        return None


def unseen_vram(cuda_free, cuda_total, index=0):
    """VRAM other processes hold that CUDA doesn't report to this one: NVML's figure for the device minus what
    cudaMemGetInfo calls used (cuda_total - cuda_free). Under WSL2 that is part of what Windows apps use, about 1 GB
    with a desktop and a browser open; elsewhere about 0. None without NVML. Measure it before loading: after that,
    this process's own memory is in both figures and the difference no longer means anything."""
    used = device_used(index)
    if used is None:
        return None
    return max(0, used - (cuda_total - cuda_free))


def torch_limit_bytes(card_bytes, environ=None):
    """The cap for this process's torch allocator: QWEN_TORCH_LIMIT_BYTES (run.sh and the manager set it to the
    model's VRAM budget plus 0.5 GB), never more than the card. None when it isn't set: no cap."""
    value = (os.environ if environ is None else environ).get("QWEN_TORCH_LIMIT_BYTES")
    if not value:
        return None
    return min(int(float(value)), int(card_bytes))


def _limit(name, default=None):
    value = os.environ.get(name)
    return int(float(value)) if value else default


class Watchdog:
    """Reads a figure (bytes) every POLL_S seconds and keeps its peak; past `limit` (None: never) it ends the process
    with `code`. `limit` may change while it runs (the RAM cache's live resize moves it)."""

    what = "memory"
    code = 1

    def __init__(self, limit):
        self.limit = limit
        self.peak = 0
        self._stop = threading.Event()
        self._thread = None

    def read(self):
        raise NotImplementedError

    def start(self):
        self._thread = threading.Thread(target=self._watch, name=f"{self.what}-guard", daemon=True)
        self._thread.start()

    def sample(self):
        value = self.read()
        self.peak = max(self.peak, value)
        return value

    def _watch(self):
        while not self._stop.is_set():
            value = self.sample()
            if self.limit is not None and value > self.limit:
                self.trip(value)
            self._stop.wait(POLL_S)

    def details(self):
        return ""

    def trip(self, value):
        print(f"GUARD {self.what}: {value:,} bytes in use, over the {self.limit:,}-byte limit; stopping the server"
              f"{self.details()}", flush=True)
        os._exit(self.code)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.sample()


class RamGuard(Watchdog):
    """The memory cgroup at `cgroup`: run.sh or bin/repertory-backend start the server in a systemd scope or unit of
    its own, with MemoryHigh/MemoryMax a little above the RAM cache and this limit just below them."""

    what, code = "ram", EXIT_RAM

    def __init__(self, cgroup, limit=None):
        super().__init__(_limit("QWEN_RAM_LIMIT_BYTES") if limit is None else limit)
        self.path = Path(cgroup)
        self.current = self.path / "memory.current"
        if not self.current.exists():
            # e.g. WSL's root cgroup, or a desktop session without a delegated memory controller
            print(f"RAM guard off: no memory.current in {self.path} (start inside a memory cgroup to enable it)",
                  flush=True)
            return
        self.start()

    def read(self):
        try:
            return int(self.current.read_text())
        except (OSError, ValueError):
            return 0


class VramGuard(Watchdog):
    """GPU `index` as NVML counts it. OSError without NVML: the server needs the GPU anyway."""

    what, code = "vram", EXIT_VRAM

    def __init__(self, limit=None, index=0):
        self.device = _nvml_device(index)
        if self.device is None:
            raise OSError("NVML is not available: no NVIDIA driver?")
        if limit is None:
            limit = _limit("QWEN_VRAM_GUARD_BYTES", _nvml_memory(self.device).total)
        super().__init__(limit)
        self.start()

    def read(self):
        return _nvml_memory(self.device).used

    def details(self):
        torch = sys.modules.get("torch")
        if torch is None or not torch.cuda.is_initialized():
            return ""
        return (f" (torch: {torch.cuda.memory_allocated():,} bytes allocated, {torch.cuda.memory_reserved():,} "
                f"reserved)")
