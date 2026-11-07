"""Minimal NVML binding (ctypes, no dependencies) for Windows and Linux, including WSL.

Only what the manager needs: device inventory, memory, utilization, PCIe link, display state and per-process usage.
Every call degrades to None/[] when the driver does not support it (WSL exposes no per-process data, for example).
"""
import ctypes
import os
import sys
from pathlib import Path

NVML_SUCCESS = 0
NOT_AVAILABLE = 0xFFFFFFFFFFFFFFFF


class _Mem(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class _Util(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _ProcV1(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong)]


class _ProcV2(ctypes.Structure):      # same layout for _v2 and _v3
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]


class _ProcUtil(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint), ("timeStamp", ctypes.c_ulonglong), ("smUtil", ctypes.c_uint),
                ("memUtil", ctypes.c_uint), ("encUtil", ctypes.c_uint), ("decUtil", ctypes.c_uint)]


def _load():
    if sys.platform == "win32":
        path = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvml.dll"
        return ctypes.WinDLL(str(path) if path.exists() else "nvml.dll")
    for name in ("libnvidia-ml.so.1", "/usr/lib/wsl/lib/libnvidia-ml.so.1", "libnvidia-ml.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise OSError("NVIDIA driver library (libnvidia-ml.so.1) not found")


class Device:
    def __init__(self, lib, index):
        self.lib, self.index = lib, index
        self.h = ctypes.c_void_p()
        if lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.h)) != NVML_SUCCESS:
            raise OSError(f"no NVIDIA GPU at index {index}")
        buf = ctypes.create_string_buffer(96)
        lib.nvmlDeviceGetName(self.h, buf, 96)
        self.name = buf.value.decode(errors="replace")
        self._util_ts = 0

    def _uint(self, fn, *pre):
        v = ctypes.c_uint()
        f = getattr(self.lib, fn, None)
        if f is None or f(self.h, *pre, ctypes.byref(v)) != NVML_SUCCESS:
            return None
        return v.value

    def memory(self):
        m = _Mem()
        self.lib.nvmlDeviceGetMemoryInfo(self.h, ctypes.byref(m))
        return float(m.total), float(m.used)

    def utilization(self):
        u = _Util()
        if self.lib.nvmlDeviceGetUtilizationRates(self.h, ctypes.byref(u)) != NVML_SUCCESS:
            return 0.0
        return float(u.gpu)

    def compute_capability(self):
        a, b = ctypes.c_int(), ctypes.c_int()
        if self.lib.nvmlDeviceGetCudaComputeCapability(self.h, ctypes.byref(a), ctypes.byref(b)) != NVML_SUCCESS:
            return None
        return f"{a.value}.{b.value}"

    def info(self):
        """Static facts for hardware detection."""
        total, _used = self.memory()
        disp = self._uint("nvmlDeviceGetDisplayActive")
        return {
            "index": self.index, "name": self.name, "vram_total": total,
            "compute_capability": self.compute_capability(),
            # the link drops to Gen1 at idle to save power; the max values are what a load gets
            "pcie_gen_max": self._uint("nvmlDeviceGetMaxPcieLinkGeneration"),
            "pcie_gen_gpu_max": self._uint("nvmlDeviceGetGpuMaxPcieLinkGeneration"),
            "pcie_width_max": self._uint("nvmlDeviceGetMaxPcieLinkWidth"),
            "pcie_gen_now": self._uint("nvmlDeviceGetCurrPcieLinkGeneration"),
            "pcie_width_now": self._uint("nvmlDeviceGetCurrPcieLinkWidth"),
            "display_active": None if disp is None else bool(disp),
            "temperature_c": self._uint("nvmlDeviceGetTemperature", ctypes.c_int(0)),
            "power_limit_w": (lambda v: None if v is None else round(v / 1000))(
                self._uint("nvmlDeviceGetPowerManagementLimit")),
        }

    def _procs(self, base):
        for suffix, struct in (("_v3", _ProcV2), ("_v2", _ProcV2), ("", _ProcV1)):
            f = getattr(self.lib, base + suffix, None)
            if f is None:
                continue
            n = ctypes.c_uint(0)
            arr = (struct * 128)()
            n.value = 128
            if f(self.h, ctypes.byref(n), arr) != NVML_SUCCESS:
                return []
            return [(p.pid, None if p.usedGpuMemory == NOT_AVAILABLE else float(p.usedGpuMemory))
                    for p in arr[:n.value]]
        return []

    def processes(self):
        """{pid: bytes or None} for compute and graphics processes. Empty under WSL."""
        out = {}
        for base in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
            for pid, used in self._procs(base):
                if used is None:
                    out.setdefault(pid, None)
                else:
                    out[pid] = max(out.get(pid) or 0.0, used)
        return out

    def process_utilization(self):
        """{pid: sm %} over the last sample window, or None if the GPU/driver does not support it."""
        f = getattr(self.lib, "nvmlDeviceGetProcessUtilization", None)
        if f is None:
            return None
        n = ctypes.c_uint(256)
        arr = (_ProcUtil * 256)()
        r = f(self.h, arr, ctypes.byref(n), ctypes.c_ulonglong(self._util_ts))
        if r == 6:          # NVML_ERROR_NOT_FOUND: no samples since last call
            return {}
        if r != NVML_SUCCESS:
            return None
        out = {}
        for s in arr[:n.value]:
            out[s.pid] = max(out.get(s.pid, 0), s.smUtil)
            self._util_ts = max(self._util_ts, s.timeStamp)
        return out


class Nvml:
    def __init__(self):
        self.lib = _load()
        if self.lib.nvmlInit_v2() != NVML_SUCCESS:
            raise OSError("nvmlInit failed (is the NVIDIA driver installed?)")
        n = ctypes.c_uint()
        self.lib.nvmlDeviceGetCount_v2(ctypes.byref(n))
        self.devices = [Device(self.lib, i) for i in range(n.value)]
        if not self.devices:
            raise OSError("no NVIDIA GPU found")

    def driver_version(self):
        buf = ctypes.create_string_buffer(96)
        if self.lib.nvmlSystemGetDriverVersion(buf, 96) != NVML_SUCCESS:
            return None
        return buf.value.decode()

    def cuda_driver_version(self):
        v = ctypes.c_int()
        if self.lib.nvmlSystemGetCudaDriverVersion_v2(ctypes.byref(v)) != NVML_SUCCESS:
            return None
        return f"{v.value // 1000}.{v.value % 1000 // 10}"
