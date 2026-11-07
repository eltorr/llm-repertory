"""nv2's CPU lane kernels at the geometry of each supported model, against the fp64 reference (cpu_lane_check.cpp):
GLM-5.3-Flash (hidden 4096, expert width 2048) and Qwen3.8-Flash-Next (2560, 640: the down projection's 160 output
tiles are not a multiple of n135's 128-tile column unit). CPU only: needs g++ and an AVX2 + FMA + F16C CPU."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNELS = HERE.parent / "kernels" / "nv2"


def avx2():
    try:
        flags = open("/proc/cpuinfo").read()
    except OSError:
        return False
    return all(f" {f}" in flags for f in ("avx2", "fma", "f16c"))


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("g++") and avx2(), "needs Linux, g++ and AVX2")
class CpuLane(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="cpu_lane_")
        cls.exe = os.path.join(cls.tmp, "cpu_lane_check")
        subprocess.run(["g++", "-O2", "-std=c++17", "-mavx2", "-mfma", "-mf16c", "-pthread", "-I", str(KERNELS),
                        str(HERE / "cpu_lane_check.cpp"), "-o", cls.exe], check=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_case(self, H, I, E):
        r = subprocess.run([self.exe, str(H), str(I), str(E), "2"], capture_output=True, text=True, timeout=900)
        print(r.stdout, end="")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stdout.strip().splitlines()[-1], "ok")

    def test_qwen38_geometry(self):
        self.run_case(2560, 640, 6)

    def test_glm53_geometry(self):
        self.run_case(4096, 2048, 2)


if __name__ == "__main__":
    unittest.main()
