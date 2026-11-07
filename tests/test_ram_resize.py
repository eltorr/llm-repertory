"""The RAM tier's live shrink plan (glm53-flash-offload/kernels/nv2/ram_resize.h), built with the system's C++
compiler and run (tests/test_ram_resize.cpp). Skipped where there is no compiler (a Windows checkout)."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CXX = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), None)
HEADER = ROOT / "glm53-flash-offload" / "kernels" / "nv2" / "ram_resize.h"


@unittest.skipUnless(CXX, "no C++ compiler")
@unittest.skipUnless(HEADER.exists(), "the engine has no ram_resize.h yet (Repertory's RAM resize is not in it)")
class RamShrinkPlanTests(unittest.TestCase):
    def test_plan(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "test_ram_resize"
            subprocess.run([CXX, "-std=c++17", "-O1", "-Wall", "-Werror",
                            "-I", str(ROOT / "glm53-flash-offload" / "kernels" / "nv2"),
                            str(ROOT / "tests" / "test_ram_resize.cpp"), "-o", str(exe)], check=True)
            r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual(r.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
