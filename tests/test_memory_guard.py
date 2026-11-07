"""The model server's watchdogs and VRAM figures (engines/exl3_offload/memory_guard.py), on a fake memory cgroup."""
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

HERE = Path(__file__).resolve().parents[1] / "engines" / "exl3_offload"
sys.path.insert(0, str(HERE))
import memory_guard  # noqa: E402

GIB = 1024 ** 3


class RamGuardTests(unittest.TestCase):
    def test_it_measures_the_cgroup_and_keeps_the_peak(self):
        with TemporaryDirectory() as d:
            cg = Path(d)
            (cg / "memory.current").write_text("123")
            guard = memory_guard.RamGuard(cg, limit=10 * GIB)
            try:
                self.assertEqual(guard.sample(), 123)
                (cg / "memory.current").write_text("456")
                self.assertEqual(guard.sample(), 456)
                (cg / "memory.current").write_text("200")
                guard.sample()
                self.assertEqual(guard.peak, 456)
                self.assertEqual(guard.path, cg)
            finally:
                guard.close()

    def test_without_a_memory_cgroup_it_is_off(self):
        with TemporaryDirectory() as d:
            guard = memory_guard.RamGuard(d, limit=1)
            self.assertIsNone(guard._thread)                    # nothing to watch, so nothing stops the server
            self.assertEqual(guard.sample(), 0)
            guard.close()

    def test_no_limit_only_measures(self):
        with TemporaryDirectory() as d, mock.patch.dict("os.environ", {"QWEN_RAM_LIMIT_BYTES": ""}):
            (Path(d) / "memory.current").write_text(str(64 * GIB))
            guard = memory_guard.RamGuard(d)
            self.assertIsNone(guard.limit)
            guard.close()

    def test_over_its_limit_it_stops_the_server_with_a_message_the_manager_reads(self):
        with TemporaryDirectory() as d:
            (Path(d) / "memory.current").write_text("2000")
            code = ("import sys, time; sys.path.insert(0, sys.argv[1]); import memory_guard; "
                    "memory_guard.RamGuard(cgroup=sys.argv[2], limit=1999); time.sleep(10)")
            r = subprocess.run([sys.executable, "-c", code, str(HERE), d], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, memory_guard.EXIT_RAM, r.stderr)
        self.assertIn("GUARD ram: 2,000 bytes in use, over the 1,999-byte limit", r.stdout)

    def test_the_limit_can_move_while_it_runs(self):
        # as the RAM cache's live resize moves it: under the limit at first, over it once it comes down
        with TemporaryDirectory() as d:
            (Path(d) / "memory.current").write_text("2000")
            code = ("import sys, time; sys.path.insert(0, sys.argv[1]); import memory_guard; "
                    "g = memory_guard.RamGuard(cgroup=sys.argv[2], limit=5000); time.sleep(0.3); "
                    "print('under', flush=True); g.limit = 1500; time.sleep(10)")
            r = subprocess.run([sys.executable, "-c", code, str(HERE), d], capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, memory_guard.EXIT_RAM, r.stdout + r.stderr)
        self.assertEqual(r.stdout.splitlines()[0], "under")
        self.assertIn("over the 1,500-byte limit", r.stdout)


class VramFiguresTests(unittest.TestCase):
    def test_what_cuda_does_not_see(self):
        # NVML: 9 GiB in use on the device; CUDA: 32 GiB total, 24 GiB free, so it sees 8 GiB in use
        with mock.patch.object(memory_guard, "device_used", lambda index=0: 9 * GIB):
            self.assertEqual(memory_guard.unseen_vram(24 * GIB, 32 * GIB), 1 * GIB)
        with mock.patch.object(memory_guard, "device_used", lambda index=0: 7 * GIB):
            self.assertEqual(memory_guard.unseen_vram(24 * GIB, 32 * GIB), 0)
        with mock.patch.object(memory_guard, "device_used", lambda index=0: None):
            self.assertIsNone(memory_guard.unseen_vram(24 * GIB, 32 * GIB))

    def test_the_torch_cap(self):
        self.assertEqual(memory_guard.torch_limit_bytes(32 * GIB, {"QWEN_TORCH_LIMIT_BYTES": "16500000000"}),
                         16_500_000_000)
        self.assertEqual(memory_guard.torch_limit_bytes(16 * GIB, {"QWEN_TORCH_LIMIT_BYTES": "4e10"}), 16 * GIB)
        self.assertIsNone(memory_guard.torch_limit_bytes(32 * GIB, {}))

    def test_without_nvml(self):
        if sys.platform != "win32":
            self.skipTest("NVML may be present here")
        self.assertIsNone(memory_guard.device_used())
        with self.assertRaises(OSError):
            memory_guard.VramGuard()


if __name__ == "__main__":
    unittest.main()
