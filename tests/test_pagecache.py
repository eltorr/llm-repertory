"""The engine's page-cache drop before the RAM tier is allocated (glm53-flash-offload/glm53/pagecache.py)."""
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "glm53-flash-offload" / "glm53"))
import pagecache  # noqa: E402


class PageCacheTests(unittest.TestCase):
    def test_the_files_the_launcher_pre_reads(self):
        with TemporaryDirectory() as d:
            names = ["a.so", "libcublas.so.13", "m.py", "m.pyc", "data.bin", "README.md", "x.safetensors"]
            for n in names:
                (Path(d) / "sub" / n).parent.mkdir(exist_ok=True)
                (Path(d) / "sub" / n).write_bytes(b"x")
            got = sorted(Path(p).name for p in pagecache.preread_files([d, os.path.join(d, "missing")]))
        self.assertEqual(got, ["a.so", "libcublas.so.13", "m.py", "m.pyc"])

    def test_each_file_is_dropped_once_and_missing_ones_are_skipped(self):
        calls = []
        with TemporaryDirectory() as d:
            a, b = Path(d) / "a.safetensors", Path(d) / "b.safetensors"
            a.write_bytes(b"x" * 4096)
            b.write_bytes(b"y")
            n = pagecache.drop([str(a), str(a), str(b), str(Path(d) / "gone")],
                               fadvise=lambda fd, off, ln, adv: calls.append((off, ln, adv)))
        self.assertEqual(n, 2)
        self.assertEqual(calls, [(0, 0, pagecache.DONTNEED)] * 2)       # the whole file
        self.assertEqual(pagecache.drop(["x"], fadvise=None), 0)        # not Linux: nothing to do

    @unittest.skipUnless(hasattr(os, "posix_fadvise"), "Linux")
    def test_the_real_call_works(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "w.safetensors"
            p.write_bytes(b"z" * 65536)
            p.read_bytes()
            self.assertEqual(pagecache.drop([str(p)]), 1)

    def test_page_cache_is_cached_less_shared_memory(self):
        with TemporaryDirectory() as d:
            m = Path(d) / "meminfo"
            m.write_text("MemTotal:  57583716 kB\nCached:    37877408 kB\nShmem:     37330096 kB\nHugePages_Total: 0\n")
            self.assertEqual(pagecache.file_cache_bytes(str(m)), (37877408 - 37330096) * 1024)
            m.write_text("garbage\n")
            self.assertIsNone(pagecache.file_cache_bytes(str(m)))
        self.assertIsNone(pagecache.file_cache_bytes(os.path.join("no", "such", "meminfo")))


if __name__ == "__main__":
    unittest.main()
