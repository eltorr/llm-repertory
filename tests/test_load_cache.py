"""exllamav3's loader drops what it has read from the page cache (exllamav3/exllamav3/loader/page_cache.py)."""
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exllamav3" / "exllamav3" / "loader"))
import page_cache  # noqa: E402

MB = 1024 ** 2


class LoadCacheTests(unittest.TestCase):
    def test_nearby_reads_merge_and_files_stay_apart(self):
        rows = [(0, 10 * MB, MB), (0, 0, 100), (0, 200, 300), (1, 0, 50), (0, 11 * MB + 512, 10)]
        self.assertEqual(page_cache.read_spans(rows), [(0, 0, 500), (0, 10 * MB, 11 * MB + 522), (1, 0, 50)])

    def test_a_gap_the_loader_would_read_through_is_dropped_with_its_neighbours(self):
        self.assertEqual(page_cache.read_spans([(0, 0, 100), (0, 100 + MB, 1)]), [(0, 0, 101 + MB)])
        self.assertEqual(len(page_cache.read_spans([(0, 0, 100), (0, 101 + MB, 1)])), 2)

    def test_only_the_spans_read_are_dropped(self):
        calls = []
        with TemporaryDirectory() as d:
            a, b = Path(d) / "a.safetensors", Path(d) / "b.safetensors"
            a.write_bytes(b"x")
            b.write_bytes(b"y")
            page_cache.drop_read([str(a), str(b), str(Path(d) / "gone")], [(0, 0, 100), (1, 4096, 10), (2, 0, 1),
                                                                            (0, 50 * MB, 1)],
                                 fadvise=lambda fd, off, n, adv: calls.append((off, n, adv)))
        dn = getattr(os, "POSIX_FADV_DONTNEED", 4)
        self.assertEqual(calls, [(0, 100, dn), (50 * MB, 1, dn), (4096, 10, dn)])     # the missing file is skipped
        page_cache.drop_read(["x"], [], fadvise=lambda *a: calls.append(a))           # nothing read: nothing to do
        self.assertEqual(len(calls), 3)

    @unittest.skipUnless(hasattr(os, "posix_fadvise"), "Linux")
    def test_the_real_call_works(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "w.safetensors"
            p.write_bytes(b"z" * 65536)
            p.read_bytes()
            page_cache.drop_read([str(p)], [(0, 0, 65536)])


if __name__ == "__main__":
    unittest.main()
