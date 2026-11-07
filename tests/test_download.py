"""The model download (engines/exl3_offload/download.py) without the network: file checks, resuming, hashes."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engines" / "exl3_offload"))
import download as D  # noqa: E402


def remote(name, data, blob=False):
    digest = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() if blob else hashlib.sha256(data).hexdigest()
    return D.Remote(name, len(data), digest, blob)


class FileTests(unittest.TestCase):
    def test_names_outside_the_model_folder_are_refused(self):
        for name in ("../evil", "/etc/passwd", "a/../../b", "sub\\..\\x", ""):
            with self.subTest(name=name), self.assertRaises(D.Failed):
                D.Remote(name, 1, "x", False)
        self.assertEqual(D.Remote("sub/config.json", 1, "x", False).name, "sub/config.json")

    def test_digests_match_the_hubs(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f"
            p.write_bytes(b"hello\n")
            self.assertEqual(D.file_digest(p, blob=True), "ce013625030ba8dba906f756967f9e9ca394464a")   # git's id
            self.assertEqual(D.file_digest(p, blob=False), hashlib.sha256(b"hello\n").hexdigest())

    def test_pieces_cover_the_file_exactly(self):
        with mock.patch.object(D, "PIECE", 10):
            r = D.Remote("x", 25, "d", False)
            self.assertEqual([r.span(i) for i in r.pieces()], [(0, 9), (10, 19), (20, 24)])
            self.assertEqual(list(D.Remote("e", 0, "d", False).pieces()), [])


class ResumeTests(unittest.TestCase):
    DATA = bytes(range(256)) * 3

    def piece(self, p, i):
        start, end = p.remote.span(i)
        p.write(start, self.DATA[start:end + 1])
        return p.got(i)

    def test_a_file_resumes_where_it_stopped_and_is_checked_before_it_takes_its_name(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(D, "PIECE", 100):
            folder, r = Path(d), remote("weights.safetensors", self.DATA)
            p = D.Partial(folder, r)
            p.open()
            for i in (0, 3, 5):
                self.assertFalse(self.piece(p, i))
            p.fd = D.os.close(p.fd)                                  # the run stops
            again = D.Partial(folder, r)
            self.assertEqual(again.todo(), [1, 2, 4, 6, 7])           # only what is missing
            again.open()
            last = [self.piece(again, i) for i in again.todo()]
            self.assertEqual(last, [False, False, False, False, True])
            verified = D.Verified(folder)
            self.assertTrue(again.finish(verified))
            self.assertEqual((folder / r.name).read_bytes(), self.DATA)
            self.assertFalse(again.part.exists() or again.state.exists())
            self.assertTrue(D.Verified(folder).has(r))                # noted, so a rerun doesn't hash it again

    def test_a_mismatch_is_deleted_not_kept(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(D, "PIECE", 1000):
            r = D.Remote("x.bin", len(self.DATA), "0" * 64, False)
            p = D.Partial(Path(d), r)
            p.open()
            self.assertTrue(self.piece(p, 0))
            self.assertFalse(p.finish(D.Verified(d)))
            self.assertFalse((Path(d) / "x.bin").exists() or p.part.exists())

    def test_a_pieces_list_without_its_data_starts_over(self):
        with tempfile.TemporaryDirectory() as d:
            r = remote("y.bin", self.DATA)
            (Path(d) / "y.bin.part.json").write_text(json.dumps([0]))
            self.assertEqual(D.Partial(Path(d), r).todo(), list(r.pieces()))


class PlanTests(unittest.TestCase):
    def test_existing_files_are_checked_once_and_wrong_ones_refused(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            good, small = remote("model.safetensors", b"weights"), remote("config.json", b"{}", blob=True)
            (folder / "model.safetensors").write_bytes(b"weights")
            (folder / "config.json").write_bytes(b"{}")
            todo = D.plan(folder, [good, small, remote("new.bin", b"n")], D.Verified(folder))
            self.assertEqual([p.remote.name for p in todo], ["new.bin"])
            self.assertTrue(D.Verified(folder).has(good) and D.Verified(folder).has(small))
            (folder / "config.json").write_bytes(b"{changed}")
            with mock.patch.object(D, "file_digest", side_effect=AssertionError("hashed again")):
                D.plan(folder, [good], D.Verified(folder))            # known to match: not hashed again
            with self.assertRaises(D.Failed):
                D.plan(folder, [remote("config.json", b"[]", blob=True)], D.Verified(folder))

    def test_every_quant_is_pinned(self):
        for branch, sha in D.PINNED.items():
            self.assertRegex(sha, r"^[0-9a-f]{40}$", branch)


if __name__ == "__main__":
    unittest.main()
