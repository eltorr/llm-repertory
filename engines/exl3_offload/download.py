"""Downloads one quant of the model from Hugging Face, at a pinned revision, and checks every file.

    python engines/exl3_offload/download.py --model-dir ./model [--branch 3.05bpw_h5_ng5] [--sha REVISION] [--repo R]

- **Pinned:** each model comes from one upload, so a later upload to its branch can't change or mix an install. Setup
  passes the catalog's pin (catalog/catalog.json: source.huggingface and source.revision) as --repo and --sha; PINNED
  below is the same pins, for running this by hand with only --branch. To move to a newer upload, test it, then
  change the revision in the catalog (and here).
- **Checked:** every file is compared with the hash the Hub lists for it (sha256 for large files, the git blob id
  for the rest) before it takes its name. Files checked once are noted in .verified.json, so a rerun is quick.
- **Resumable:** a file in progress is NAME.part, with NAME.part.json listing the pieces it has; a rerun fetches the
  rest. A file whose pieces were all in when a run stopped is checked and finished first.
- **Parallel:** pieces of PIECE bytes, fetched by WORKERS threads, the biggest files first, so the 30+ GB n-gram
  table doesn't end up as one slow tail.
- **Light on RAM:** written data is dropped from the page cache as it lands (Linux). A page cache full of ~100 GB of
  downloads would otherwise crowd out the model, and under WSL count as used memory for Windows.

Prints VERIFY / COMPLETE for each file, PROGRESS now and then, and DONE at the end (setup/50_model.sh shows them).
Needs huggingface_hub (in setup/requirements.txt).
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

REPO = "turboderp/Qwen3.8-Flash-Next-exl3"   # the default --repo; setup passes the catalog's
# the catalog's pins for Qwen's two quants, kept for a run by hand with only --branch (a test checks they match)
PINNED = {"3.05bpw_h5_ng5": "69e33439ae950f17bcbe95c98f117d80f759ab6d",
          "4.05bpw_h6_ng6": "55a732e0c4c3d4614bc42b68493bb930d9b02c0a"}
PIECE = 32 * 1024 ** 2
WORKERS = 16
TRIES = 6
TIMEOUT_S = 90
GB = 1e9


class Failed(SystemExit):
    """A download that can't go on; the message says what to do."""


def check(ok, message):
    # explicit checks, not asserts: python -O would drop those
    if not ok:
        raise Failed(message)


# ------------------------------------------------------------------------------------------------------------- files
class Remote:
    """One file of the upload: its name, size, and the hash the Hub lists for it."""

    def __init__(self, name, size, digest, blob):
        path = PurePosixPath(name)
        check(name and not path.is_absolute() and ".." not in path.parts and "\\" not in name,
              f"refusing a file outside the model folder: {name!r}")
        self.name, self.size, self.digest, self.blob = name, int(size or 0), digest, blob

    @classmethod
    def from_hub(cls, sibling):
        if sibling.lfs:
            return cls(sibling.rfilename, sibling.size, sibling.lfs.sha256, False)
        return cls(sibling.rfilename, sibling.size, sibling.blob_id, True)

    def pieces(self):
        return range((self.size + PIECE - 1) // PIECE)

    def span(self, index):
        """The byte range of piece `index`, both ends included."""
        start = index * PIECE
        return start, min(self.size, start + PIECE) - 1


def file_digest(path, blob):
    """sha256 of the bytes, or for a small file kept in git, its blob id (sha1 of a header and the bytes)."""
    # git names a file by this sha1: a fixed format to compare against, not a choice of hash for security
    h = (hashlib.sha1(b"blob %d\0" % Path(path).stat().st_size, usedforsecurity=False) if blob
         else hashlib.sha256())
    with open(path, "rb") as f:
        while block := f.read(16 * 1024 ** 2):
            h.update(block)
    return h.hexdigest()


def upload(branch, sha, repo=REPO):
    """(revision, [Remote]) of the pinned upload."""
    from huggingface_hub import HfApi
    revision = sha or PINNED.get(branch)
    check(revision, f"no pinned revision for {branch}; pass --sha")
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    check(info.sha == revision, f"the Hub answered with revision {info.sha}, not {revision}")
    return revision, [Remote.from_hub(s) for s in info.siblings]


class Verified:
    """The files already checked against their hash (.verified.json in the model folder)."""

    def __init__(self, folder):
        self.path = Path(folder) / ".verified.json"
        try:
            self.done = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.done = {}

    def has(self, remote):
        return self.done.get(remote.name) == remote.digest

    def add(self, remote):
        self.done[remote.name] = remote.digest
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.done, indent=1))
        tmp.replace(self.path)


# ------------------------------------------------------------------------------------------------------- in progress
class Partial:
    """A file being downloaded: NAME.part (its final size from the start) and NAME.part.json, the pieces it has."""

    def __init__(self, folder, remote):
        self.remote = remote
        self.final = Path(folder) / remote.name
        self.part = self.final.with_name(self.final.name + ".part")
        self.state = self.final.with_name(self.final.name + ".part.json")
        try:
            self.have = set(json.loads(self.state.read_text()))
        except (OSError, ValueError):
            self.have = set()
        if not self.part.exists():
            self.have = set()                      # the pieces list outlived its data: start over
        self.lock = threading.Lock()
        self.fd = None

    def todo(self):
        return [i for i in self.remote.pieces() if i not in self.have]

    def open(self):
        self.final.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.part, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o600)   # owner only
        os.ftruncate(self.fd, self.remote.size)

    def write(self, offset, data):
        """Write `data` at `offset`, and let the page cache drop it once it is on disk."""
        if hasattr(os, "pwrite"):
            done = 0
            while done < len(data):
                done += os.pwrite(self.fd, data[done:], offset + done)
        else:                                       # Windows: no pwrite (tests only; installs run on Linux)
            with self.lock:
                os.lseek(self.fd, offset, os.SEEK_SET)
                os.write(self.fd, data)
        if hasattr(os, "posix_fadvise"):
            os.fdatasync(self.fd)
            os.posix_fadvise(self.fd, offset, len(data), os.POSIX_FADV_DONTNEED)

    def got(self, index):
        """Piece `index` is written. True when it was the last one."""
        with self.lock:
            self.have.add(index)
            tmp = self.state.with_suffix(".tmp")
            tmp.write_text(json.dumps(sorted(self.have)))
            tmp.replace(self.state)
            return len(self.have) == len(self.remote.pieces())

    def finish(self, verified):
        """Every piece is in: check the hash and give the file its name. False (the part deleted) on a mismatch."""
        if self.fd is not None:
            os.fsync(self.fd)
            os.close(self.fd)
            self.fd = None
        if file_digest(self.part, self.remote.blob) != self.remote.digest:
            self.part.unlink()
            self.state.unlink(missing_ok=True)
            return False
        self.part.replace(self.final)
        self.state.unlink(missing_ok=True)
        verified.add(self.remote)
        print("COMPLETE", self.remote.name, flush=True)
        return True


def fetch(url, start, end, size):
    """Bytes start..end of the file at `url`, retried; the server must answer with exactly that range."""
    check(urllib.parse.urlsplit(url).scheme in ("https", "http"), f"refusing a URL that isn't HTTP(S): {url[:80]}")
    want = f"bytes {start}-{end}/{size}"
    for attempt in range(TRIES):
        try:
            ask = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(ask, timeout=TIMEOUT_S) as reply:
                got = reply.headers.get("Content-Range")
                check(reply.status == 206 and got == want,
                      f"unexpected answer to a range request ({reply.status}, {got})")
                data = reply.read()
            if len(data) != end - start + 1:
                raise OSError(f"short read: {len(data)} of {end - start + 1} bytes")
            return data
        except (OSError, Failed):
            if attempt == TRIES - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return b""


# -------------------------------------------------------------------------------------------------------------- main
def plan(folder, files, verified):
    """The files to fetch, as Partials; existing files are checked (those not yet known to match)."""
    todo = []
    for remote in files:
        path = Path(folder) / remote.name
        if path.exists():
            if not verified.has(remote):
                print("VERIFY", remote.name, flush=True)
                check(file_digest(path, remote.blob) == remote.digest,
                      f"{path} does not match the pinned upload; delete it and run this again")
                verified.add(remote)
            continue
        todo.append(Partial(folder, remote))
    return todo


def download(folder, revision, files, repo=REPO):
    folder.mkdir(parents=True, exist_ok=True)
    verified = Verified(folder)
    partials = plan(folder, files, verified)
    for p in [p for p in partials if not p.todo()]:          # all pieces in when an earlier run stopped
        check(p.finish(verified), f"{p.remote.name} does not match the pinned upload (deleted; run this again)")
    partials = sorted((p for p in partials if p.todo()), key=lambda p: -p.remote.size)
    total = sum(p.remote.size for p in partials)
    left = sum(len(p.todo()) for p in partials)
    print(f"fetching {len(partials)} files, {total / GB:.1f} GB ({left} pieces)", flush=True)
    for p in partials:
        p.open()
    jobs = [(p, i) for p in partials for i in p.todo()]
    t0, fetched, lock = time.monotonic(), [0], threading.Lock()

    def one(job):
        p, index = job
        start, end = p.remote.span(index)
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{p.remote.name}"
        p.write(start, fetch(url, start, end, p.remote.size))
        with lock:
            fetched[0] += end - start + 1
        return p, p.got(index)

    with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
        for n, future in enumerate(concurrent.futures.as_completed([pool.submit(one, j) for j in jobs]), 1):
            p, last = future.result()
            if last and not p.finish(verified):
                pool.shutdown(wait=False, cancel_futures=True)
                raise Failed(f"{p.remote.name} does not match the pinned upload (deleted; run this again)")
            if n % 64 == 0 or n == len(jobs):
                s = time.monotonic() - t0
                print(f"PROGRESS {fetched[0] / GB:.1f} of {total / GB:.1f} GB, {fetched[0] / GB / max(s, 1e-3):.2f} "
                      f"GB/s", flush=True)
    print(f"DONE {len(files)} files at revision {revision}, {fetched[0] / GB:.1f} GB fetched in "
          f"{time.monotonic() - t0:.0f} s", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-dir", type=Path, required=True, help="where the model goes, e.g. ./model")
    p.add_argument("--branch", default="3.05bpw_h5_ng5", help="the quant's branch (default 3.05bpw_h5_ng5)")
    p.add_argument("--sha", help="the revision to fetch (default: the one pinned for --branch)")
    p.add_argument("--repo", default=REPO, help=f"the Hugging Face repository (default {REPO})")
    a = p.parse_args(argv)
    check(re.fullmatch(r"[\w.-]+/[\w.-]+", a.repo) and (a.sha is None or re.fullmatch(r"[0-9a-f]{40}", a.sha)),
          f"bad --repo or --sha: {a.repo} {a.sha}")
    revision, files = upload(a.branch, a.sha, a.repo)
    print("REVISION", a.branch, revision, flush=True)
    download(a.model_dir.expanduser(), revision, files, a.repo)


if __name__ == "__main__":
    main()
