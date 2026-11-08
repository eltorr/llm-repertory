"""Bytes a model load has read leave the page cache once their tensors are in place. Stdlib only.

Nothing reads them again, and under WSL2 the VM's page cache is memory Windows counts as used: a model's non-expert
weights are gigabytes of it, held through the rest of the load. Only the ranges read are dropped (with the small gaps
the C++ loader reads through), so a file that is also streamed from (an n-gram table) keeps the rows it has cached.
Linux only; EXL3_LOAD_KEEP_CACHE=1 keeps them.
"""
import os

_DROP_CACHE = hasattr(os, "posix_fadvise") and os.environ.get("EXL3_LOAD_KEEP_CACHE", "0") != "1"
_DROP_GAP = 1024**2


def read_spans(rows, gap: int = _DROP_GAP) -> list[tuple[int, int, int]]:
    """(file, start, end) spans covering rows of (file, offset, size), rows within gap bytes of each other in the
    same file merged."""
    out = []
    for f, off, size in sorted(rows):
        if out and out[-1][0] == f and off <= out[-1][2] + gap:
            out[-1][2] = max(out[-1][2], off + size)
        else:
            out.append([f, off, off + size])
    return [tuple(x) for x in out]


def drop_read(filenames, rows, fadvise = None):
    """DONTNEED on the spans of rows (file index into filenames, offset, size) that a load has read."""
    fadvise = fadvise or (getattr(os, "posix_fadvise", None) if _DROP_CACHE else None)
    if fadvise is None or not len(rows):
        return
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", 4)
    fds = {}
    try:
        for f, beg, end in read_spans(rows):
            if f not in fds:
                try:
                    fds[f] = os.open(filenames[f], os.O_RDONLY)
                except OSError:
                    fds[f] = None
            if fds[f] is not None:
                try:
                    fadvise(fds[f], beg, end - beg, dontneed)
                except OSError:
                    pass
    finally:
        for fd in fds.values():
            if fd is not None:
                os.close(fd)
