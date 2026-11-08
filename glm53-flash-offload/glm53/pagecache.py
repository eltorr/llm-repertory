"""Hand clean page cache back to the OS: posix_fadvise(DONTNEED) on whole files. Pure Python (no torch).

Under WSL2 the VM's page cache is memory Windows counts as used, and the VM gives it up only once it is full itself. A
model load reads the engine's libraries (the launcher's pre-read) and the checkpoint's non-expert weights through it;
dropping both before the RAM tier is allocated keeps them from sitting on top of the tier at the load's peak. DONTNEED
leaves pages a process maps (the code of the loaded libraries) and dirty pages alone.
"""
import os

DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", 4)


def preread_files(dirs):
    """The files under dirs that the launcher pre-reads (engines/exl3_offload/run.sh): *.so, *.so.*, *.py, *.pyc."""
    out = []
    for d in dirs:
        for root, _, names in os.walk(d):
            out += [os.path.join(root, n) for n in names if n.endswith((".so", ".py", ".pyc")) or ".so." in n]
    return out


def drop(paths, fadvise=getattr(os, "posix_fadvise", None)):
    """DONTNEED on each file; missing or unreadable ones are skipped. Returns how many files it was done on."""
    if fadvise is None:
        return 0
    n = 0
    for p in sorted(set(paths)):
        try:
            fd = os.open(p, os.O_RDONLY)
        except OSError:
            continue
        try:
            fadvise(fd, 0, 0, DONTNEED)
            n += 1
        except OSError:
            pass
        finally:
            os.close(fd)
    return n


def file_cache_bytes(meminfo="/proc/meminfo"):
    """Page cache held for files: Cached less shared memory (which the RAM tier is). None without /proc/meminfo."""
    try:
        with open(meminfo) as f:
            kb = {k.strip(): int(v.split()[0]) for k, v in (line.split(":", 1) for line in f if ":" in line)}
        return (kb["Cached"] - kb.get("Shmem", 0)) * 1024
    except (OSError, KeyError, ValueError, IndexError):
        return None
