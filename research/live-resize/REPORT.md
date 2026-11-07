# Switching full <-> lean without a restart

When a game needs VRAM, the manager used to restart the model in the lean profile: about 45 s without it, and the
caches started empty. The engine now resizes its VRAM expert cache in place between requests
(`POST /expert_cache/resize`, [manager guide](../../docs/manager.md#how-it-shares-the-gpu)). This run checks that
it's fast, that it frees and takes back the VRAM, and that it doesn't change the answers.

- **Machine:** RTX 5090 (32 GB), 64 GB RAM, Windows 11 with the engine in WSL2, driver 617.14, PyTorch 2.15.0 on
  CUDA 13.2, commit `e1569fd`.
- **Settings:** 4.05 bpw, whole card, 256K context, 34 GiB RAM cache.
- **Method:** `repertory bench current --workload resize`. The model starts full. At each size it answers one
  prompt twice (greedy, to the natural end), reads a 32K-token document of its own, and runs the engine's cache
  check (`/nv_verify`: 64 VRAM and 64 RAM experts against the store, plus the lookup tables). It switches full ->
  lean -> full, then cycles lean <-> full three more times. The switch requests are the ones the manager sends.

## Results

| At | Experts in VRAM | VRAM in use | Answer, then again | 32K document | Expert picks VRAM / RAM / NVMe | Cache check |
|---|---|---|---|---|---|---|
| Full (after the start) | 7,264 | 29.5 GB | 56.4, 65.8 tok/s | 3,174 tok/s | 93 / 5 / 2 % | ok |
| Lean | 1,824 | 16.1 GB | 35.3, 36.9 tok/s | 2,004 tok/s | 82 / 17 / 1 % | ok |
| Full again | 7,200 | 29.3 GB | 62.0, 67.9 tok/s | 4,082 tok/s | 85 / 14 / 1 % | ok |
| Full after 3 more cycles | 7,200 | 29.3 GB | 65.4, 65.5 tok/s | 4,081 tok/s | 87 / 12 / 1 % | ok |

| Switch | Time | Experts in VRAM | Written back to RAM | VRAM in use |
|---|---|---|---|---|
| Full -> lean | 0.50 s | 7,264 -> 1,824 | 5,439 | 29.5 -> 16.1 GB |
| Lean -> full | 0.26 s | 1,824 -> 7,200 | 0 | 16.1 -> 29.2 GB |
| Full -> lean | 0.50 s | 7,200 -> 1,824 | 5,318 | 29.3 -> 16.1 GB |
| 3 more cycles | 0.19-0.28 s | 7,200 <-> 1,824 | 0 | 29.2 <-> 16.1 GB |

- **The answers were identical** at every size, and every cache check passed.
- **Lean frees 13.3 GB**, and full regrows to within 0.22 GB of the cache it started with (7,200 experts instead of
  7,264: buffers the engine kept since the start now count as in use).
- **Shrinking takes about 0.5 s** when it writes experts back to the RAM cache (those that answers had loaded only
  into VRAM), and about 0.2 s when there are none. Growing takes 0.25 s: the new space starts empty and fills as
  answers need experts.
- **Lean keeps the full profile's 34 GiB RAM cache.** A lean restart gets 8 GiB, and both caches start empty.

## Notes

- The first full answer varies between runs (56.4 tok/s here, 60.2 in the earlier one), and the benchmark's
  comparison with the earlier run (-6.3%) uses it. The second answers matched (65.8 and 65.1).
- The first full document is read from cold caches (3,174 tok/s); the later ones read at 4,082 tok/s.
- Lean holds 1,824 experts in VRAM, but the plan showed 2,272. From a 16 GB budget, `sizing.py` assumed a 0.96-1.7 GB
  reserve where the engine keeps its own 1.7 GiB. Fixed the same day: the plan now says 1,920 for lean and 7,264 for
  full, which is what full loaded. The live shrink lands 96 below the lean plan, because buffers the engine
  allocated since the start now count as in use.
- **An earlier run** (`results/4.05-card-256k-8k-shared-document.json`) gave every size the same 8K document. After
  the first read, it came from the prompt cache (12,888-17,250 tok/s), so it never tested a long prompt at lean,
  where the VRAM cache is lent to prefill. Its switch times, VRAM and checks match this run's.

## The RAM cache too (later the same day)

The RAM cache resizes in place as well now (`live_ram_resize`). Same machine and settings, commit `5d361ab`, a
35 GiB RAM cache. The workload ran the VRAM part above first, with the same results (answers identical, 13.1 GB
freed, 0.2-0.5 s a switch). Then it shrank the RAM cache to lean's 8 GiB with the VRAM cache, grew it back, and
cycled twice more.

| Switch | Time | RAM cache | Experts |
|---|---|---|---|
| Full -> lean, first | 3.9 s | 35 -> 7.9 GiB (15,291 -> 3,456 experts) | 2,774 moved into the memory that stays, 11,835 dropped |
| Lean -> full, first | 19.6 s | 7.9 -> 35 GiB | the new space starts empty |
| 2 more cycles | shrink 3.3-4.1 s, grow 10.5-11.8 s | the same | |

| At | Answer, then again | 32K document | Cache check |
|---|---|---|---|
| Lean, lean's RAM cache | 22.7, 22.6 tok/s | 975 tok/s | ok |
| Full, RAM cache regrown | 43.6, 66.5 tok/s | 2,858 tok/s | ok |
| Full after 2 more cycles | 44.6, 66.7 tok/s | 2,850 tok/s | ok |

Windows' free RAM after the first shrink: +0.2 GB at once, +0.0 at 5 s, -0.5 at 15 s, +13.5 at 30 s and +29.5 GB
at 60 s.

- **The answers were identical** at every size, and every cache check passed.
- **Windows gets the RAM back 15-60 s after the shrink**, all 27 GiB within a minute (the engine compacts free
  memory after a shrink under WSL). A restart frees it the same way, so a game gets it no later than after one,
  and the model keeps answering meanwhile.
- **Growing takes 10-20 s:** it pins 27 GiB of new memory. The first grow, after Windows had taken the memory
  back, took the longest. Requests wait meanwhile. A restart takes about 45 s, and both caches start empty.
- **After a grow the RAM cache refills from NVMe:** the first answer runs at 44 tok/s and the second at 66.5;
  documents read at 2,850 tok/s instead of 4,100-4,270 until it has.
- **Lean with 8 GiB of RAM cache** answers at 22.7 tok/s, against 36-37 with full's: the cost of handing the RAM
  back. A lean restart gets the same 8 GiB.
- **The memory limits followed the cache:** the engine started in a cgroup with a 54G MemoryMax, and the RAM guard
  went from 52.6 to 25.5 GiB at each shrink and back at each grow.
- **Startup is unchanged** at 45 s with the RAM cache in chunks of 128 experts.

Follow-ups (done below): grow without holding requests, and skip zero-filling the new chunks.

## The RAM cache in the background

The engine now grows the RAM cache on a thread of its own, a chunk at a time, while requests run; a shrink holds
requests only while the experts it keeps move, and frees the rest in the background. Same machine and settings.

**The first stress run passed every check but found the real problem.** A request sent as the cache began to grow
(commit `db4d7e0`) had its first token after 12.1 s, when the 12.3 s grow ended. Under WSL the CUDA driver
serializes pinning memory (about 0.13 s a 0.3 GB chunk) with the model's GPU work, and the grow left 2 ms between
chunks. That run had no check for it. Since commit `99a99b7`, while a request runs or waits a grow adds only 32
experts (~75 MB) a second and goes at full speed between requests; a free hands back a chunk every 0.1 s while
one runs. The stress workload now fails a run where such a request waits more than 4 s.

Two stress runs on `99a99b7` (the second with that check) passed every check:

| Variation | First token | Decode | Notes |
|---|---|---|---|
| An answer and a 32K document while the cache grows | 1.6-1.7 s (12.1 s before) | 36-38 tok/s | the document read at 2,339 tok/s while it grew; the grow took 50 s, 23 small steps while requests ran |
| A shrink while it grows | | | the grow stopped at 8,192-8,960 experts; the shrink took 0.2-0.3 s, nothing to move |
| A grow stopped partway, then finished | | | stopped at 5,760-6,400 of 15,291 experts, then grown to all of them |
| 4 quick switches back and forth | | | 0.2-0.4 s a switch; one right after a shrink took 1.6-2.2 s, waiting for its free with requests running |
| Requests around a shrink, and while its memory goes back | 2.3 s at lean (1.7-1.8 s cold) | 16.5 tok/s (20.6 after) | the free took 13.6 s with a request running, about 3 s without |

After every variation the answers matched the first, the cache check passed, the engine's RAM stayed under its
guard (43.5 of 56.5 GB at full, 14.3 of 27.4 GB at lean), the limits followed the cache, and Windows had at
least 10.4 GB free.

The resize workload on `99a99b7`, as above:

| Switch | Time | RAM cache |
|---|---|---|
| Full -> lean, shrinking it | 0.84-0.86 s, holding requests (3.3-4.1 s before) | 35 -> 7.9 GiB; the kept experts moved in 0.37-0.43 s |
| Lean -> full | 0.24-0.38 s | grown in the background: 27.6 s with an answer running, 9.0-10.4 s without (10.5-19.6 s holding requests before) |

| At | Answer, then again | 32K document | Cache check |
|---|---|---|---|
| Lean, lean's RAM cache | 22.9, 23.1 tok/s | 972 tok/s | ok |
| While the cache grows | 36.3 tok/s, first token after 1.6 s | | |
| Full, RAM cache regrown | 66.2, 66.5 tok/s | 2,753 tok/s | ok |
| Full after 2 more cycles | 45.2, 66.7 tok/s | 2,855 tok/s | ok |

- **The VRAM part matched the earlier runs:** answers identical, 13.3 GB freed, 0.2-0.5 s a switch.
- **Requests no longer wait for a grow.** One sent as it starts runs on a cache that is still filling, at 36
  tok/s; before, it waited 10-20 s and then ran at 44. Unless the answer runs past about 2,000 tokens, that
  finishes sooner, and it streams from the start.
- **The grow takes longer while requests run**, up to a minute here, and finishes between them. The manager's
  brake still stops it if Windows' free RAM nears the scale-back line.
- **A free slows a request running meanwhile by about a fifth** for the 10-15 s it takes, the price of handing the
  memory back promptly.
- **Windows got the 27 GiB back within a minute of the shrink,** on its own schedule: +29.9 GB at 60 s and nothing
  yet at 30 s in this run, +11.3 GB at 15 s and +30.0 GB at 60 s in the rerun below, 13.5 GB by 30 s before.
- **Startup took 41-45 s** without zero-filling the RAM cache, no clear change.

### Memory reused after a shrink

The chunks a shrink takes out now stay pinned until the background free reaches them, the highest first, and a grow
that comes meanwhile takes them back before it pins anything (commit `7d0e87d`): nothing to allocate, and the memory
limits never came down for them. In the stress run on it (every check passed again), the first of the quick switches
went back to full 2 s after a 27 GiB shrink, took back 3,584 experts' memory (8.8 GB) and answered in 0.4 s; before,
such a switch waited 1.6-2.2 s for the free. The other three followed shrinks of a few thousand empty slots, already
freed by then. The resize workload, rerun on it, matched the run above: shrinks 0.86-0.87 s, grows 10-12 s without
requests and 28 s with one, its first token after 1.7 s, the answers identical.

This only helps a switch back within seconds of a shrink (the dashboard, or a client switching quickly): the manager
waits at least a minute before scaling back up (`min_dwell_s`), and by then the free is done (about 3 s between
requests, up to 15 s while they run).

Raw results: `results/` (`4.05-card-256k-ram-resize.json` for the run before this section,
`4.05-card-256k-stress-first.json` for the first stress run, `4.05-card-256k-stress.json` and
`4.05-card-256k-ram-background.json` for the runs on `99a99b7`, `4.05-card-256k-stress-reuse.json` and
`4.05-card-256k-ram-reuse.json` for those on `7d0e87d`).
