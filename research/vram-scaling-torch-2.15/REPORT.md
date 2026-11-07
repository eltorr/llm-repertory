# VRAM scaling on PyTorch 2.15: 8 to 32 GB cards

Measured 2026-11-05 on one RTX 5090 (32 GB, PCIe 5.0 x16) under WSL2, with the engine capped to smaller cards:
Ryzen 9 9950X, 64 GB DDR5, Samsung 990 PRO, driver 617.14, PyTorch 2.15.0 on CUDA 13.2, commit `e70fbe6`. The 14 runs
took an hour. This replaces [vram-scaling](../vram-scaling/REPORT.md) and [rtx5090-wsl](../rtx5090-wsl/REPORT.md)
(PyTorch 2.11) as the source of the README's tables. Results: [results/](results/).

```
repertory bench matrix 3.05:card:262144:ram=26 4.05:card:262144:ram=34 4.05:card:262144:ram=40
repertory bench --table research/vram-scaling-torch-2.15/results
```

## Method

- **Card size emulation,** as in the earlier report: `QWEN_VRAM_LIMIT_BYTES` (the VRAM in use before the start plus
  the card's usable VRAM) sizes the expert cache, and `QWEN_VRAM_GUARD_BYTES` stops the server past it, as running
  out of memory would on that card. "VRAM for the model" is what a card leaves after the desktop: 30, 22, 14.5 and
  10.5 GB for 32, 24, 16 and 12 GB cards; 8 GB for an 8 GB card without a display, 7 GB with one.
- **What CUDA can't see is counted now.** Under WSL2, CUDA doesn't report part of what Windows apps use: 1.0-1.1 GB
  during these runs, with a desktop and a browser open. The engine measures that at startup and counts it against
  the limit (see [below](#fixed-before-these-runs)).
- **RAM:** every capped run had a 38 GiB RAM cache, as in the earlier matrix, so the runs differ only in VRAM. The
  three runs on the whole card had 26, 34 and 40 GiB, the full profile's sizes.
- **Workload:** the prompts and documents of the original `benchmark.py`
  ([MaxDevv/lean-qwenflash](https://github.com/MaxDevv/lean-qwenflash/blob/fa9a772d9aa9b94a5653a2c6880eeb97af0d2e9d/benchmark.py),
  which this project began as a fork of), sent over the server's API. Repertory's benchmark has its own since, so its
  runs don't compare 1:1 with these:
  - thinking off, greedy decoding, and every answer to its natural end (no output cap);
  - a warm-up (a short arithmetic question, then one unscored trial), then three trials. Each trial is one answer of
    three paragraphs (208-243 tokens), then documents of 1K, 8K, 32K and 64K tokens of repeated filler text, each
    answered in one sentence. Each trial's text differs from the start, so nothing comes from the prompt cache;
  - a retrieval test: three codes at 10%, 50% and 90% of a document of up to 100K tokens (the context less 2K).
- **Measures:**
  - decode is completion tokens over the time between the first and last streamed token, the median of three;
  - prefill is prompt tokens over the time to the first token. The "Prefill" column is the retrieval document.
- **Not emulated:** a smaller card's memory bandwidth, compute and PCIe link. Every row runs on the 5090's 1.8 TB/s and
  PCIe 5.0, so **real smaller cards will be slower**, most of all where many expert picks come from RAM.

## Results

| Card | Quant | VRAM for the model | Context | RAM cache | Prefill chunk | Cached experts | Expert picks VRAM / RAM / NVMe | Decode, median (min) | Prefill | First token | Startup |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 GB | 3.05 | 30 GB | 256K | 38 GiB | 24,576 | 10,400 | 95 / 4 / 1 % | **74.4** (69.9) tok/s | 4,722 tok/s (100K prompt) | 0.18 s | 42 s |
| 24 GB | 3.05 | 22 GB | 256K | 38 GiB | 24,576 | 6,048 | 88 / 10 / 1 % | **62.5** (58.2) tok/s | 4,628 tok/s (100K prompt) | 0.21 s | 49 s |
| 16 GB | 3.05 | 14.5 GB | 256K | 38 GiB | 8,192 | 2,560 | 70 / 28 / 1 % | **41.9** (41.3) tok/s | 2,684 tok/s (100K prompt) | 0.28 s | 45 s |
| 12 GB | 3.05 | 10.5 GB | 64K | 38 GiB | 2,048 | 1,472 | 58 / 40 / 2 % | **36.1** (35.9) tok/s | 688 tok/s (63K prompt) | 0.29 s | 50 s |
| 12 GB | 3.05 | 10.5 GB | 128K | 38 GiB | 1,536, no staging | 1,120 | 53 / 46 / 1 % | **34.4** (34.1) tok/s | 1,933 tok/s (100K prompt) | 0.27 s | 53 s |
| 8 GB | 3.05 | 8 GB | 16K | 38 GiB | 1,536, no staging | 512 | 26 / 71 / 2 % | **26.4** (25.0) tok/s | 1,490 tok/s (14K prompt) | 0.31 s | 58 s |
| 8 GB | 3.05 | 7 GB | 16K | 38 GiB | 1,536, no staging | 0 | 0 / 99 / 1 % | **20.3** (19.7) tok/s | 1,637 tok/s (14K prompt) | 0.29 s | 62 s |
| 32 GB, uncapped | 3.05 | whole card | 256K | 26 GiB | 24,576 | 9,728 | 94 / 4 / 1 % | **72.1** (70.5) tok/s | 4,668 tok/s (100K prompt) | 0.20 s | 50 s |
| 32 GB | 4.05 | 30 GB | 256K | 38 GiB | 24,576 | 7,616 | 92 / 7 / 1 % | **63.0** (59.1) tok/s | 4,296 tok/s (100K prompt) | 0.24 s | 66 s |
| 24 GB | 4.05 | 22 GB | 256K | 38 GiB | 24,576 | 4,352 | 82 / 16 / 1 % | **51.4** (47.4) tok/s | 4,252 tok/s (100K prompt) | 0.29 s | 58 s |
| 16 GB | 4.05 | 14.5 GB | 256K | 38 GiB | 4,096 | 1,696 | 61 / 37 / 2 % | **34.5** (33.9) tok/s | 1,183 tok/s (100K prompt) | 0.35 s | 53 s |
| 12 GB | 4.05 | 10.5 GB | 32K | 38 GiB | 1,536, no staging | 1,088 | 54 / 43 / 3 % | **29.6** (29.3) tok/s | 1,104 tok/s (30K prompt) | 0.55 s | 62 s |
| 32 GB, uncapped | 4.05 | whole card | 256K | 34 GiB | 24,576 | 7,104 | 91 / 8 / 1 % | **60.9** (57.6) tok/s | 4,364 tok/s (100K prompt) | 0.25 s | 53 s |
| 32 GB, uncapped | 4.05 | whole card | 256K | 40 GiB | 24,576 | 7,104 | 91 / 8 / 1 % | **61.0** (58.3) tok/s | 4,233 tok/s (100K prompt) | 0.25 s | 58 s |

Every run completed, and every retrieval test found all three codes. "First token" is for the 27-token question.

Document reading by size (median of three):

| Run | 1K | 8K | 32K | 64K |
|---|---|---|---|---|
| 3.05, whole card, 26 GiB | 182 tok/s | 1,420 tok/s | 3,408 tok/s | 4,823 tok/s |
| 3.05, 30 GB | 184 tok/s | 1,429 tok/s | 3,425 tok/s | 4,736 tok/s |
| 3.05, 22 GB | 183 tok/s | 1,418 tok/s | 3,357 tok/s | 4,625 tok/s |
| 3.05, 14.5 GB | 178 tok/s | 1,429 tok/s | 2,358 tok/s | 2,660 tok/s |
| 3.05, 10.5 GB, 64K | 179 tok/s | 607 tok/s | 727 tok/s | - |
| 3.05, 10.5 GB, 128K, no staging | 831 tok/s | 1,697 tok/s | 1,928 tok/s | 1,963 tok/s |
| 3.05, 8 GB, no staging | 830 tok/s | 1,654 tok/s | - | - |
| 3.05, 7 GB, no staging | 814 tok/s | 1,614 tok/s | - | - |
| 4.05, whole card, 40 GiB | 144 tok/s | 1,122 tok/s | 2,876 tok/s | 4,191 tok/s |
| 4.05, whole card, 34 GiB | 144 tok/s | 1,129 tok/s | 2,914 tok/s | 4,263 tok/s |
| 4.05, 30 GB | 144 tok/s | 1,127 tok/s | 2,888 tok/s | 4,287 tok/s |
| 4.05, 22 GB | 144 tok/s | 1,124 tok/s | 2,858 tok/s | 4,168 tok/s |
| 4.05, 14.5 GB | 142 tok/s | 773 tok/s | 1,056 tok/s | 1,127 tok/s |
| 4.05, 10.5 GB, 32K, no staging | 327 tok/s | 1,091 tok/s | - | - |

## What it shows

- **Decode follows the share of expert picks served from VRAM,** as before:
  - 3.05 bpw: 74 tok/s on 30 GB, 62.5 on 22, 42 on 14.5, 34-36 on 10.5, 26 on 8 and 20 on 7.
  - 4.05 bpw: 63, 51, 34.5 and 30 tok/s.
  - 3.05 is faster at every size: its experts are 25% smaller, so about 40% more fit.
- **An 8 GB card that drives a display now runs.** With 7 GB, no expert fits in VRAM: every pick comes from RAM. It
  still decodes 20 tok/s here and reads a 14K document. On PyTorch 2.11 the same configuration ran out of VRAM on the
  long prompt, so `sizing.py` still refuses it; a real 8 GB card, on a slower PCIe link, will be well below 20 tok/s.
- **Staged prefill costs about 5.5 s per prompt, whatever its size** (the NVMe, it turned out:
  [follow-up](#follow-up-2026-11-06-short-prompts-skip-staging)). On the whole card a 1K document takes 5.6 s (182
  tok/s) and an 8K one 5.8 s. Staging-free runs read 1K in 1.2 s and 8K in 4.8 s, even on a 10.5 GB card; staged,
  32K takes 9.6 s and staging-free 17 s. So staging pays off somewhere between 8K and 32K tokens, and below that a
  prompt waits about 6 s for its first token. A 27-token question isn't staged (first token in 0.18-0.55 s).
- **Small chunks pay that cost many times.** On 10.5 GB with 2,048-token chunks, 3.05 reads 607-727 tok/s; the same
  card without staging reads 1,700-1,960 tok/s, with twice the context and 5% slower decode. On 12 GB cards
  staging-free is the better choice, and the 12 GB preset already picks it. 4.05 on 14.5 GB (4,096-token chunks)
  reads about 1,100 tok/s.
- **On the whole card, 34 and 40 GiB of RAM cache run at the same speed** at 4.05 bpw (60.9 and 61.0 tok/s, 1% of
  picks from NVMe in both). The extra 6 GiB only cost RAM in this workload: free RAM fell to 3.4 GB during the 40 GiB
  run.
- **The whole card and a 30 GB cap run alike** at 3.05 bpw: 72.1 tok/s (26 GiB RAM cache) and 74.4 (38 GiB).
- **Startup is 42-66 s.**
- **Against PyTorch 2.11:** the workload changed (short answers with thinking off, the documents above), and the
  earlier capped runs had the VRAM CUDA couldn't see on top of their limit. The two reports don't compare row by row.
  For the stack alone, see [torch-2.15](../torch-2.15/REPORT.md).

## Follow-up, 2026-11-06: short prompts skip staging

The ~5.5 s "staging cost" above was the NVMe. Staged prefill copies every expert of a layer that isn't in VRAM; one
that isn't in the RAM cache either is read from the NVMe each time, and staging doesn't keep it. A fresh server's RAM
cache starts empty and only decode filled it, so every staged prompt read most of the model from disk (about 43 GB at
4.05 bpw, ~6 GB/s).

Prompts under 16,384 expert picks per layer (1,638 tokens; nv2's pick buffer holds 16,384) now skip staging. They go
through nv2's per-pick path, which keeps what it reads in the RAM cache, so later staged prompts read from RAM. Same
card and settings (4.05 bpw, whole card, standard workload, commit `6e0f68c`; results in
[followup-2026-11-06/](followup-2026-11-06/)):

| | Matrix (staging from 512 picks) | Staging from 16,384 picks |
|---|---|---|
| 1K document | 144 tok/s | 712 tok/s (312 on the first trial) |
| 8K document | 1,129 tok/s | 2,546 tok/s |
| 32K document | 2,914 tok/s | 4,593 tok/s |
| 64K document | 4,263 tok/s | 5,449 tok/s |
| 100K retrieval | 4,364 tok/s | 5,312 tok/s |
| Answers | 60.9 tok/s | 59.7 tok/s (trials 58.9-59.8) |

On the live service, prompts of 1,375-1,712 tokens got their first token in 1.4 s and 2,000 tokens in 2.6 s, against
5.6-7.1 s before; 1,628 tokens ran unstaged and 1,642 staged, both without errors. Filling the RAM cache from the NVMe
at startup instead (the engine's warm start, 5.4 s for 14,854 experts; tried with a local change only, recorded in
`4.05-card-256k-stage16384-warmstart.json`) helped only the first trial's 1K document (312 to 446 tok/s) and added
6 s to every start, so it stays off.

The capped rows above were measured before this change; their prefill figures are low for the same reason.

## Fixed before these runs

The first try of this matrix found two problems; both were fixed and every run was redone:

- **The VRAM limit under WSL2.** Every emulated card of 16 GB or less stopped at startup ("VRAM LIMIT EXCEEDED", by
  25-50 MB). A VRAM timeline showed the expert cache filling past the limit: the engine sized it from
  `cudaMemGetInfo`, which under WSL2 misses part of what Windows apps use, while the limit and the guard count it
  (NVML). With the model stopped, CUDA reported 1,654 MiB in use and NVML 2,778 MiB; the cache overshot by 1,126 MiB.
  On 2026-11-04 the desktop used 0.95 GB, little enough to hide in the reserve; on 2026-11-05 it used 2.3 GB. The
  engine now measures the difference at startup (`GLM53_VRAM_UNSEEN_BYTES`) and counts it. The same fix keeps a lean
  profile within its budget next to a game.
- **The RAM cache varied between runs.** Sized from the RAM free when each run started, it went from 8 to 25 GiB,
  depending on how much memory WSL had handed back from the run before; the 22 GB run measured 23% slower for that
  alone. The matrix now gives every run the same 38 GiB.

Looking into the first problem also turned up a leftover CUDA 12 runtime from the 2.11 install, which the engine
loaded for pinned memory; a fresh CUDA 13 install doesn't have it. The engine now uses PyTorch's own runtime (see the
[changelog](../../CHANGELOG.md)).

## Next measurements

- **8 GB with a display:** decide whether `sizing.py` should offer it, now that it runs.
- **Short prompts:** the ~5.5 s staging cost for prompts under 8-32K tokens. Raising the size at which prefill starts
  staging could cut the first token of a 1K prompt from ~6 s to about 1 s.
- **RAM cache on the whole card:** whether 34 GiB serves other workloads as well as 40, which would leave 6 GiB more
  RAM for the PC.
- **Real 12, 16 and 24 GB cards,** and 16-32 GB or 128 GB of RAM (the README's presets for those are computed).
