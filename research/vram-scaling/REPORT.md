# VRAM scaling: 8 to 32 GB cards (README To-Do)

Measured 2026-11-04 on one RTX 5090 (32 GB, PCIe 5.0 x16) under WSL2, with the engine capped to smaller cards.
Ryzen 9 9950X, 64 GB DDR5 (56 GB for WSL), Samsung 990 PRO. Results: [results/](results/). Runner:
`bench/vram_matrix.py`, which became [`repertory bench`](../../repertory/bench.py) on 2026-11-05; table:
`repertory bench --table research/vram-scaling/results`.

## Method

- **Card size emulation:** the engine is capped the way the upstream 16 GB numbers were taken.
  - `QWEN_VRAM_LIMIT_BYTES` (the VRAM in use before the start plus the card's usable VRAM) sizes the expert cache.
  - `QWEN_VRAM_GUARD_BYTES` stops the server if device usage ever goes past that. On a real card of that size,
    that would have been an out-of-memory error.
  - The prefill chunk, staging and reserve come from [`sizing.py`](../../engines/exl3_offload/sizing.py) through `run.sh`, as they would on
    such a card.
- **"VRAM for the model"** is what is left after the desktop:
  - 30 GB for a 32 GB card, 22 GB for 24 GB, 14.5 GB for 16 GB, 10.5 GB for 12 GB.
  - For 8 GB: 8 GB when the card doesn't drive a display, 7 GB when it does.
- **RAM:** the RAM cache holds every expert that is not in VRAM, up to 38 GiB. The few NVMe reads are experts beyond
  that.
- **Workload:**
  - One cold request and one warm-up, then five decode runs capped at 700 tokens, with thinking on (default effort).
  - One long prompt (up to 94K tokens, at most the context minus 6K) with a needle to find.
  - Decode speed is completion tokens over the time between the first and last streamed token.
  - Prefill speed is prompt tokens over the time to the first token.
- **Not emulated:** a smaller card's memory bandwidth, compute and PCIe link. Every row runs on the 5090's
  1.8 TB/s and PCIe 5.0, so **absolute speeds on real smaller cards will be lower**, most of all where many
  expert picks come from RAM (they cross PCIe every token). What these rows show is the effect of VRAM capacity.

## Results

| Card | Quant | VRAM for the model | Context | Prefill chunk | Cached experts | Expert picks VRAM / RAM / NVMe | Decode, median (min) | Prefill | First token | Startup |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 GB | 3.05 | 30 GB | 256K | 24,576 | 10,304 | 95 / 4 / 1 % | **64.0** (61.1) tok/s | 5,754 tok/s (93K prompt) | 1.19 s | 44 s |
| 24 GB | 3.05 | 22 GB | 256K | 24,576 | 5,984 | 87 / 12 / 1 % | **51.5** (49.0) tok/s | 5,896 tok/s (93K prompt) | 1.43 s | 60 s |
| 16 GB | 3.05 | 14.5 GB | 256K | 8,192 | 2,400 | 66 / 33 / 1 % | **37.8** (36.5) tok/s | 4,338 tok/s (93K prompt) | 1.62 s | 58 s |
| 12 GB | 3.05 | 10.5 GB | 64K | 2,048 | 1,376 | 53 / 44 / 3 % | **33.2** (32.5) tok/s | 1,816 tok/s (58K prompt) | 1.63 s | 41 s |
| 12 GB | 3.05 | 10.5 GB | 128K | 1,536, no staging | 992 | 43 / 55 / 2 % | **29.9** (28.9) tok/s | 1,564 tok/s (93K prompt) | 0.44 s | 54 s |
| 8 GB, no display | 3.05 | 8 GB | 16K | 1,536, no staging | 352 | 0 / 95 / 5 % | **20.2** (20.0) tok/s | 1,166 tok/s (9K prompt) | 0.48 s | 40 s |
| 8 GB, with display | 3.05 | 7 GB | 16K | 1,536, no staging | - | - | **fails**: ran out of VRAM on the long prompt (21.9 tok/s decode before that) | - | - | 61 s |
| 32 GB | 4.05 | 30 GB | 256K | 24,576 | 7,520 | 91 / 8 / 1 % | **56.1** (51.6) tok/s | 5,293 tok/s (93K prompt) | 2.07 s | 58 s |
| 24 GB | 4.05 | 22 GB | 256K | 24,576 | 4,256 | 80 / 18 / 1 % | **42.6** (40.5) tok/s | 5,006 tok/s (93K prompt) | 1.99 s | 63 s |
| 16 GB | 4.05 | 14.5 GB | 256K | 4,096 | 1,600 | 56 / 41 / 2 % | **30.3** (29.8) tok/s | 1,384 tok/s (93K prompt) | 2.93 s | 56 s |
| 12 GB | 4.05 | 10.5 GB | 32K | 1,536, no staging | 992 | 43 / 53 / 4 % | **26.6** (24.5) tok/s | 431 tok/s (25K prompt) | 0.60 s | 62 s |

Every completed run found the needle.

## What it shows

- **Decode follows the share of expert picks served from VRAM.**
  - At 3.05 bpw it goes 64 to 51.5 to 37.8 to 33 tok/s from 32 to 12 GB.
  - At 4.05 bpw it goes 56 to 43 to 30 to 27 tok/s.
  - Even a 2,400-expert cache (16 GB) serves two thirds of the picks, because a few experts are picked far more often
    than the rest.
- **3.05 bpw beats 4.05 at every size.** Its experts are 25% smaller, so about 40% more of them fit.
- **Prefill holds up to 16 GB at 3.05.** It drops on 12 GB cards, and on 16 GB at 4.05, because staging a whole
  layer's experts (1.9 GB at 3.05, 2.5 GB at 4.05) no longer fits next to big chunks. The chunks shrink, or prefill
  runs staging-free and reads every miss over PCIe. 4.05 on a 12 GB card prefills only ~430 tok/s.
- **12 GB cards work:**
  - 3.05 bpw up to 128K context (staging-free above ~64K).
  - 4.05 bpw at 32K.
  - For agents, 3.05 at 64K is the better balance.
- **8 GB cards work only without a display, and slowly:**
  - With ~8 GB free, 3.05 runs at 16K context, staging-free. It decoded ~20 tok/s here, with 95% of picks coming
    from RAM over PCIe 5.0.
  - A real 8 GB card (PCIe 4.0 or 3.0, less bandwidth) will be well below that.
  - When the card also drives a desktop (~7 GB left), short prompts run but long prompts run out of VRAM, so
    `sizing.py` refuses that configuration.
  - Making room means shrinking the fixed part first. About 6.2 GB of VRAM is used before any expert is cached.
    See the engine To-Do in the README.
- **Startup is 40-60 s at every size.**

## How the sizing model got here

`sizing.py` is calibrated from these runs:
- VRAM outside the expert cache was 9.0 / 9.44 GB at 256K, 6.85 GB at 64K and 6.39 GB at 16K. That is
  6.15 / 6.62 GB plus the KV cache's 10,752 bytes per token.
- The first pass showed four configurations running 20-180 MB past the emulated card during prefill:
  - **Staging allocated on small cards:** PyTorch held 0.8-1 GB more than it used.
  - **Staging-free with a 0.35-0.4 GB reserve:** too little for its own prefill.
- The fixes:
  - 0.21 MB per prefill token (from the research runs, instead of 0.15).
  - 0.5 GB of fragmentation margin before choosing staging.
  - A reserve that covers staging-free prefill.
  - The engine's own activation estimate kept on small cards.
- With those, every configuration it accepts ran to completion.

## A dead end worth knowing: emulating with a ballast under WSL2

The first attempt held the rest of the card in a separate process ([`vram_ballast.py`](vram_ballast.py))
instead of capping the engine. Under Windows/WSL2, the driver (WDDM) **over-commits VRAM**: it pages idle allocations
out to system RAM. The engine got the ballast's memory back: 18 GB of expert cache on an emulated 10.5 GB card. Where
the model's own memory got paged, decode fell to 32-47 tok/s. Those runs are kept in [attempt1-ballast/](attempt1-ballast/)
and are not valid card emulation.

The same mechanism means a game that grabs VRAM on Windows slows the model through paging instead of failing it.
That is why the manager watches other apps' VRAM and steps the model down instead of waiting for errors.

## Next measurements

- **RAM, for the presets:** every row here had a 25-38 GiB RAM cache (a 64 GB machine). The README presets for 16-32 GB
  of RAM (4-15 GiB caches) and for 128 GB with a 24 GB card or smaller are computed, not measured.
  `repertory bench` takes `:ram=GIB` per run (e.g. `4.05:22:262144:ram=48`, `3.05:14.5:262144:ram=4`) and
  `card` instead of a VRAM figure to run a real card uncapped; `repertory bench --sizing` prints the rows for
  `sizing.MEASURED`.
- **Real 12, 16 and 24 GB cards:** what PCIe 4.0 and lower bandwidth cost, compared with these capped numbers.
- **8 GB with a display:** after shrinking the fixed footprint, profile the ~2.5 GB of VRAM no component accounts for
  (`torch.cuda.memory_snapshot()` after load).
- **Prefill on 12-16 GB cards:** a single staging buffer instead of two would save 0.9-1.3 GB, at the cost of
  copy/compute overlap.
