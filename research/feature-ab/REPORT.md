# Engine features and where a token's time goes

2026-11-07, RTX 5090 (32 GB, PCIe 5.0 x16), 64 GB RAM, Windows 11 + WSL2, PyTorch 2.15 on CUDA 13.2, Qwen3.8-Flash-Next
EXL3, 256K context, whole card. Repertory's engine at `a44e26d`.

## The three features that are built but off

The standard workload (three short answers, documents of 1K to 64K tokens, a 99K-token retrieval test), one run per
configuration, `repertory bench` (results in [results/](results/)).

| Run | Answers, median | 99K-token prompt | 1K / 8K / 32K / 64K prompts | Experts from VRAM / RAM / SSD |
|---|---|---|---|---|
| 4.05, today's settings | 60.9 tok/s | 5,417 tok/s | 649 / 2,152 / 4,233 / 5,647 | 92 / 8 / 0 % |
| 4.05, layer-ahead prefetch | 57.5 (-6%) | 5,495 (+1%) | 664 / 2,165 / 4,253 / 5,756 | 92 / 8 / 0 % |
| 4.05, MTP drafting 1 token | 62.3 (+2%) | 4,774 (-12%) | 520 / 1,819 / 3,804 / 5,315 | 88 / 11 / 1 % |
| 4.05, MTP drafting 2 tokens | 63.8 (+5%) | 4,804 (-11%) | 477 / 1,777 / 3,760 / 5,367 | 87 / 11 / 2 % |
| 3.05, today's settings | 65.8 tok/s | 5,465 tok/s | 851 / 2,512 / 4,640 / 5,985 | 95 / 4 / 0 % |
| 3.05, CPU lane | 38.3 (-42%) | 5,092 (-7%) | 697 / 2,487 / 4,446 / 5,511 | 78 / 22 / 0 % |

All retrieval tests found their three combinations.

- **Prefetch** reads the next layer's predicted experts from the SSD early. Only 0.35% of picks come from the SSD
  here, so it has little to hide, and its work costs 6% in answers. Off.
- **MTP drafting** answers 2-5% faster but reads prompts 11-12% slower: the draft layer and its experts take VRAM
  (6,240 experts cached instead of 7,040), and with the 35 GiB RAM cache the first try ran Windows down to 1 GB of
  free RAM (the benchmark stepped aside and ran it again). For an agent's long prompts that is a loss. Off.
- **The CPU lane** computes RAM-resident experts on the CPU. On a desktop with dual-channel memory, copying them to
  the GPU is much faster. Off.

Today's defaults (all three off) are the right ones on this PC.

## Where an answer token's time goes

One 1,227-token answer (thinking off), with the engine's counters read before and after, and its per-layer timeline
(medians of the last 4,096 layer calls). About 20.5 ms per token, 48 layers:

| Part | Per layer | Per token | Share |
|---|---|---|---|
| GPU compute: attention, the 10 experts, the rest | ~214 us | ~10.3 ms | ~50% |
| Copying RAM-resident experts to VRAM (~40 a token) | ~96 us on average | ~4.6 ms | ~20-25% |
| The expert engine's per-layer exchange with the host | ~25 us | ~1.2 ms | ~6% |
| Waiting on the host's replies and on the SSD | | ~0.9 ms | ~5% |
| Everything outside the layers (sampling, Python) | | ~1 ms | ~5% |

Expert picks per token: 480; 437.8 from VRAM (91.2%), 40.5 from RAM (8.4%), 1.7 from the SSD (0.35%).

- **Python is not the bottleneck.** Everything outside the layers is about 5% of a token; rewriting it in a faster
  language would gain little.
- **The copies are.** The engine copies RAM-resident experts at 23.5 GB/s (median). [h2d_bench.py](h2d_bench.py)
  shows the same link moves 2.46 MB experts at 52-55 GB/s, one at a time or batched, from the same kind of pinned
  memory the RAM tier uses:

| Source memory | One expert at a time | All at once |
|---|---|---|
| `torch` pinned | 53.5 GB/s | 55.1 GB/s |
| `cudaHostAlloc(Portable|Mapped)` (the RAM tier's, under WSL2) | 51.9 GB/s | 53.4 GB/s |

The engine appears to copy with GPU threads reading host memory directly rather than with the GPU's copy engines.
Moving those copies to the copy engines could save about 2-2.5 ms a token, roughly 12% (4.05 answers from about 61
to about 68 tok/s). That is the next engine change, to be measured the same way.
