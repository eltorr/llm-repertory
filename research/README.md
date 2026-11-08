# Run records

Every benchmark behind the numbers in this repo, newest first. Each folder has a report and the raw results. These
runs predate the name Repertory: their raw results say `qwenflash` for the manager and `QWEN38_` for the engine's
settings (now `repertory` and `GLM53_`), and they used the workload of the original `benchmark.py` of
[MaxDevv/lean-qwenflash](https://github.com/MaxDevv/lean-qwenflash), which this project began as a fork of.
Repertory's benchmark has prompts and documents of its own, so new runs don't compare 1:1 with them.

| Date | Machine | Stack | What it measured | Record |
|---|---|---|---|---|
| 2026-11-08 | RTX 5090, 64 GB RAM, WSL2 | PyTorch 2.15, CUDA 13.2 | Qwen3.8-Flash-Next 4.05 against GLM-5.3-Flash 3.05 and 2.05 on seven hard tasks: 6/7, 5/7, 5/7. Each solved two of the three coding tasks; GLM did them in 11-12 minutes, Qwen (at xhigh) in 45 | [head-to-head](head-to-head/REPORT.md) |
| 2026-11-07 | RTX 5090, 64 GB RAM, WSL2 | PyTorch 2.15, CUDA 13.2 | The engine features that are off (prefetch, MTP drafting, the CPU lane): all slower overall on this PC. Where a token's time goes: GPU compute ~50%, RAM-to-VRAM expert copies ~20-25% at 23.5 GB/s where the link does 52-55 GB/s, Python ~5% | [feature-ab](feature-ab/REPORT.md) |
| 2026-11-07 | RTX 5090, 64 GB RAM, WSL2 | PyTorch 2.15, CUDA 13.2 | Repertory's engine against the earlier fork's: five greedy requests (short, code, tool call, 73K-token document, thinking) give identical answers | [engine-parity](engine-parity/REPORT.md) |
| 2026-11-07 | RTX 5090, 64 GB RAM, WSL2 | PyTorch 2.15, CUDA 13.2 | Switching full <-> lean without a restart: 0.2-0.5 s, 13.3 GB freed, same answers. The RAM cache too: under a second to shrink it, grown back in the background without holding requests (stress-tested) | [live-resize](live-resize/REPORT.md) |
| 2026-11-05 | RTX 5090, capped to 7-32 GB and uncapped, 64 GB RAM, WSL2 | PyTorch 2.15, CUDA 13.2 | VRAM scaling: the README's tables. Follow-up 2026-11-06: short prompts skip staging (documents read 1.2-5 times faster) | [vram-scaling-torch-2.15](vram-scaling-torch-2.15/REPORT.md) |
| 2026-11-05 | RTX 5090, 64 GB RAM, WSL2 | 2.11 / 12.8 vs 2.15 / 13.2 | The stack upgrade: prefill 47-72% faster, decode the same | [torch-2.15](torch-2.15/REPORT.md) |
| 2026-11-04 | RTX 5090, capped to 8-32 GB | PyTorch 2.11, CUDA 12.8 | VRAM scaling, replaced by vram-scaling-torch-2.15. Decode answers were capped at 700 tokens | [vram-scaling](vram-scaling/REPORT.md) |
| 2026-11-03 | RTX 5090, 64 GB RAM, WSL2 | PyTorch 2.11, CUDA 12.8 | 3.05 vs 4.05 bpw on the full card, replaced by vram-scaling-torch-2.15. Decode answers were capped at 700 tokens | [rtx5090-wsl](rtx5090-wsl/REPORT.md) |

MaxDevv's own runs (an RTX 4090 held to 16 GB of VRAM and 8 GiB of RAM, long context up to 262K tokens, and an RTX
3080 Laptop) are in [his repository](https://github.com/MaxDevv/lean-qwenflash/tree/fa9a772d9aa9b94a5653a2c6880eeb97af0d2e9d/research).

## Adding a run

1. **Run it** with [`repertory bench`](../docs/manager.md#benchmarks), or from the dashboard's Benchmark tab. The
   manager frees the GPU and puts the model back afterwards. Thinking is off, decoding is greedy, and every answer
   runs to its natural end.

   ```
   repertory bench matrix --out research/NAME/results      the README's table: 8-32 GB cards on a 32 GB one
   repertory bench quants --out research/NAME/results      3.05 and 4.05 bpw on this card
   ```

   Each result records the machine and its stack (GPU, driver, CUDA, PyTorch, commit). The matrix takes about an
   hour. A run steps aside while a game needs the GPU, and `repertory bench --resume` continues a benchmark cut short.
2. **Tables:** `repertory bench --table research/NAME/results` prints the report's tables, and `--sizing` prints the
   rows for `sizing.MEASURED`. Then `python engines/exl3_offload/sizing.py presets --write README.md` regenerates
   the README's tables.
3. **Record:** write the folder's `REPORT.md`, and add a row here with the date, machine and stack.
