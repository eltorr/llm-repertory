# Benchmarks

Results from Repertory's own benchmark and head-to-head, by GPU and preset. A preset is the card size the model had
and the PC's RAM, as the README's presets name them. "Capped" rows hold the engine to a smaller card's VRAM on a
bigger one. Real smaller cards also have less memory bandwidth, so expect somewhat less from them.

- **Quality** is the [head-to-head](../research/head-to-head/REPORT.md): seven short but hard tasks. Three are
  coding tasks graded by tests, two are reasoning problems with one right answer, one is a support ticket solved with
  tools, and one is a log investigation. Thinking is on at high effort, with the model's own sampling, no cap on output, and
  20 minutes a task. Each task runs once, so one task can flip between runs. The time limit makes slow setups fail
  tasks a faster one would finish.
- **Speed** is the [benchmark](manager.md#benchmarks): decode over short answers with thinking off, greedy, and
  prefill on the longest document. The `standard` and `reference` workloads use different requests, so compare
  rows of the same workload.

<!-- results:begin -->
## Quality: head-to-head

| Model | Quant | Preset | Passed | calc | sched | cache | paths | puzzle | ticket | log | Run on |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next | 4.05 bpw | 32 GB card | 6/7 | cut off (20 min) | pass (6.5 min) | pass (18.6 min) | pass (6.3 min) | pass (0.4 min) | pass (0.4 min) | pass (2.7 min) | RTX 5090 (32 GB), 64 GB RAM, Windows (WSL2), 2026-11-08 01:15 |
| GLM-5.3-Flash | 2.05 bpw | 32 GB card | 5/7 | pass (3.9 min) | pass (6.6 min) | fail (1.8 min) | pass (5.7 min) | pass (1.2 min) | pass (0.6 min) | fail (2.1 min) | RTX 5090 (32 GB), 64 GB RAM, Windows (WSL2), 2026-11-08 13:26 |
| GLM-5.3-Flash | 3.05 bpw | 32 GB card | 5/7 | pass (3.7 min) | fail (3.4 min) | pass (4.2 min) | cut off (20 min) | pass (1.8 min) | pass (1.9 min) | pass (4.9 min) | RTX 5090 (32 GB), 64 GB RAM, Windows (WSL2), 2026-11-08 02:12 |

## Speed, by GPU

### RTX 5090 (32 GB), 64 GB RAM, Windows (WSL2)

| Preset | Model | Quant | Context | RAM cache | Engine options | Decode | Prefill | First token | Workload | Date |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 GB card | Qwen3.8-Flash-Next | 3.05 bpw | 256K | 22 GiB | - | 65.8 tok/s | 5,465 tok/s (99K) | 0.28 s | standard | 2026-11-07 19:35 |
| 32 GB card | Qwen3.8-Flash-Next | 3.05 bpw | 256K | 22 GiB | cpu | 38.3 tok/s | 5,092 tok/s (99K) | 0.35 s | standard | 2026-11-07 19:39 |
| 32 GB card | Qwen3.8-Flash-Next | 4.05 bpw | 256K | 35 GiB | - | 60.9 tok/s | 5,417 tok/s (99K) | 0.37 s | standard | 2026-11-07 19:17 |
| 32 GB card | Qwen3.8-Flash-Next | 4.05 bpw | 256K | 35 GiB | mtp=1 | 62.3 tok/s | 4,774 tok/s (99K) | 0.47 s | standard | 2026-11-07 19:28 |
| 32 GB card | Qwen3.8-Flash-Next | 4.05 bpw | 256K | 35 GiB | mtp=2 | 63.8 tok/s | 4,804 tok/s (99K) | 0.50 s | standard | 2026-11-07 19:32 |
| 32 GB card | Qwen3.8-Flash-Next | 4.05 bpw | 256K | 35 GiB | prefetch | 57.5 tok/s | 5,495 tok/s (99K) | 0.36 s | standard | 2026-11-07 19:20 |
| 32 GB card | GLM-5.3-Flash | 2.05 bpw | 256K | 39 GiB | - | 20.1 tok/s | 2,262 tok/s (58K) | 0.84 s | reference | 2026-11-08 03:04 |
| 32 GB card | GLM-5.3-Flash | 2.05 bpw | 256K | 38 GiB | - | 19.7 tok/s | 2,277 tok/s (58K) | 0.94 s | reference | 2026-11-08 20:30 |
| 32 GB card | GLM-5.3-Flash | 3.05 bpw | 256K | 38 GiB | - | 8.8 tok/s | 1,228 tok/s (58K) | 2.66 s | reference | 2026-11-08 00:41 |
| 32 GB card | Qwen3.8-Flash-Next | 3.05 bpw | 256K | 22 GiB | - | 49.0 tok/s | 6,219 tok/s (73K) | 0.15 s | reference | 2026-11-07 23:20 |
| 32 GB card | Qwen3.8-Flash-Next | 4.05 bpw | 256K | 35 GiB | - | 42.9 tok/s | 6,209 tok/s (73K) | 0.24 s | reference | 2026-11-07 23:17 |
<!-- results:end -->

## Adding results

1. Benchmark your machine: the dashboard's Benchmark tab, or `python -m repertory bench current` (and `quants` for
   each quant of the configured model). The last line says where the results are.
2. For quality, run the head-to-head against the model the service serves: `python research/head-to-head/run.py
   LABEL`. It takes about an hour on a fast setup.
3. Log them and regenerate this page:

   ```
   python -m repertory results add <the bench folder> research/head-to-head/results/LABEL.json
   python -m repertory results page
   ```

4. Open a pull request with `results/` and this page.

The log keeps what the tables show and the machine's GPU, RAM, platform and software stack. It holds no host names,
prompts or answers. Earlier measurements, from the benchmark this project began with, are in
[Measured](measured.md).
