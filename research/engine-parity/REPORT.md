# Repertory's engine against the earlier one: identical answers

2026-11-07, RTX 5090 (32 GB), 64 GB RAM, Windows 11 + WSL2, PyTorch 2.15 on CUDA 13.2, Qwen3.8-Flash-Next EXL3
4.05 bpw, 256K context, 35 GiB RAM cache.

Repertory's engine is sybil-solutions' glm53-flash-offload with Qwen3.8 support written independently of the earlier
fork's (see the commit history). This run checks that it gives the same answers as the engine the earlier fork
served, on the same model files and expert store.

**Method.** [engine_compare.py](engine_compare.py) sends five greedy chat requests (temperature 0, streamed, each
answer to its natural end) to a server: a short answer, code, a tool call, a 30K-token document with one fact to find,
and a short answer with thinking on. First the earlier engine (as it was serving, warm from earlier use), then
Repertory's on its own port with the earlier one paused. [compare_report.py](compare_report.py) compares text,
reasoning, tool calls, finish reasons and token counts.

| Request | Answer | Tokens | First token (earlier / Repertory) | Decode (earlier / Repertory) |
|---|---|---|---|---|
| short | identical | 92 | 1.69 / 0.81 s | 30.5 / 48.3 tok/s |
| code | identical | 262 | 0.66 / 0.30 s | 32.6 / 46.5 tok/s |
| tool call | identical (same call and arguments) | 28 | 3.01 / 2.21 s | - |
| 30K-token document | identical | 23 | 12.39 / 12.58 s | 28.4 / 36.3 tok/s |
| thinking on | identical | 58 | 0.76 / 0.46 s | 37.0 / 39.4 tok/s |

**All five answers are identical.** Repertory's engine started in 67 s, and its startup check found no bad VRAM or
RAM slots or tables.

The speeds are not a like-for-like comparison: Repertory's server was started by `run.sh` with its own VRAM limit
(30.9 GB, a VRAM cache of 8,736 experts) where the manager had given the earlier engine 29.1 GB (about 7,200), and the
earlier engine had been serving long agent prompts before. Speed is measured with the benchmarks once Repertory's
manager runs it.

Found on the first start: the nv2 control region tripped a page-alignment check, because `cudaHostAlloc` under WSL2
handed back an unaligned address for a small request. Fixed in `glm53/wsl_pinned.py` (over-allocate by a page and
align); the run above is with the fix.

Raw results: [results/](results/).
