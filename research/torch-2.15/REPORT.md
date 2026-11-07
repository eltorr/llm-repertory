# PyTorch 2.15 on CUDA 13.2 vs PyTorch 2.11 on CUDA 12.8

The engine moved from PyTorch 2.11.0 (CUDA 12.8) to PyTorch 2.15.0 (CUDA 13.2), the newest PyTorch release and its
CUDA build, on 2026-11-05. Same machine, model and settings; only the stack changed.

- **Machine:** RTX 5090 (32 GB), 64 GB RAM, Windows 11 with the engine in WSL2, driver 617.14.
- **Settings:** 4.05 bpw, full profile, 256K context, 40 GiB RAM cache, 24,576-token prefill chunks.
- **Method:** [`bench/sweep.py`](../../glm53-flash-offload/bench/sweep.py), fresh random-token prompts, 3 runs per
  prefill size (median), and decode to the natural end of each answer (no output cap), 2 rounds each. The manager's
  auto-switching was paused during both runs.

| | 2.11 / CUDA 12.8 | 2.15 / CUDA 13.2 | Change |
|---|---|---|---|
| Prefill, 8K tokens | 506 tok/s | 869 tok/s | +72% |
| Prefill, 16K tokens | 1,052 tok/s | 1,651 tok/s | +57% |
| Prefill, 32K tokens | 1,448 tok/s | 2,234 tok/s | +54% |
| Prefill, 64K tokens | 2,103 tok/s | 3,098 tok/s | +47% |
| Decode, 1 request | 48.8 tok/s | 51.2 tok/s | +5% |
| Decode, 2 requests | 53.7 tok/s | 53.9 tok/s | same |
| Decode at 32K context | 55.1 tok/s | 54.8 tok/s | same |

Random-token prompts spread the work over all experts, so these prefill figures are lower than the README's, which
use natural text. Compare the two columns with each other.

What changed to build on CUDA 13.2: the offload extensions compile as C++20 (PyTorch 2.15's headers require it), and
the build never installs an exllamav3 module made for another PyTorch. Raw results: `sweep-*.json`.
