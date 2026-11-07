# RTX 5090 under WSL2: 3.05 vs 4.05 bpw at full VRAM

Measured 2026-11-03 on the machine Repertory is developed on, before the VRAM scaling matrix
([../vram-scaling/REPORT.md](../vram-scaling/REPORT.md)).

- **Host:**
  - RTX 5090 32 GB on PCIe 5.0 x16, driver 617.14.
  - Ryzen 9 9950X, 64 GB DDR5; WSL2 Ubuntu 24.04 with 56 GB.
  - Samsung 990 PRO.
- **Engine:** this project at the time (configurable limits + WSL2 pinned-memory fix). The server used its default
  262,144-token context, 4-bit KV and 24,576-token prefill chunks, with about 28 GB of VRAM for the model.
- **Method ([bench_compare.py](bench_compare.py)):**
  - A fresh server for each run, then one cold request and three warm-ups.
  - Eight decode runs capped at 700 tokens, with thinking on.
  - A 94,466-token prompt with a needle, sent twice: cold, then from the prefix cache.
  - Decode speed is completion tokens over the time between the first and last streamed token.

| Quant | RAM cache | Startup | Decode, median (min-max) | First token, short | 94K prefill | 94K repeat | Needle | VRAM peak (device) |
|---|---|---|---|---|---|---|---|---|
| 3.05 bpw | 26 GiB (every expert in VRAM or RAM) | 66 s | **60.8** (55.2-65.8) tok/s | 1.02 s | **6,407 tok/s** (14.7 s) | 0.14 s | yes | 30,169 MiB |
| 4.05 bpw | 34 GiB (~11% of experts on NVMe) | 58 s | 56.3 (45.0-59.7) tok/s | 1.48 s | 5,732 tok/s (16.5 s) | 0.15 s | yes | 29,416 MiB |
| 4.05 bpw | 40 GiB (every expert in VRAM or RAM) | 49 s | 51.0 (47.1-58.4) tok/s | 1.32 s | 6,326 tok/s (14.9 s) | 0.13 s | yes | 29,600 MiB |

**Takeaways:**
- **3.05 is the faster quant.** Its experts are 25% smaller, so more of them fit in the VRAM cache.
- **4.05's RAM cache size barely matters for decode.** Its two decode medians are within run-to-run noise.
- **A smaller 4.05 RAM cache costs ~10% of cold prefill.** It also leaves Windows about 6 GB more free RAM.
- **Startup time** is dominated by WSL's page cache. The later launcher pre-reads libraries, and a warm start
  takes 40-50 s.
- **The VRAM peak** includes about 2 GB of the Windows desktop.
