# Measured

Every configuration measured so far, on an RTX 5090 with 64 GB of RAM under Windows 11 and WSL2. The README shows
the fastest setup per card size; each row here links to the run's report.

- **Capped** rows hold the engine to a smaller card's VRAM. Real smaller cards also have less memory bandwidth, so
  expect somewhat less from them.
- **Decode** is short answers with thinking off. **Prefill** is a document of up to 100K tokens.
- These runs used an earlier benchmark workload. Repertory's own benchmark (protocol `repertory-1`) has different
  prompts and documents, so new runs don't compare 1:1 with these.

<!-- measured:begin -->
| Card (VRAM) | Quant | Context | RAM cache | Prefill | Decode | Measured on |
|---|---|---|---|---|---|---|
| **32 GB** | 3.05 | 256K | 26 GiB | 4,668 tok/s | 72.1 tok/s | [RTX 5090](../research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **32 GB** | 3.05 | 256K | 38 GiB | 4,722 tok/s | 74.4 tok/s | [RTX 5090 capped to 32 GB](../research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **24 GB** | 3.05 | 256K | 38 GiB | 4,628 tok/s | 62.5 tok/s | [RTX 5090 capped to 24 GB](../research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **16 GB** | 3.05 | 256K | 38 GiB | 2,684 tok/s | 41.9 tok/s | [RTX 5090 capped to 16 GB](../research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **12 GB** | 3.05 | 64K | 38 GiB | 688 tok/s | 36.1 tok/s | [RTX 5090 capped to 12 GB](../research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **12 GB** | 3.05 | 128K, no staging | 38 GiB | 1,933 tok/s | 34.4 tok/s | [RTX 5090 capped to 12 GB](../research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **8 GB, no display** | 3.05 | 16K, no staging | 38 GiB | 1,490 tok/s | 26.4 tok/s | [RTX 5090 capped to 8 GB](../research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **8 GB, with display** | 3.05 | 16K, no staging | 38 GiB | 1,637 tok/s | 20.3 tok/s | [RTX 5090 capped to 8 GB](../research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **32 GB** | 4.05 | 256K | 34 GiB | 5,312 tok/s | 59.7 tok/s | [RTX 5090](../research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **32 GB** | 4.05 | 256K | 40 GiB | 4,233 tok/s | 61.0 tok/s | [RTX 5090](../research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **32 GB** | 4.05 | 256K | 38 GiB | 4,296 tok/s | 63.0 tok/s | [RTX 5090 capped to 32 GB](../research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **24 GB** | 4.05 | 256K | 38 GiB | 4,252 tok/s | 51.4 tok/s | [RTX 5090 capped to 24 GB](../research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **16 GB** | 4.05 | 256K | 38 GiB | 1,183 tok/s | 34.5 tok/s | [RTX 5090 capped to 16 GB](../research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **12 GB** | 4.05 | 32K, no staging | 38 GiB | 1,104 tok/s | 29.6 tok/s | [RTX 5090 capped to 12 GB](../research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
<!-- measured:end -->

The table is generated from the measurements in [sizing.py](../engines/exl3_offload/sizing.py)
(`python3 engines/exl3_offload/sizing.py presets --write docs/measured.md`). Every run, with its raw results, is
listed in [research/](../research/README.md).
