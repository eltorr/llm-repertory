# Repertory

Repertory runs a large local model as a service on a PC that is also used for other things. It started because my
AI machine is also my gaming PC, and I wanted the model's resources to adapt: when a game needs the GPU or RAM, the
model scales itself back or pauses, and when the game closes, the model takes the resources back.

Today it runs Qwen3.8-Flash-Next, a 125B mixture-of-experts model, at up to 256K context on 8-48 GB cards, with the
experts spread over VRAM, RAM and NVMe. Clients reach it through an OpenAI-compatible API on
`http://127.0.0.1:30000/v1`.

- **A service:** the model loads on the first request, scales back or pauses for games (in half a second, without a
  restart, when only VRAM is short), comes back when they're done, and restarts after a crash.
- **Setup** on Windows (with the engine in WSL2) and Linux, from the dashboard's Setup tab or a terminal.
- **Presets** that size the model to the card and the RAM.
- **Benchmarks** of any configuration on this machine, from the dashboard or `repertory bench`.
- **The API:** OpenAI Responses and Chat Completions, with streaming, tools and reasoning levels ([api.md](docs/api.md)).

## Quick start

**You need:** an NVIDIA GPU (RTX 20 or newer, 12 GB+) with driver 595 or newer, 32 GB of RAM (16 GB on Linux), a
fast NVMe SSD with about 170 GB free, and Git and Python 3. On Windows, tick *Add python.exe to PATH* when installing
Python. On Linux, setup automates Ubuntu 22.04+ and Debian 12+.

**Windows:**

```powershell
git clone https://github.com/eltorr/repertory.git
cd repertory
.\Repertory.cmd
```

**Linux:**

```bash
git clone https://github.com/eltorr/repertory.git
cd repertory
./repertory.sh open
```

The dashboard opens at http://127.0.0.1:30000, the same port as the API. In its **Setup** tab, click
**Apply baseline**, then run the steps from top to bottom; each can be run again safely.

Then point any OpenAI-compatible client at `http://127.0.0.1:30000/v1`, with model `qwen3.8-flash-next` and any
API key. The model loads on the first request, which takes about a minute.

More: [manager guide](docs/manager.md) · [API](docs/api.md) · [engines and models](docs/engines.md) · [run records](research/README.md)

## The catalog

Repertory is built to run more than one model. [catalog/catalog.json](catalog/catalog.json) lists curated entries:
each engine and each model, pinned to one upload, with how to build, prepare and serve every pair, and what was
measured on which machine. Each engine has one adapter under [engines/](engines): its model server, launcher, sizing
and setup helpers. Today there is one, [exl3_offload](engines/exl3_offload): ExLlamaV3 with an offload engine that
keeps the hottest experts in VRAM, more in RAM, and streams the rest from NVMe, serving Qwen3.8-Flash-Next at 3.05
and 4.05 bpw.

## Presets

The manager starts from a preset for the card and the RAM. To see yours, run `./repertory.sh plan`.

- **✅** measured, with the generation speed;
- **⚪** computed by [`sizing.py`](engines/exl3_offload/sizing.py), not tested yet.

The figures are for Linux; Windows keeps about 8 GB more RAM for itself.

<!-- presets:begin -->
| Card (VRAM) | 16 GB RAM | 32 GB RAM | 64 GB RAM | 128 GB RAM |
|---|---|---|---|---|
| **8 GB, no display** | **3.05** · 16K, no staging<br>4 GiB RAM cache, 10% in memory<br>⚪ not tested | **3.05** · 16K, no staging<br>15 GiB RAM cache, 36% in memory<br>⚪ not tested | **3.05** · 16K, no staging<br>38 GiB RAM cache, 91% in memory<br>✅ *26.4 tok/s* · RTX 5090 capped to 8 GB | **3.05** · 16K, no staging<br>38 GiB RAM cache, 91% in memory<br>✅ *26.4 tok/s* · RTX 5090 capped to 8 GB, 64 GB RAM |
| **8 GB, with display** | not enough VRAM | not enough VRAM | not enough VRAM | not enough VRAM |
| **12 GB** | **3.05** · 64K, no staging<br>4 GiB RAM cache, 15% in memory<br>⚪ not tested | **3.05** · 64K, no staging<br>15 GiB RAM cache, 41% in memory<br>⚪ not tested | **3.05** · 64K, no staging<br>36 GiB RAM cache, 91% in memory<br>⚪ not tested | **3.05** · 64K, no staging<br>36 GiB RAM cache, 91% in memory<br>⚪ not tested |
| **16 GB** | **3.05** · 256K<br>4 GiB RAM cache, 18% in memory<br>⚪ not tested | **3.05** · 256K<br>15 GiB RAM cache, 44% in memory<br>⚪ not tested | **3.05** · 256K<br>35 GiB RAM cache, 92% in memory<br>✅ *41.9 tok/s* · RTX 5090 capped to 16 GB | **3.05** · 256K<br>35 GiB RAM cache, 92% in memory<br>✅ *41.9 tok/s* · RTX 5090 capped to 16 GB, 64 GB RAM |
| **24 GB** | **3.05** · 256K<br>4 GiB RAM cache, 34% in memory<br>⚪ not tested | **3.05** · 256K<br>15 GiB RAM cache, 60% in memory<br>⚪ not tested | **4.05** · 256K<br>41 GiB RAM cache, 90% in memory<br>✅ *51.4 tok/s* · RTX 5090 capped to 24 GB | **4.05** · 256K<br>41 GiB RAM cache, 90% in memory<br>✅ *51.4 tok/s* · RTX 5090 capped to 24 GB, 64 GB RAM |
| **32 GB** | **3.05** · 256K<br>4 GiB RAM cache, 52% in memory<br>⚪ not tested | **3.05** · 256K<br>15 GiB RAM cache, 79% in memory<br>⚪ not tested | **4.05** · 256K<br>33 GiB RAM cache, 90% in memory<br>✅ *59.7 tok/s* · RTX 5090 | **4.05** · 256K<br>33 GiB RAM cache, 90% in memory<br>✅ *59.7 tok/s* · RTX 5090, 64 GB RAM |
| **48 GB** | **3.05** · 256K<br>4 GiB RAM cache, 90% in memory<br>⚪ not tested | **3.05** · 256K<br>6 GiB RAM cache, 95% in memory<br>⚪ not tested | **4.05** · 256K<br>18 GiB RAM cache, 92% in memory<br>⚪ not tested | **4.05** · 256K<br>18 GiB RAM cache, 92% in memory<br>⚪ not tested |
<!-- presets:end -->

## Measured

On an RTX 5090 with 64 GB of RAM, under Windows 11 and WSL2. "Capped" rows hold the engine to a smaller card's
VRAM; real smaller cards also have less bandwidth, so expect less from them. Decode is short answers with thinking
off; prefill is a document of up to 100K tokens ([report](research/vram-scaling-torch-2.15/REPORT.md)). These runs
used an earlier benchmark workload; Repertory's benchmark has its own prompts and documents, so new runs don't
compare 1:1 with them.

<!-- measured:begin -->
| Card (VRAM) | Quant | Context | RAM cache | Prefill | Decode | Measured on |
|---|---|---|---|---|---|---|
| **32 GB** | 3.05 | 256K | 26 GiB | 4,668 tok/s | 72.1 tok/s | [RTX 5090](research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **32 GB** | 3.05 | 256K | 38 GiB | 4,722 tok/s | 74.4 tok/s | [RTX 5090 capped to 32 GB](research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **24 GB** | 3.05 | 256K | 38 GiB | 4,628 tok/s | 62.5 tok/s | [RTX 5090 capped to 24 GB](research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **16 GB** | 3.05 | 256K | 38 GiB | 2,684 tok/s | 41.9 tok/s | [RTX 5090 capped to 16 GB](research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **12 GB** | 3.05 | 64K | 38 GiB | 688 tok/s | 36.1 tok/s | [RTX 5090 capped to 12 GB](research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **12 GB** | 3.05 | 128K, no staging | 38 GiB | 1,933 tok/s | 34.4 tok/s | [RTX 5090 capped to 12 GB](research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **8 GB, no display** | 3.05 | 16K, no staging | 38 GiB | 1,490 tok/s | 26.4 tok/s | [RTX 5090 capped to 8 GB](research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **8 GB, with display** | 3.05 | 16K, no staging | 38 GiB | 1,637 tok/s | 20.3 tok/s | [RTX 5090 capped to 8 GB](research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **32 GB** | 4.05 | 256K | 34 GiB | 5,312 tok/s | 59.7 tok/s | [RTX 5090](research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **32 GB** | 4.05 | 256K | 40 GiB | 4,233 tok/s | 61.0 tok/s | [RTX 5090](research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **32 GB** | 4.05 | 256K | 38 GiB | 4,296 tok/s | 63.0 tok/s | [RTX 5090 capped to 32 GB](research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **24 GB** | 4.05 | 256K | 38 GiB | 4,252 tok/s | 51.4 tok/s | [RTX 5090 capped to 24 GB](research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **16 GB** | 4.05 | 256K | 38 GiB | 1,183 tok/s | 34.5 tok/s | [RTX 5090 capped to 16 GB](research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
| **12 GB** | 4.05 | 32K, no staging | 38 GiB | 1,104 tok/s | 29.6 tok/s | [RTX 5090 capped to 12 GB](research/vram-scaling-torch-2.15/REPORT.md), 64 GB RAM, Windows (WSL2) |
<!-- measured:end -->

## Credits

- [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp: the inference library, and the
  [EXL3 quants of Qwen3.8-Flash-Next](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) on Hugging Face that
  setup downloads.
- [glm53-flash-offload](https://github.com/sybil-solutions/glm53-flash-offload) by sybil-solutions (0xSero): the
  offload engine, with its VRAM, RAM and NVMe expert tiers, the nv2 engine and its kernels.
- [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team: the model.

The project began as a fork of [MaxDevv/lean-qwenflash](https://github.com/MaxDevv/lean-qwenflash), and no code from
it remains.

## License

MIT, for Repertory's own code ([LICENSE](LICENSE)). The bundled `exllamav3/` and `glm53-flash-offload/` keep their
own MIT licenses.
