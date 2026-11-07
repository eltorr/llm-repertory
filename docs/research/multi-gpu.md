# Multi-GPU for Repertory: research notes (2026-11-04)

**[M]** = measured or documented (source given). **[E]** = my estimate. Nothing was run on a GPU.

## Short answer

- **This engine can't use two GPUs today.** That isn't because only llama.cpp or vLLM can. The vendored exllamav3 already supports layer split (`-gs`) and tensor parallel (`-tp`, expert-parallel for MoE) for `qwen4_exp` (`architecture/qwen4_exp.py`: `"supports_tp": True`) [M]. Our offload layer (`nv2`, `expert_cache`, guards) is what is built for one GPU.
- **Best fit: a layer (pipeline) split.** Each GPU owns about half the layers and caches their experts. With **two 32 GB cards at 3.05 bpw, the whole model except the n-gram table (52.4 GB of shards [M]) fits in VRAM**, so no offload engine is needed. 4.05 bpw (about 68 GB) still needs offload.
- **Peer-to-peer (P2P) is not needed.** Every design below moves only KB per token between the GPUs.
- **Measure first.** The gain depends on how much of today's 16-20 ms per token goes to cache misses.

## Single-GPU assumptions [M]

- `glm53/nv2.py`: the global `NV` is overwritten by each `attach_pool`. It asserts that the pool holds layers `0..L-1` with `first == 0` and the full `E`. `kernels/nv2/nv2_host.cpp` has `static Engine* G`: one controller, one reader pool, one RAM tier.
- `nv_tier.py` `HostArena` uses `cudaHostRegister(...,2)`: Mapped but not Portable, so the pinned memory belongs to one GPU's context. The WSL path, `wsl_pinned.py`, is already Portable\|Mapped.
- The loader and `engines/exl3_offload/server.py`: `mem_get_info` and the memory fraction act on the current device only, with one `QWEN_VRAM_LIMIT_BYTES` for all GPUs. `engines/exl3_offload/memory_guard.py` and the manager (`repertory/host.py`) read NVML index 0. `k_lookahead.py` uses `current_device()`. The build targets one architecture, the GPU's own (`setup/30_build.sh`).
- Already per-device: `expert_cache.attach_modules` builds one `Pool` per device, and the loader keeps a bounce slot per device.
- **Today's risk:** if a second GPU is visible and exllamav3's autosplit puts a layer on it, the `nv2` assert fails. `run.sh` therefore pins the server to one GPU (`QWEN_GPU`, default 0) when several are visible (upstream `docker/entrypoint.sh` warns the same way).

## What limits decode here

- 5090 at 4.05 bpw: 7,520 VRAM slots (18.48 GB, about 31 % of 24,576 experts), a 34 GiB RAM tier, CPU lane off [M] (`logs/backend.log`).
- On the 4090, 19 % of experts in VRAM already served about 410 of 480 picks per token (about 86 %) [M] ([4090-final-sweep.json](https://github.com/MaxDevv/lean-qwenflash/blob/fa9a772d9aa9b94a5653a2c6880eeb97af0d2e9d/research/4090-final/results/4090-final-sweep.json), published with MaxDevv/lean-qwenflash, which this project began as a fork of).
- At 60 tok/s, 10 % misses is only about 7 GB/s over a PCIe 5.0 x16 link [E]. The limit is **miss latency on the per-layer critical path plus non-MoE work**, not link bandwidth. A second GPU helps only by removing misses.

## Options

| option | split | benefit | cost / risk |
|---|---|---|---|
| **A. Layer split, fully resident** (2 x 32 GB, 3.05) | Stock exllamav3 `-gs a,b`: layers 0-23 with PLE and embeddings on GPU0, 24-47 and the head on GPU1. About 40 KB of hyper-connection state crosses per token. | No misses. No 26 GB RAM tier. No PCIe staging in prefill. Plausibly 70-110 tok/s if misses are 40-50 % of today's step [E]. | Tight fit [E]: 52.4 GB + about 1.7 GB of Q4 KV at 262k + 2 CUDA contexts + 2.2 GB desktop is about 58-60 of 64 GB. Small engineering effort. Untested with `qwen4_exp`. Doesn't fit 4.05. |
| **B. Layer split + per-GPU expert cache** | As A, but offload stays on and each GPU caches its own layers' experts. | For 4.05 or a 24 GB card: about 40-45 GB of expert cache instead of 18.5 GB, so about 65-75 % of 4.05 experts resident instead of 31 % [E]. | Medium to large effort (see the recommended path). Misses on a chipset x4 link cost about 7x more [E]. |
| **C. Tensor / expert parallel** (`-tp`) | MoE experts are divided between the GPUs (`block_sparse_mlp.py`: `channel_unit="experts"`). The native backend reduces through host shared memory, so no P2P is needed (`model_tp_backend.py`) [M]. | Both GPUs work in prefill and batched decode. | Spawns worker processes (`model_tp.py`), so the offload patches don't run inside them, and `nv2` rejects partial expert ranges. Fully-resident only. About 96 host round trips per token: likely slower than A for a single stream [E]. |
| **D. B70-style expert server** | Warm experts live on GPU1, which computes their picks. | Upstream: 17.5 to 26.4 tok/s [M] (`docs/b70-tier.md`). | Probably a loss here (below). |

### How the B70 tier works [M]

At warm-up, a separate server process (`b70tier/b70srv.py`, Intel SYCL kernels) loads the next 3,000 experts after the VRAM set onto an Arc Pro B70. They are dropped from RAM, so the tiers stay exclusive. In each decode layer:

1. The planner gives B70-resident picks the lane `LN_B70` (`nv2_host.cpp`).
2. The **CPU-lane worker** writes their input rows into a tmpfs ring (`ring.py`).
3. The B70 copies them in, computes, and copies the result back.
4. The worker adds the result into the fp32 partial that the 3090 adds back.

No P2P. Cost: 0.17 ms per layer call + 0.021 ms per expert, hidden behind the slower CPU lane.

**Reusing it for a second NVIDIA GPU:**

- **Not as-is:**
  - `GLM53_B70=1` asserts `GLM53_NV_CPU=1`. Repertory keeps the CPU lane off because it measured slower (29 vs 18-22 tok/s, [MaxDevv's 4090 report](https://github.com/MaxDevv/lean-qwenflash/blob/fa9a772d9aa9b94a5653a2c6880eeb97af0d2e9d/research/4090-final/REPORT.md)).
  - The ring is hard-coded for H = 4096, E = 288, MAXP = 64. Our model has H = 2560 and E = 512.
  - The server is XPU-only.
  - The upstream gain was measured on 9.44 MB experts on a 3090 with DDR4.
- **Even rebuilt, it probably loses.** Our step is about 0.35 ms per layer. A host round trip of 0.1-0.2 ms on each of 48 layers adds 5-10 ms per token unless it fully overlaps GPU0's own work. A RAM miss over PCIe 5.0 costs only about 50 µs [E].

## Other engines

| engine | support for this model | multi-GPU | fits 2 consumer GPUs + 64 GB RAM? |
|---|---|---|---|
| llama.cpp | Merged (PR #27742). | `-sm layer` / `row` with `-ts`. **`-sm tensor` is disabled for qwen4exp** (commit 36b1015). `--n-cpu-moe` computes experts **on the CPU** (PR #15077). | Only with CPU experts on dual-channel DDR5: slower than ours [E]. |
| vLLM | Needs >= 0.30 (nightly). TP or TP+EP. **PP not supported with the N-gram embedding.** | Even sharding. Without P2P it falls back to NCCL. | No. FP8 is 173 GiB. The community W4A16 is about 66 GB GPU-resident and needs at least 110 GB RAM. |
| SGLang | From source (PR #36497). | TP/EP. Targets H200 / B200-class GPUs. | No. |

Sources: [llama.cpp #27742](https://github.com/ggml-org/llama.cpp/pull/27742), [36b1015](https://github.com/ggml-org/llama.cpp/commit/36b10154383b60eb15baac2c7a40d2a5f784faa7), [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next), [W4A16](https://huggingface.co/VnimanieAI/Qwen3.8-Flash-Next-W4A16), [SGLang](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next).

## Hardware and software requirements

- **Lanes.** X870E boards split the CPU's x16 into x16 or x8/x8. M.2_2 and USB4 often share those lanes, and the bottom slots are usually chipset-fed PCIe 4.0 x4 ([MSI](https://us.msi.com/blog/pcie-lane-bifurcation-and-bandwidth-sharing-on-msi-x870e-x870-motherboards)). x8/x8 halves the 5090's link: that hurts B and prefill, not A.
- **P2P.** Not supported on the 5090 ([NVIDIA forum](https://forums.developer.nvidia.com/t/p2p-issue-using-two-rtx-5090-gpus/326776), [vLLM](https://discuss.vllm.ai/t/what-means-there-is-no-p2p-support/1928)), so traffic goes through host RAM, which is fine here. The P2P-patched open kernel module is a Linux driver and can't apply under WSL2 [E].
- **Mixed GPUs.**
  - Rebuild exllamav3_ext and the three offload extensions for both architectures (for example `"8.6;12.0"`).
  - Use uneven `-gs` values.
  - A 4090 or 3090 gives 56 GB total: marginal for A, fine for B [E].
  - A 2080 Ti (11 GB, Turing, PCIe 3.0) adds little and takes cuBLAS fallbacks (`hyperconnections.py`).
- **Power and space.** 575 W per 5090 (vendor spec), so a 1,400-1,600 W system. Many boards space slots too closely for 3-slot cards. A power cap likely costs little in decode [E].
- **WSL2** ([NVIDIA guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)):
  - Docker supports only `--gpus all`.
  - Pinned memory is "limited". We already pin 26-34 GiB.
  - NVML can't change settings: set power limits from Windows.
  - P2P is undocumented.

## Recommended path

0. **Measure now, on the 5090 alone.**
   - Record `/stats` `decode_per_token` (VRAM, RAM and NVMe hits, `reply_wait_ms`) and an `nsys` window (`GLM53_PROF`) to get the miss-free step time.
   - Move the Windows desktop to the Ryzen iGPU. That frees about 2.2 GB, roughly 900 more 4.05 expert slots.
1. **If misses are 30 % or more of the step, build A** with a second 32 GB card at 3.05.
   - Add a loader path that skips `nv_tier` / `expert_cache` and passes `-gs`.
   - Make the memory guard and the switcher watch every GPU.
   - A/B `-gs` against `-tp`.
2. **For 4.05 or a 24 GB card, build B.**
   - Give the `nv2` Engine per-layer device ownership: one request ring and control region per GPU, one shared exclusive RAM tier, and one controller (only one GPU is active at a time in single-stream decode).
   - Register pinned memory Portable\|Mapped.
   - Set per-device VRAM budgets and guards.
   - Use an explicit layer-to-device map, because autosplit sees almost no expert bytes.
   - Give each GPU its own staging buffers and lookahead.
3. **Skip D** unless profiling shows large per-layer slack.

**Risks:** layer split is unvalidated with the QSA indexer, recurrent checkpoints and CUDA-graph decode. A may not fit at 262k context. Also: WSL pinned-memory limits, board lane sharing, power.

## Open questions to measure

1. Misses per token and the time spent waiting on them, at 3.05 and 4.05.
2. Miss-free step time (the ceiling for A).
3. Does `-gs` load `qwen4_exp` and match single-GPU outputs (KL / top-1)?
4. VRAM peak per GPU for A at 262k context.
5. Cross-GPU hop cost under WSL2 without P2P, and what `can_device_access_peer` reports.
6. Prefill at x8/x8 vs x16.
