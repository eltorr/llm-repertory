# loader.py: the model loader

`loader.load()` loads an EXL3 MoE model with exllamav3 and attaches the glm53-flash-offload engine in its NVMe mode
(nv2). The routed experts are not read from the checkpoint: they come from the packed expert store, through a pinned
RAM tier, into a VRAM expert cache, and exllamav3's own kernels compute them. It works for any model the engine
supports: GLM-5.3-Flash and Qwen3.8-Flash-Next (3.05 and 4.05 bpw).

```python
from loader import load
model, config, cache, tok, pools, tables, draft_model, draft_cache = load(
    "./model", "./store/qwen.bin", chunk=2048, cache_tokens=16384, ram_gb=6.0, cpu=False, mtp=0,
    embed_disk=True, cache_quant=4, torch_limit_bytes=None)
```

## Arguments

| argument | meaning |
|---|---|
| `model_dir` | the EXL3 checkpoint directory (`config.json`, safetensors shards, tokenizer) |
| `store` | the expert store `.bin` made from the same checkpoint by `glm53-flash-offload/scripts/pack_glm53_store.py`; its `.json` manifest sits next to it. A store packed from another quant is refused before the load starts. |
| `chunk` | prefill chunk size in tokens (exllamav3 `-chunk_size`) |
| `cache_tokens` | KV cache size in tokens, rounded up to whole 256-token pages |
| `ram_gb` | size of the pinned RAM tier in GiB. `None` sizes it from the memory cgroup cap (`GLM53_NV_RAM_GB=auto`). |
| `cpu` | `True`: the CPU lane computes RAM-resident experts in decode. Only for 3-bit experts (3.05 bpw); a 4-bit store with `cpu=True` raises `ValueError`. |
| `mtp` | `> 0`: load the model's MTP layer as the draft model, drafting `mtp` tokens (exllamav3 `--mtp -ndt mtp`) |
| `embed_disk` | `True`: the n-gram embedding table (Qwen3.8) is streamed from disk, exllamav3's default. `False` loads it into RAM (`--ngram_ram`, tens of GB). The token embedding is not affected. |
| `cache_quant` | KV cache bits (exllamav3 `-cq`); `None`, `0` or `16` for an FP16 cache |
| `torch_limit_bytes` | when given, `torch.cuda.set_per_process_memory_fraction(min(1, torch_limit_bytes / total))` on the current device before anything is allocated. The expert cache then sizes itself inside that limit. |

## Return values

`(model, config, cache, tok, pools, tables, draft_model, draft_cache)`:

- `model`, `config`, `cache`, `tok`: as exllamav3's `model_init.init` returns them
- `pools`: the VRAM expert cache, one `expert_cache.Pool` per GPU (`expert_cache.POOLS`)
- `tables`: the NVMe tier of each pool, `nv2.Nv2` (store, RAM tier, residency and pointer tables). The same object is
  `nv2.NV`.
- `draft_model`, `draft_cache`: the MTP draft model and its cache, or `None` when `mtp == 0`

Callers do not need `pools` and `tables` except to pass them on; `/stats` in `serve.py` reads the same objects.

## Environment

The loader sets these, because its arguments decide them:

| variable | value |
|---|---|
| `GLM53_NV` | `2` (the NVMe mode, nv2) |
| `GLM53_EC` | `1` (the VRAM expert cache) |
| `GLM53_NV_STORE` | `store` |
| `GLM53_NV_RAM_GB` | `ram_gb` (unless `None`) |
| `GLM53_NV_CPU` | `1` when `cpu`, else `0` |

It sets these only when the caller has not:

| variable | default | |
|---|---|---|
| `GLM53_EC_STAGE_GB` | `auto` | prefill staging buffer size: `auto` = one layer's experts |
| `GLM53_EC_STAGE_MIN` | `512` | picks per layer call from which a forward is staged (prefill) |
| `GLM53_NV_CTL_CPU`, `GLM53_NV_READER_CPUS`, `GLM53_NV_CPU_CPUS` | from `docker/preflight.py` | the engine's pinned threads: the measured layout when this CPU set has CPUs 2-39, else one derived from the CPUs this process may use |

Every other setting is the engine's own `GLM53_*` flag with the engine's default
(`glm53-flash-offload/docs/how-it-works.md`, `glm53/nv2.py`, `glm53/expert_cache.py`). The ones callers usually tune:

| variable | default | |
|---|---|---|
| `GLM53_EC_RESERVE_GB` | `2.5` | VRAM left free after the expert cache (activations, prefill) |
| `GLM53_EC_MAX_SLOTS` | none | cap on the VRAM expert cache, in experts |
| `GLM53_EC_ELASTIC_GB` | `0` | VRAM of the cache handed back to torch for each prefill |
| `GLM53_EC_WARM` | none | JSON `{layer key: [score per expert]}` for the initial RAM and VRAM fill. Without it both tiers are filled in expert order. |
| `GLM53_NV_PREFETCH` | `0` | `1`: read the next layer's predicted experts one layer early (decode) |
| `GLM53_NV_PFSIDE` | `0` | `1`: run that prediction on a side stream (single-sequence decode) |
| `GLM53_NV_PUBFAST` | `0` | `1`: faster per-layer request kernel (same request contents) |
| `GLM53_NV_CPU_KERN` | `0` | `1`: the N135 CPU-lane kernel (with `cpu=True`) |
| `GLM53_NV_CPU_CPUS`, `GLM53_NV_CPU_THREADS` | see above, all lane CPUs | CPU lane cores and thread count |
| `GLM53_NV_THREADS`, `GLM53_NV_PIECE_KB` | `16`, `2304` | NVMe reader threads and read size |
| `GLM53_NV_VRING` | `32` | VRAM ring for evicted experts on their way to RAM |
| `GLM53_MAIN_CPUS` | none | pins the calling thread after the load (as `serve.py` does) |

New in this repository:

| variable | default | |
|---|---|---|
| `GLM53_MAX_BATCH` | `1` | sequences the cache keeps recurrent state for (exllamav3 `-ambs`): the most the generator may run at once |

## What it does not do

- It does not start a server or a generator: the caller builds the `AsyncGenerator` (`serve.py`'s `startup`, or its own).
- It does not pack or check the whole store (`pack_glm53_store.py pack | verify | cmp`).
- It does not enable the GLM-5.3-Flash-only kernels (`GLM53_K_HCFUSE`, `GLM53_LA`, `GLM53_TRITON_PIN`, `GLM53_K_OVL`) or the
  all-RAM modes (`GLM53_ZC_VRAM`, `GLM53_CPU_TIER`, which it refuses). It honours the first four if the caller sets them.
- It keeps the MTP layer's experts out of the engine: with `mtp > 0` they load into VRAM with the draft model, as in
  upstream (about 0.9 GB at 3.05 bpw, 1.3 GB at 4.05 bpw). The store holds them (layer `mtp_layer_li`) for later use.
- It does not resize anything after the load.
- Loading uses the GPU: nothing here runs without one, except the argument and store checks.
