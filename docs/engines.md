# Engines and models

What runs where today, and what it takes to add more. The catalog ([catalog/](../catalog)) lists what Repertory
offers, and setup, the dashboard, the API's model name and sizing read it. An entry is only offered once it has
passed curation: a compatibility check and a regression check on the GPU (`repertory curate`, see
[the catalog](catalog.md)).

## Qwen and GLM

- **The engine runs both.** sybil-solutions' glm53-flash-offload was built for GLM-5.3-Flash. Repertory added
  Qwen3.8-Flash-Next and kept GLM working: the engine's size limits cover both models (512 or 288 experts per layer,
  top-10 or top-8).
- **GLM-5.3-Flash 3.05 bpw is in the catalog** (`glm-3.05`, served as `glm-5.3-flash`), **curated on the RTX 5090**
  (2026-11-08): a fresh run reproduced its reference exactly (KL 0 on every answer, the same text), its decode took
  the forward pass's likeliest token at all but 1-6 near ties an answer (at most 0.64 nat below it), and its speed
  came within 3%. Repertory gives it:
  - the engine's `supports` and a catalog entry, with its facts (below);
  - sizing from those facts: its staging buffers (two of one layer's experts: 5.4 GB), the activations of a prefill
    chunk, the staging-free chunk for top-8 (1,792 tokens) and an elastic share big enough for its prefill;
  - its chat format in the model server (`engines/exl3_offload/glm_chat.py`, chosen by config.json's
    `model_type`): the model's template in the sandbox, its `<arg_key>`/`<arg_value>` tool calls (the format
    glm53-flash-offload's `serve.py` parses), generation_config.json's sampling values, its reasoning levels (Low,
    High, Max) and its end-of-turn tokens. The template has no thinking switch: it always opens the think block, so
    thinking off writes it empty, as the engine's own server did;
  - its download (pinned like Qwen's) and its expert store (`store/glm305.bin`, 117.3 GB with the MTP layer).
- **Its facts and where they come from** (the ones marked are to be measured on the GPU):

  | Fact | Value | From |
  |---|---|---|
  | experts per layer, top-k, MoE layers, MTP | 288, 8, 42, yes | config.json: `n_routed_experts`, `num_experts_per_tok`, 45 layers less the 3 dense (`mlp_layer_types`), `num_nextn_predict_layers` 1 |
  | expert bytes (a VRAM slot) | 9,437,184 | gate, up and down at 3 bits (4,096 x 2,048 each); glm53-flash-offload's docs |
  | store record bytes | 9,474,048 | the packer, and glm53-flash-offload's docs |
  | KV bytes per token (2, 4, 8, 16 bits) | 8,096, 9,504, 12,320, 17,600 | ExLlamaV3's MLA cache for its 11 sparse-attention layers: the 512-wide latent, the indexer's key plane and its pooled keys (`exllamav3/cache/mla.py`) |
  | largest context | 1,048,576 (sized up to the manager's 262,144) | config.json `max_position_embeddings` |
  | non-expert VRAM | 8.65 GB | sybil-solutions' runs on an RTX 3090: 10.55-10.95 GB outside the expert cache at a 131,072-token FP16 context, less that context's KV cache. On the RTX 5090 sizing planned 1,728 VRAM experts and the engine took 1,696, the same 1-2% short of the plan as Qwen's runs |
  | prefill activations per token | 0.6 MB | the engine's own estimate for this model (`GLM53_EC_ACT_MB_PER_TOK`); the 73K-token document ran within it |

- **Not run on this engine yet:** MTP and the CPU lane for GLM under Repertory.
- **Still Qwen's:** the preset rule and the benchmark's matrix ([the catalog](catalog.md#an-entry)).
- **Whether GLM is worth it on a given PC** is a separate question. At 3.05 bpw it is about 125 GB, more than a
  32 GB card plus 64 GB of RAM, so more of it comes from the SSD. On the RTX 5090 with 64 GB, at 256K with a 38 GiB
  RAM cache, a token's expert picks came 50% from VRAM, 36% from RAM and 14% from the SSD, and curation's requests
  ran at 8-11 tok/s (Qwen 4.05: 42-62 on the same requests) with the 73K-token document read at about 1,500 tok/s
  (Qwen 4.05: about 6,200). Each GLM token moves about three times the expert bytes of a Qwen one, and a third of
  them aren't in memory. sybil-solutions measured 19.5 tok/s on an RTX 3090 with a 55 GiB RAM cap, 8-channel RAM and
  four NVMe drives in RAID0.
- **GLM-5.3-Flash 2.05 bpw** (`glm-2.05`) is in the catalog too, not offered until it is curated. Its files, pin,
  architecture and fit pass, and its reference is recorded: 20.1 tok/s on curation's requests and the document read
  at 2,262 tok/s, 2.3 and 1.8 times 3.05's. The comparison run is still to do: loading it with the 40 GiB RAM cache
  the manager gives it took free RAM down to 1.4 GB on this PC (9.6 GB once settled). On the
  [head-to-head](../research/head-to-head/REPORT.md) it passed 5 of 7, as 3.05 did. It takes about 164 GB on disk
  (85.2 GB download, 78.4 GB store).
- **Disk:** GLM 3.05 takes about 243 GB (download and store). On Windows, check the free space of the drive that holds
  WSL's disk image, not `df` inside WSL: WSL went down here when that drive filled during the store pack.

## Where the code runs

- **The parts that set the speed are compiled**: ExLlamaV3's and the engine's CUDA kernels, and the engine's C++ host
  code (the NVMe readers, the RAM tier, the per-layer controller). Decode replays CUDA graphs.
- **The adaptive scaling's heavy work is in C++ too.** Shrinking and growing the RAM cache (`ram_shrink`, `ram_grow`)
  run inside the engine: they move the experts it keeps and free or add memory blocks. Python only paces it: the
  background grow takes small steps while requests run, and the free hands back a block at a time.
- **Who does what:**
  - the engine knows how to resize its VRAM and RAM caches and exposes those operations;
  - the model server ([engines/exl3_offload/server.py](../engines/exl3_offload/server.py)) turns them into HTTP
    endpoints;
  - the manager ([repertory/](../repertory)) decides when to scale. It is Python, which is fine: it is idle most of
    the time.
- **Shared inside the engine:** its load sequence (`glm53/offload_stack.py`) serves both the loader and the engine's
  own server, and the engine's server answers the API for any model (`glm53/serve.py`):
  - one pipeline: a request becomes a turn (prompt, tools, sampling), the turn's answer streams as reasoning, content
    and tool calls (`stream_split.py`), and a formatter writes it out: the Responses API first
    (`openai_responses.py`), Chat Completions on the same turn for compatibility;
  - the checks every model gets: a prompt that fills the context is refused before it runs, a request that isn't
    one or that the chat template refuses is a 400, and usage reports the prompt tokens the prefix cache served;
  - the model's own parts are hooks a model server sets: the chat template, the tool-call parser, sampling defaults,
    reasoning levels (`EFFORTS`), stop tokens and the tags. The model server sets the model's (`qwen_chat.py` or
    `glm_chat.py`, by config.json's `model_type`), lists the catalog's name for it, and adds only Repertory's routes.

## Open

- Nothing has moved out of Python yet; the plan is to measure first. Native code would pay off:
  - in the per-token loop, if profiling shows Python costing real time there;
  - in the network handoff of a split across PCs ([notes](research/cross-pc-split.md)), which belongs in C++ from
    the start.
