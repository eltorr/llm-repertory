# Engines and models

What runs where today, and what it takes to add more. The catalog ([catalog/](../catalog)) lists what Repertory
offers; an entry is only offered once it has passed curation (compatibility and regression checks).

## Qwen and GLM

- **The engine runs both.** sybil-solutions' glm53-flash-offload was built for GLM-5.3-Flash. Repertory added
  Qwen3.8-Flash-Next and kept GLM working: the engine's size limits cover both models (512 or 288 experts per layer,
  top-10 or top-8), and GLM's chat and tool-call handling is still there.
- **GLM's side is untested here**: this PC has no GLM model to run it with.
- **Repertory offers only Qwen today**: the catalog, sizing, setup and the model server all target it. Adding GLM
  means:
  - a catalog entry;
  - sizing for its shape (288 experts, top-8);
  - its chat format in the model server;
  - downloading it and packing its expert store;
  - the curation checks and a GPU run.
- **Whether GLM is worth it on a given PC** is a separate question. At 3.05 bpw it is about 125 GB, more than a
  32 GB card plus 64 GB of RAM, so more of it comes from the SSD. sybil-solutions measured 19.5 tok/s on an RTX 3090
  with a 55 GiB RAM cap.

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
  own server, and the Responses API and request clean-up (`openai_responses.py`, `openai_compat.py`) live there too.

## Open

- The engine and the model server both have chat and Responses routes, and the model server's replace the engine's.
  They should become one.
- Nothing has moved out of Python yet; the plan is to measure first. Native code would pay off:
  - in the per-token loop, if profiling shows Python costing real time there;
  - in the network handoff of a split across PCs ([notes](research/cross-pc-split.md)), which belongs in C++ from
    the start.
