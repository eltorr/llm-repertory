# Changelog

What changed in Repertory, newest first.

## 2026-11-07: docs and dashboard

- The repository is now `eltorr/llm-repertory` (the old address redirects); the product, the package and the
  command stay `repertory`. A fresh clone lands in `llm-repertory/`.
- The README is reorganized: quick start, how to use it (clients, games, other devices, benchmarks, updating), the
  parts it's built from, dashboard screenshots, presets and a short table of measured speeds. Every measured
  configuration moved to [docs/measured.md](docs/measured.md). The preset table says "measured" or "not tested yet"
  in words. A tagline says what it does ("Local AI that shares your GPU"), and the opening tells how it started:
  scaling the model's resources up and down on a cloud gaming machine.
- The dashboard's prompt speed counts only the prompt tokens the model processed, not the ones the prompt cache
  served (a follow-up in a long conversation showed 15,000 tok/s), and an answer that arrives in one piece, such as
  a tool call, no longer shows an impossible generation speed.
- The Benchmark tab keeps QwenFlash's earlier runs: the settings import copies them.
- A link can open a dashboard tab directly: `/#setup`, `/#bench`, `/#settings`, `/#logs`.
- `tools/screenshots.py` takes the docs' screenshots from the running dashboard without changing anything: it
  aborts every request that isn't a read, scrubs private values from what the page shows, and refuses to save an
  image with any left.

## 2026-11-07: Repertory

The project's first release under its own name, as a repository of its own: a platform for running curated local
models on a PC that is also used for other things, with one adapter per engine.

- **Layout:**
  - `repertory/`: the manager. A service with the OpenAI-compatible API on port 30000 (and the optional second
    port for other devices), the dashboard, setup, benchmarks and the model's lifecycle: full, lean or paused,
    switching in place when only VRAM or RAM is short, and recovering from crashes.
  - `engines/exl3_offload/`: the adapter for ExLlamaV3 with glm53-flash-offload's expert tiers: the model server,
    its launcher (`run.sh`), the VRAM and RAM model (`sizing.py`), the memory watchdogs, the model download and the
    build of the native extensions.
  - `catalog/`: the curated engines and models, each pinned, with how to build, prepare and serve every pair.
  - `exllamav3/` and `glm53-flash-offload/`: the engine as published upstream, under their own MIT licenses.
  - `setup/`, `bin/`, `tools/`, `docs/`, `research/` and `tests/`.
- **The model server** (`engines/exl3_offload/server.py`) serves Qwen3.8-Flash-Next through the engine's own app,
  with the model's chat format in the engine's hooks: its chat template in Jinja's sandbox, its tool calls as OpenAI
  `tool_calls`, prose that streams even when tools are offered, OpenAI's reasoning levels, the Responses API, and
  the model's sampling values where a request leaves them out. New: a prompt that fills the context gets a 400 with
  `context_length_exceeded`, and chat usage reports the tokens reused from the prefix cache
  (`prompt_tokens_details.cached_tokens`).
- **Downloads** (`download.py`) are pinned to one upload per quant, checked against Hugging Face's hashes, resumable,
  and dropped from the page cache as they land, so a 100 GB download doesn't crowd out the model's RAM.
- **The benchmark** has its own workload (protocol `repertory-1`): a heat pump explained in three paragraphs, weather
  station logs of 1K to 64K tokens, and three locker combinations to find in a log of up to 100K tokens. Its results
  don't compare 1:1 with the runs in `research/`, which used an earlier workload.
- **Names:** the manager is `python -m repertory` (`Repertory.cmd`, `repertory.sh`); settings live in
  `%LOCALAPPDATA%\Repertory` or `~/.local/state/repertory`, and the engine's state in `.repertory/`. The engine's
  settings are its upstream `GLM53_*` flags.
- **Moving from QwenFlash:** on its first start, Repertory takes over QwenFlash's settings and learned limits, so the
  dashboard's private link and signed-in devices keep working. On Windows, Repertory keeps an engine checkout of its
  own inside WSL (`/root/repertory`) and links the models and expert stores QwenFlash downloaded instead of
  fetching them again. The ports and the API are unchanged.
- **Same answers as before:** on the same model files, Repertory's engine gives identical answers to the engine
  the earlier fork served, for five greedy requests from a short answer to a 73K-token document
  ([report](research/engine-parity/REPORT.md)). Found on its first start: under WSL2, `cudaHostAlloc` can hand back
  an unaligned address for a small request; the engine now aligns it. Through the manager, the answers are
  identical again and the speeds close to the earlier engine's.
- **The VRAM expert cache sizes itself to the manager's device-wide VRAM limit** (`GLM53_VRAM_LIMIT_BYTES`, set from
  `QWEN_VRAM_LIMIT_BYTES`), as the earlier engine did: sized only inside torch's memory cap, it left out the CUDA
  context and ended 2.5 GB past the limit.
- **Engines and models** ([docs](docs/engines.md)): the engine runs Qwen3.8-Flash-Next and GLM-5.3-Flash; Repertory
  offers Qwen so far. What runs compiled and what in Python, and what is still open.
- **The model server uses this checkout's ExLlamaV3** (`run.sh` puts it first on `PYTHONPATH`), even with a
  Python environment that has another copy installed.
- **One implementation of the API:** the engine's server (`glm53/serve.py`) answers `/v1/responses`, the main API,
  and `/v1/chat/completions` on one pipeline and one streaming splitter; the model server only sets Qwen's hooks
  (template, tool calls, sampling, stop tokens) and no longer replaces the engine's routes. The context check, the
  400s in OpenAI's shape and the prefix-cache figures moved into the engine, for every model. Responses usage now
  reports `input_tokens_details.cached_tokens`, and `/v1/completions` and `/generate` refuse a prompt that fills the
  context with the same 400 (it was FastAPI's error shape, or a stream cut off). Chat Completions answers are
  unchanged.
- **License:** MIT for Repertory's own code.
