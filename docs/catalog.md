# The catalog

What Repertory offers: the engines it runs and the models each one runs, in
[catalog/catalog.json](../catalog/catalog.json), checked against [catalog/schema.json](../catalog/schema.json).
The manager reads it for setup, the dashboard, the API's model name, the engine's paths and sizing. An entry is
offered once it has passed curation: a compatibility check and a regression check on a GPU.

## An entry

**An engine** (`engines`):
- `id`, `name`, `license`, and `source`: a git repository at a pinned commit, or this repository with its adapter
  in `subdir`.
- `supports`: the architectures (as a model's `config.json` names them) and the formats it runs.
- `requires`: platforms, GPU (vendor, compute capability, driver, VRAM), CUDA, Python, PyTorch and the hash-pinned
  lock file.
- `build`, `serve` and `placement`: how to install it, the command and environment that start it (the manager fills
  the `{placeholders}`), and where experts can live and what moves them.

**A model** (`models`):
- `id`, such as `qwen3.8-flash-next-exl3-4.05` or `glm-5.3-flash-exl3-3.05`, and `label`.
- `family`: the name clients use, as `/v1/models` lists it (`qwen3.8-flash-next`, `glm-5.3-flash`).
- `quant`: its short name in the manager: the `quant` setting, bench run specs and run.sh's `QUANT`. Its setup step
  is `m` and the quant without dots or dashes (`m405`, `mglm305` for `glm-3.05`). Quants are unique across models.
- `source`: the Hugging Face repository, branch and pinned revision. Downloads check every file against the Hub's
  hashes.
- `format`, `bpw`, `engines`.
- `facts`: what sizing needs: the expert layout and size, the VRAM that isn't experts, the KV cache's bytes per
  token, the activations of a prefill chunk per token, the largest context (sizing stops at the manager's 262,144).
- `disk`, `paths` (the model folder and expert store, inside the engine checkout), `prepare` (the download and pack
  commands), `measured`, `cpu_execution`.
- `offer_untested` and `curation`: see below.

**What reads it:**

| Where | What it takes from the catalog |
|---|---|
| Setup | one model step per model, with its label, sizes, folder, store, repository, revision and branch (`setup/50_model.sh` takes them as options; `setup/check.sh` is told where each model lives) |
| Settings > Model version, Setup | the models on offer, with their curation state |
| The API | the configured model's family, as `/v1/models` lists it (the manager's and the model server's) |
| Starting the engine | `MODEL_DIR` and `STORE`; run.sh by hand reads them for any quant but Qwen's two |
| Sizing | expert bytes, the VRAM that isn't experts, the expert layout and top-k, KV bytes per token, prefill activations per token, the largest context |

**Per model:** sizing's numbers (the staging buffers, a chunk's activations, the staging-free chunk from top-k, the
elastic share a prefill needs), the model server's chat format (by config.json's `model_type`:
`engines/exl3_offload/qwen_chat.py`, `glm_chat.py`) and the name it lists, and the benchmark's `quants` preset (the
configured model's quants).

**Still written for Qwen:**
- the preset rule in `sizing.baseline` (4.05 where it keeps 256K context and nearly every expert in memory, else
  3.05). With another model configured, the baseline recommendation keeps that model and its context and applies
  only the RAM thresholds; a model that doesn't fit is pointed to Qwen's 3.05;
- the benchmark's `matrix` preset (the README's measured tables);
- the defaults `setup/50_model.sh`, `setup/check.sh` and `download.py` keep for a run by hand with only a quant or a
  branch (`download.py` pins Qwen's branches; GLM's entry passes `--repo` and `--sha`).

The manager runs one engine, `exl3-offload`.

## Curation

```
repertory curate                           the models and their curation state
repertory curate MODEL                     both checks; the result goes into the entry
repertory curate MODEL --checks compat     the compatibility check only (no GPU)
repertory curate MODEL --record            record the regression reference
```

MODEL is an id or a quant. On Windows, run these as `.venv\Scripts\python -m repertory curate ...`; on Linux, as
`./repertory.sh curate ...`. `--no-write` prints the result without writing it; `--json` prints all of it.

**Compatibility** needs no GPU:
- the entry validates against the schema;
- its engine runs on this platform, GPU and driver, and supports the model's architecture and format. The model's
  `config.json` comes from its folder if it is downloaded, else from Hugging Face at the pinned revision; its
  expert counts (`num_experts` or `n_routed_experts`, top-k), MoE layers, MTP layer and bits must match the entry's;
- the pin resolves on Hugging Face (a branch that has moved on is reported, not failed);
- the model fits this PC: the preset's sizing with this machine's VRAM and RAM;
- if it is downloaded: every file is in what the download recorded (`.verified.json`) with the pinned revision's
  hash, none is still downloading, and the expert store is packed. Nothing is hashed again.

**Regression** takes the GPU, through the benchmark: the running model is set aside, this one starts on the whole
card at its full context, the run steps aside while a game needs the GPU, and the model comes back afterwards.
After a warm-up it sends five fixed requests, greedy, each answer to its natural end, thinking off except the last
(the cases of [research/engine-parity](../research/engine-parity/REPORT.md)):
- a short answer, code, a tool call, a 73K-token document with one fact to find, and a short answer with thinking
  on.

They are compared with the model's reference, `catalog/reference/<model-id>.json`:
- **Answers, token by token.** The engine gives the same answers run after run, but exact text would fail any
  change that moves logits in their last bits (the CPU lane computing some experts, a kernel adding them in another
  order, a new exllamav3): where two tokens are nearly tied, a greedy answer flips and goes its own way, which text
  can't tell from a real regression. So after the timed requests, the run replays the
  reference: each answer but the document's is fed to the model as the reference's own token ids after its prompt
  (teacher forcing: one forward pass, through the engine's `/generate`), and at each position the model's 20
  likeliest tokens are compared with the reference's. Per answer:
  - KL(reference || run) at each position, over the tokens either list holds (one missing from a list takes that
    list's smallest logprob less one nat), both renormalized: a mean of at most 0.002;
  - where this run's likeliest token isn't the reference's, the reference must rate it within 1 nat of its own
    likeliest: a near tie that went the other way, not another answer;
  - top-1 agreement and how far the answer token's logprob moved: reported.
- **Decode, against the forward pass.** Teacher forcing takes the path a prompt takes, so a change to decode alone
  (the CPU lane, the expert copies, MTP) doesn't show in it. Each run's own greedy answer is teacher-forced too: at
  each position, decode should have taken the token the forward pass rates likeliest, or one within 1 nat of it,
  and may be off it at no more than 10% of an answer's positions. A recorded reference must pass this too.

  A prompt whose token ids differ from the reference's (a new chat template or tokenizer) fails.

  How the limits were set, on an RTX 5090 (2026-11-07):

  | Run | Mean KL (short, code, tool, reasoning) | Decode off the forward pass's top |
  |---|---|---|
  | Fresh runs of the same setup (two per quant) | 0, 0, 0, 0 | at most 2.5% of positions, 0.11 nat |
  | 3.05 with the CPU lane, against 3.05's reference | 0, 0, 0, 0 | at most 2% of positions, 0.39 nat |
  | 3.05 against 4.05's answers (a different quant) | 0.015, 0.0027, 0.00003, 0.0056 | (3.05's own: as above) |

  The engine is deterministic: the same setup gives the same distributions to the last digit. A whole-quant
  difference shows a mean KL above 0.002 in three of the four answers (the tool call is one the model is sure of).
  The CPU lane changes what decode picks at near ties, never by more than 0.39 nat.
- **The document's answer** must name the technician and the code word. It isn't replayed: one forward pass over
  its 73K tokens would need far more memory than a run has spare.
- Whether each answer is identical (text, reasoning, tool calls, finish reason, token counts) is reported, and
  decides nothing.
- **Speed:** decode (over all the answers) at most 15% slower than the reference's, prefill (the document) at
  most 10%. Faster always passes. The margins come from the recorded runs:
  - decode: the trials of one whole-card run were up to 9.6% apart, the first one colder, though the medians of
    separate runs of one configuration came within 2%; a whole-card run's VRAM cache also follows what the desktop
    holds, and these answers are short. A 20% smaller VRAM cache cost 17%
    ([engine-parity](../research/engine-parity/REPORT.md)), which 15% still catches;
  - prefill: the 73K-token document's first token came within 2.5% across four runs, and 32-64K documents within
    0.3-6.7% between trials.
- Speeds are compared only on the machine the reference was recorded on (the same GPU, card and platform).
  Elsewhere the answers are still compared, and the check stays undecided.

A run uses the reference's context and RAM cache, so it starts the same configuration. If the PC can't spare that
RAM cache when the run starts, the benchmark skips it: the check is not run and keeps its earlier result, as it does
when the run is cut short.

**Recording a reference** (`--record`): when there is none, or to accept a change, such as new answers after an
engine update you have looked at. The answers must be fit to be a reference: each ends by itself, the tool call is
the weather lookup for Lisbon, the document's facts are found, the last one reasons, and the others came with their
token data. A recording keeps the context of the reference it replaces and sizes the RAM cache afresh, 2 GiB under
what the PC can spare then, so the check can start the same cache on a day with a little less free RAM. The
reference keeps the answers, the speeds, the run's configuration, the engine commit and the
machine; for each answer but the document's, its prompt and answer as token ids (from the engine's `/prepare`, which
builds the prompt as the chat routes do, and a greedy `/generate` to the answer's natural end) with each position's
logprobs, the answer token's and the 20 likeliest, to three decimals, a line a position; for the document, the facts
its answer must name. Recording doesn't pass the check: the next run compares a fresh start against it. A reference
recorded before the token comparison has no token data: the check says it has to be recorded again, and stays
undecided (`untested`) until it is, without taking the GPU.

**The result** goes into the entry's `curation`:
- `status`: `curated` (both checks passed), `failed` (one failed) or `untested` (one not run or undecided, or a
  reference just recorded);
- the date, the machine (GPU, VRAM, RAM, platform; no host names) and the engine commit;
- each check's items, one per thing checked, and its numbers: each answer's agreement, KL and logprob change
  (`short_top1`, `short_kl_mean`...), how many answers were identical, speeds and their change, the limits and
  tolerances, the run's configuration, and the plan for this PC.

A check that isn't run keeps its earlier result.

**What is offered** (setup and the dashboard):
- curated entries;
- untested entries that have `offer_untested`, shown as untested. The two Qwen quants have it: they predate
  curation. GLM-5.3-Flash's entry doesn't: it is offered once curated;
- never a failed entry.

The setting *Also offer models that haven't passed curation* (`offer_uncurated`) offers every entry. The model
chosen in Settings always shows there, and Setup always shows a model that is already downloaded.

## Adding a model

1. The engine has to run it: its architecture in the engine's `supports`, its chat format in the model server (a
   module like `qwen_chat.py` or `glm_chat.py`, in `server.CHAT_FORMATS` under its `model_type`). For this engine,
   a new family also needs its sizing numbers measured on a GPU.
2. Add the entry: id, family, quant, label, the source pinned to a revision, the facts (from `config.json`, and the
   non-expert VRAM and prefill activations measured), disk, paths and prepare. Leave `offer_untested` out (or
   false): it isn't offered until it is curated.
3. `repertory curate MODEL --checks compat`, and fix what it reports.
4. Download and prepare it: `repertory setup m<quant>`. Setup lists the step with *Also offer models that haven't
   passed curation*; by name it runs either way.
5. `repertory curate MODEL --record`, then `repertory curate MODEL`: the first records the reference, the second
   compares a fresh start with it. Once curated, the entry is offered.
6. Commit the entry with its `curation`, and its reference.
