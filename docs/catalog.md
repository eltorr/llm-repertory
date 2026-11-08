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
- `id`, such as `qwen3.8-flash-next-exl3-4.05`, and `label`.
- `family`: the name clients use, as `/v1/models` lists it (`qwen3.8-flash-next`).
- `quant`: its short name in the manager: the `quant` setting, bench run specs and run.sh's `QUANT`. Its setup step
  is `m` and the quant without dots (`m405`).
- `source`: the Hugging Face repository, branch and pinned revision. Downloads check every file against the Hub's
  hashes.
- `format`, `bpw`, `engines`.
- `facts`: what sizing needs: the expert layout and size, the VRAM that isn't experts, the KV cache's bytes per
  token, the largest context.
- `disk`, `paths` (the model folder and expert store, inside the engine checkout), `prepare` (the download and pack
  commands), `measured`, `cpu_execution`.
- `offer_untested` and `curation`: see below.

**What reads it:**

| Where | What it takes from the catalog |
|---|---|
| Setup | one model step per model, with its label, sizes, folder, store, repository, revision and branch (`setup/50_model.sh` takes them as options; `setup/check.sh` is told where each model lives) |
| Settings > Model version, Setup | the models on offer, with their curation state |
| The API | the configured model's family, as `/v1/models` lists it |
| Starting the engine | `MODEL_DIR` and `STORE` |
| Sizing | expert bytes, the VRAM that isn't experts, the expert layout, KV bytes per token, the largest context |

**Still written for Qwen,** until a second model needs something else: the preset rule in `sizing.baseline` (4.05
where it keeps 256K context and nearly every expert in memory, else 3.05); the benchmark's `matrix` preset; the
activations per prompt token and the prefill staging thresholds in `sizing.py`; the defaults run.sh,
`setup/50_model.sh`, `setup/check.sh` and `download.py` keep for a run by hand; and the model server's chat format
(`engines/exl3_offload/qwen_chat.py`). The manager runs one engine, `exl3-offload`.

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
  expert counts and bits must match the entry's;
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

They must match the model's reference, `catalog/reference/<model-id>.json`:
- **Answers, exactly:** text, reasoning, tool calls, finish reasons and token counts.
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

A run uses the reference's context and RAM cache, so it starts the same configuration.

**Recording a reference** (`--record`): when there is none, or to accept a change, such as new answers after an
engine update you have looked at. The answers must be fit to be a reference: each ends by itself, the tool call is
the weather lookup for Lisbon, the document's fact is found, and the last one reasons. The reference keeps the
answers, the speeds, the run's configuration, the engine commit and the machine. Recording doesn't pass the check:
the next run compares a fresh start against it.

**The result** goes into the entry's `curation`:
- `status`: `curated` (both checks passed), `failed` (one failed) or `untested` (one not run or undecided, or a
  reference just recorded);
- the date, the machine (GPU, VRAM, RAM, platform; no host names) and the engine commit;
- each check's items, one per thing checked, and its numbers: speeds and their change, the tolerances, the run's
  configuration, and the plan for this PC.

A check that isn't run keeps its earlier result.

**What is offered** (setup and the dashboard):
- curated entries;
- untested entries that have `offer_untested`, shown as untested. The two Qwen quants have it: they predate
  curation and stay offered until their GPU run passes;
- never a failed entry.

The setting *Also offer models that haven't passed curation* (`offer_uncurated`) offers every entry. The model
chosen in Settings always shows there, and Setup always shows a model that is already downloaded.

## Adding a model

1. The engine has to run it: its architecture in the engine's `supports`, its chat format in the model server.
   For this engine, a new family also needs its sizing numbers measured on a GPU.
2. Add the entry: id, family, quant, label, the source pinned to a revision, the facts (from `config.json`, and the
   non-expert VRAM measured), disk, paths and prepare. Leave `offer_untested` out: it isn't offered until it is
   curated.
3. `repertory curate MODEL --checks compat`, and fix what it reports.
4. Download and prepare it: `repertory setup m<quant>`. Setup lists the step with *Also offer models that haven't
   passed curation*; by name it runs either way.
5. `repertory curate MODEL --record`, then `repertory curate MODEL`: the first records the reference, the second
   compares a fresh start with it. Once curated, the entry is offered.
6. Commit the entry with its `curation`, and its reference.
