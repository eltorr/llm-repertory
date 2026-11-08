# Repertory manager

The manager runs the model as a service on a PC that is also used for other things, such as gaming, and adapts the
model's resources to them:

- **One endpoint:** an OpenAI-compatible API on `http://127.0.0.1:30000/v1` ([api.md](api.md)).
- **Shares the GPU:**
  - loads the model when a client calls it;
  - sizes it to the free VRAM and RAM;
  - scales it back or pauses it when games or other GPU apps need the card;
  - comes back to full when they're done.
- **A local dashboard** shows the state, speed, GPU and RAM use, setup, benchmarks and logs.
- **Setup** for native Linux, or Windows with the engine in WSL2.

The manager is plain Python (`psutil`, plus `pywin32` on Windows) and never touches the GPU itself.

## Start it

| | Windows (engine in WSL2) | Linux |
|---|---|---|
| Start and open the dashboard | double-click `Repertory.cmd` | `./repertory.sh open` |
| Run in the foreground | `.venv\Scripts\python -m repertory serve` | `./repertory.sh` |
| Start at login | Settings: *Start the service when I sign in to Windows* | Settings: *Start at login* |

From a terminal, which helps on headless machines:

```
repertory detect          platform, GPU, CPU, RAM, WSL, engine and recommendations
repertory setup           the setup steps and their status
repertory setup STEP      run one step: wsl, wslconfig, distro, driver, base, engine, build, and one per catalog
                          model: m305, m405
repertory plan            this machine's preset, and what fits at each quant
repertory status          what the service is doing (--json for everything)
repertory bench           benchmark this machine (see Benchmarks)
repertory curate [MODEL]  the catalog's curation state, or curate a model (see docs/catalog.md)
```

On Windows, run these as `.venv\Scripts\python -m repertory ...`; on Linux, as `./repertory.sh ...`.

## Setup

| Step | Windows + WSL2 | Linux |
|---|---|---|
| WSL 2 | installs WSL; needs administrator approval, and a first install needs a restart | - |
| WSL memory limit | lets WSL use most of the RAM (`.wslconfig`) | - |
| Ubuntu 24.04 | installs the distro on the drive you pick, with systemd | - |
| NVIDIA driver | - | checks for driver 595 or newer (CUDA 13.2), and otherwise shows the install commands |
| Build tools + CUDA 13.2 | inside WSL | Ubuntu 22.04+ or Debian 12+; uses `sudo`, or prints the command to run |
| Engine + Python environment | a clone of this checkout inside WSL, at the same commit | this checkout |
| Compile GPU kernels | for the local GPU (Turing to Blackwell); after an update that changed the kernels, it shows **Update available** and rebuilds only those (`FULL_BUILD=1` rebuilds exllamav3 too) | same |
| Models | one step per catalog model on offer: 3.05 bpw (~85 GB download + 47 GB store) or 4.05 bpw (~107 + 62 GB); downloads resume | same |

On Windows, when you commit a change here, *Engine + Python environment* shows **Update available**. Running it
again syncs the WSL clone; only committed code is synced.

**Model versions:**
- **Source:** [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) on
  Hugging Face.
- **Pinned:** each quant is pinned to one upload (its `source.revision` in the [catalog](catalog.md)), so every
  install gets the same files, even after newer uploads appear. A newer upload is only used once it has been tested
  and the pin changed.
- **Offered:** setup and the Settings tab offer the models that passed curation; the two Qwen quants are offered
  as untested until their GPU run passes ([curation](catalog.md#curation)).
- **Changing the pin:** running a model step again skips files that are already there. So after a pin change,
  delete that model's folder and its store file (`store/qwen*.bin`, `.json`), then run the step again.

## How it shares the GPU

| Profile | VRAM | RAM cache | When |
|---|---|---|---|
| **Full** | all that's free, minus ~10% headroom (at most 2.4 GB) | with the VRAM cache, 90% of the experts, as far as RAM allows | nothing else needs the GPU |
| **Lean** | about half the card | 8 GiB | a game or another app needs VRAM or RAM |
| **Paused** | 0 | 0 | a watched program runs, or VRAM or RAM is critically short |

- **Scaling back:** happens between requests, after about 10 s of pressure. When only VRAM is short, the engine
  shrinks its VRAM expert cache in place, in about half a second: experts held only in VRAM go back to the RAM
  cache first, and the RAM cache stays as it is (if RAM runs short later, it gets lean's: see below). Coming back
  to full grows it the same way, unless the model started lean: then it restarts, to get full's larger RAM cache.
  RAM pressure, a staging-free plan, or an engine that hasn't been updated restart the model in the lean profile
  instead (about 45 s, and the caches start empty). The setting is
  *Switch between full and lean without a restart* (`live_resize`); to check it on this PC, run
  `repertory bench current --workload resize` ([measured](../research/live-resize/REPORT.md)).
- **The RAM cache without a restart** (*Resize the RAM cache without a restart too*, `live_ram_resize`): RAM
  pressure shrinks the RAM cache to lean's in place as well, holding requests under a second. The most
  recently used experts stay, and the memory is freed in chunks of about 0.3 GB in the background; under WSL,
  Windows gets it back within a minute, as after a restart. Coming back to full grows it again in the background
  (9-16 s for 27 GiB between requests; while one runs it adds only a little at a time, so requests don't wait
  for it; right after a shrink, it first takes back the memory not freed yet), and a RAM cache sized while RAM was
  short grows the same way. A grow stops early if free RAM nears the
  scale-back line, or if this system runs low ([measured](../research/live-resize/REPORT.md#the-ram-cache-in-the-background)). The engine's memory limits (its systemd scope and the RAM
  guard) move with it. It needs kernels compiled after the update (*Compile GPU kernels*); with older ones, these
  switches restart as before. The resize workload measures it too, including how soon Windows sees the freed RAM.
- **Pausing:** happens about 3 s into heavy pressure, once the current request finishes (2 minutes at most).
- **Coming back:**
  - A paused model wakes on the next request.
  - A lean model returns to full after the GPU has been clear for 30 s.
- **Your choice wins:** *Start lean* stays lean until you start full. A request made under pressure wakes the model
  lean and holds off auto-switching for 5 minutes.
- **The RAM cache:**
  - It holds 90% of the experts together with the VRAM cache. The rest are rarely picked: at 4.05 bpw on a 32 GB
    card, 34 and 40 GiB ran at the same speed, and the last 6 GiB only cost the PC RAM.
  - A restart reuses the RAM the stopped model held. Under WSL2, Windows only sees freed memory again after a few
    minutes, and sizing from its figure gave restarts a third of the cache.
  - Under WSL2, a start right after a stop waits until Windows has most of the stopped model's RAM back (all but a
    fifth, or 4 GB), for up to 90 s. Windows gets it back over 60-80 s, and a model loading meanwhile took the PC
    down to 4 GB free. A switch takes up to a minute longer for it.
  - If full runs short of RAM and drops to lean, the next full starts get a smaller RAM cache for an hour, so
    full -> lean -> full doesn't repeat.
  - A full model whose RAM cache was sized while RAM was short grows it, with a restart, once 8 GiB more fits.
- **Short prompts** (up to 1,638 tokens, or each new turn of a conversation that adds no more) skip prefill
  staging. Staging reads every expert that is in neither VRAM nor the RAM cache from the NVMe, and doesn't keep it;
  short prompts read what they need and keep it in the RAM cache, so longer prompts after them read from RAM. A
  1-2K-token prompt gets its first token in 1.4-2.6 s instead of 6-7.
- **Thresholds:** all are in Settings, scaled to the card; 0 means automatic.

[`sizing.py`](../engines/exl3_offload/sizing.py) turns a VRAM budget into a plan: the expert cache, the prefill chunk, and whether
staging fits.

**Recovery:**

- **Model kept running:** the model runs detached, so restarting the manager doesn't stop it.
- **Crashes:** a crashed model is restarted after 10 s, 30 s and 90 s. After three failures in 10 minutes, the
  manager stops and shows the error.
- **Out of memory:**
  - VRAM: the next start uses 2 GB less.
  - RAM: the RAM cache shrinks by 4 GiB.
  - *Settings > Forget learned limits* clears these.

## Benchmarks

The **Benchmark** tab and `repertory bench` measure this machine: they start the model in each configuration, time
answers and prompt reading, and record the results. The model stops for the run, requests get an error meanwhile,
and the model comes back afterwards. Only the PC itself can start one.

```
repertory bench                      this machine's settings on the whole card
repertory bench quants               3.05 and 4.05 bpw
repertory bench lean                 the lean profile
repertory bench 4.05:22:262144       4.05 bpw held to 22 GB of VRAM (a 24 GB card), 256K context
repertory bench matrix               8-32 GB cards emulated on a 32 GB one: the README's measured table (~1 h;
                                     needs 64 GB of RAM: every run gets the same 38 GiB RAM cache)
repertory bench 3.05:card:262144:cpu engine features, off unless named (experimental, not in the manager yet):
                                     :mtp=N the MTP layer drafts N tokens a step; :cpu[=CPUS] the CPU lane
                                     computes RAM experts (3-bit only); :prefetch early NVMe reads
  --workload quick                   ~2 min a run instead of 3-6
  --workload resize                  full -> lean -> full without a restart, then three quick cycles: answers,
                                     a 32K document of its own and the engine's cache check at each size; then
                                     the same with the RAM cache shrunk to lean's and grown back, and Windows'
                                     free RAM up to 60 s after the shrink (whole-card runs)
  --workload resize-stress           the RAM cache's background work where it could go wrong: requests while it
                                     grows, a shrink that cancels a grow, a grow stopped partway, rapid
                                     switching, requests around a shrink and while its memory goes back, each
                                     with how soon a request gets its first token (whole-card runs)
  --out research/NAME/results        also write the results there
repertory bench --resume             continue the last benchmark that was cut short
repertory bench --list | --table FOLDER | --sizing FOLDER
```

- **What a run measures** (protocol `repertory-1`):
  - thinking off and greedy decoding; every answer ends on its own (the output is never capped);
  - a warm-up, then three trials of one answer (how a heat pump works, in three short paragraphs) and documents of
    1K, 8K, 32K and 64K tokens (a weather station log), each read and answered in one sentence. Each trial's text
    differs from the start, so nothing comes from the prompt cache;
  - a retrieval test: three locker combinations at 15%, 50% and 85% of a log of up to 100K tokens;
  - startup time, VRAM, and where the experts came from (VRAM, RAM or NVMe).

  The runs in [research/](../research/README.md) used the prompts and documents of the original `benchmark.py` of
  [MaxDevv/lean-qwenflash](https://github.com/MaxDevv/lean-qwenflash), which this project began as a fork of.
  Results from this workload don't compare 1:1 with those, or with his; a run is only compared with earlier runs of
  the same protocol.
- **The PC comes first.** When a game or another app needs the GPU (an app on the watchlist, another app's GPU load,
  or its VRAM crowding the card) or free RAM falls below `ram_pause_gb`, the run steps aside: its server stops, and
  the run starts again from the beginning once the PC has been free for `recover_after_s`. A run interrupted four
  times is recorded as interrupted, and the benchmark moves on.
- **RAM:** as the manager does, a run drops the page cache in WSL once its server is up and after it stops (loading
  leaves gigabytes there that Windows counts as used). A run with a set RAM cache (`:ram=GIB`) starts only if the PC
  can spare it, as the manager would size it; just after a stop it first waits for WSL to hand the last server's
  memory back. Otherwise it is skipped with the RAM it would have had. Each run starts, as the manager's model does,
  once Windows has most of the last server's RAM back. Each run records the least free RAM while its server was up
  (`ram_free_min_gb`).
- **Resuming:** a benchmark cut short (cancelled, the manager restarted, the PC rebooted) continues with
  `repertory bench --resume` or the tab's **Resume** button. The runs left go into the same folder, with the same
  workload.
- **A run spec** is `QUANT:VRAM_GB|card:CONTEXT[:nostage][:ram=GIB]`. A VRAM figure holds the engine to that much, as on
  a smaller card. It can't emulate a smaller card's bandwidth, so real smaller cards are slower.
- **Results:** one JSON file per run in `bench/` in the settings folder, saved after every step, with the machine
  and its stack (GPU, driver, CUDA, PyTorch, commit). Each run shows the change since the last run of the same
  configuration, which makes a quick before-and-after check of a change.
- Without the service running, `repertory bench` runs in the terminal, and leaves the model stopped.

## Where things are

| Path | What |
|---|---|
| `repertory/` | the manager: `service.py`, `host.py` (Windows + WSL2 or Linux), `hardware.py`, `setup.py`, `bench.py`, `ui/` |
| `engines/exl3_offload/` | the engine's adapter: `server.py` (the model server), `run.sh` (starts it, sized from the hardware), `sizing.py` (the VRAM and RAM model, shared with the manager), `memory_guard.py`, `download.py`, `build_ext.py` |
| `glm53-flash-offload/`, `exllamav3/` | the engine itself, as published upstream |
| `catalog/` | the models and engines, how to prepare and serve each pair, and each model's curation; `reference/` holds the regression references ([catalog](catalog.md)) |
| `bin/repertory-backend` | runs the engine detached (a systemd unit where available) |
| `setup/` | the install steps, shared by both platforms; `CUDA_VER` and `DRIVER_MIN` in `lib.sh` |
| `tools/` | the checks (`scan.sh`), the package locks (`lock.sh`) and the pre-push hook |
| settings and logs | Windows `%LOCALAPPDATA%\Repertory`, Linux `~/.local/state/repertory`; on its first start Repertory takes over QwenFlash's settings (its earlier name) |

## Security

- **Dashboard:** every port needs the dashboard token, from the private link's cookie or a Bearer header. It's
  `dashboard_token` in `config.json`.
- **Other websites:** browser requests from other sites, and DNS rebinding, are refused
  ([details](api.md#control-api-the-dashboards-not-openai)).
- **Settings:** paths, programs, ports and the token change only in `config.json`. Other devices can't run setup,
  quit, change autostart, make a new link or change the PC's name in it.
- **Model server:** it listens on `127.0.0.1` and answers only requests that carry the manager's key (`backend.key`
  in the settings folder).
- **Files:** the settings folder is private to your user. On Windows, setup limits the WSL folder to you, SYSTEM and
  Administrators.
- **Packages and downloads:**
  - every Python package is pinned by hash;
  - NVIDIA's repository key is checked before it's trusted;
  - model files are checked against Hugging Face's hashes.
- **The API (`/v1`) takes no key,** like a local OpenAI server: anything that reaches the port can use the model.
  Only expose the network port, on a private network such as a tailnet.

## Other machines

Every port uses the same layout: the API is under `/v1`, and the web page is at `/`. Port 30000 listens on
`127.0.0.1` only. To reach the PC from your phone and other devices, use the network port:

1. **Turn it on:** set `"network_port"` in `config.json` to any free port, such as `8080`, then restart the service.
2. **Expose it,** for example on a tailnet:

   ```
   tailscale serve --bg --tcp=8080 tcp://127.0.0.1:8080
   ```

3. **API:** clients use `http://<tailnet-host>:8080/v1`.
4. **Dashboard:**
   1. On the PC, copy the private link from **Settings > Other devices**.
   2. Open it once on each device; the device then remembers it.
   3. **New link** makes the old one stop working.

   Without the link, the port's root shows a page where you can paste the link, and the dashboard's controls
   refuse requests.

Don't expose port 30000 itself; it only answers requests addressed to `127.0.0.1` or `localhost`. On the PC, the
dashboard needs the same link: `Repertory.cmd` and `repertory open` sign the browser in.

Over SSH, `ssh -L 30000:127.0.0.1:30000 host` reaches everything instead.
