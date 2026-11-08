"""Curation: the checks a catalog entry passes before setup and the dashboard offer it.

    python -m repertory curate                                    the catalog's models and their curation state
    python -m repertory curate MODEL [--checks compat|regression|all] [--record] [--no-write] [--json]

MODEL is a model's id (qwen3.8-flash-next-exl3-4.05, glm-5.3-flash-exl3-3.05) or its quant (4.05, glm-3.05).

compat      No GPU. The entry validates; its engine is in the catalog, runs on this platform, GPU and driver, and
            supports the model's architecture and format (config.json from the downloaded folder, else from Hugging
            Face at the pinned revision; its expert counts and bits must match the entry's facts); the pin resolves
            on Hugging Face; the model fits this PC (the manager's sizing with this machine's VRAM and RAM); and, if
            it is downloaded, every file was checked against the Hub's hashes (what download.py recorded in
            .verified.json: nothing is hashed again) and its expert store is packed.
regression  Takes the GPU, through the benchmark (repertory/bench.py, workload "reference"): the running model is set
            aside, this one starts on the whole card at its full context, the run steps aside for games, and the
            model comes back afterwards. The fixed requests (bench.REFERENCE_CASES: a short answer, code, a tool call,
            a 73K-token document with one fact to find, and one with thinking on; greedy, each answer to its natural
            end) are compared with the reference token by token: its answers, as token ids, are teacher-forced on
            this run, and each position's distribution must stay close to the reference's (KL_MEAN_MAX, TIE_GAP);
            the run's own answers, teacher-forced too, show its decode picking what the forward pass rates likeliest,
            near ties aside (TIE_GAP, OFF_TOP_MAX). The document's answer must name its facts. Whether the text is
            identical is reported, not judged. Decode
            and prefill may be at most DECODE_TOLERANCE and PREFILL_TOLERANCE slower than the reference's, which is
            only compared on the machine it was measured on (elsewhere the check can't pass).
            --record writes a new reference from the run (catalog/reference/MODEL.json) when there is none, or to
            accept a change; the next run compares against it. A reference from before the token comparison has no
            token data: until it is recorded again, the check is undecided.

The result goes into the entry's `curation` in catalog/catalog.json: curated when both checks passed, failed when one
failed, untested otherwise (one not run yet, or a reference just recorded). With it: the date, the machine (GPU and
VRAM, RAM, platform; no host names), the engine commit, and each check's items and numbers. A check that isn't run
keeps its earlier result.
"""
import json
import math
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

from . import bench, catalog
from .config import GB, GIB, MAXIMUM

from engines.exl3_offload import sizing  # noqa: E402

# How much slower than the reference a run may be before the regression check fails. Faster never fails.
# Decode: in research/vram-scaling-torch-2.15, the three trials of one whole-card run were up to 9.6% apart (the first,
# with colder caches, slowest), though the medians of separate runs of one configuration came within 2%. A
# whole-card run's VRAM cache also follows what the desktop holds at the time, and these answers are short (90-260
# tokens). 15% stays clear of that, and still catches a loss like the one a 20% smaller VRAM cache caused (17% on the
# short answer, research/engine-parity).
DECODE_TOLERANCE = 0.15
# Prefill: the 73K-token document's first token came within 2.5% across four runs (12.39-12.70 s,
# research/engine-parity), and 32-64K documents within 0.3-6.7% between trials; 10%.
PREFILL_TOLERANCE = 0.10
# The answers are compared token by token, not as text. The engine gives the same answers run after run, but a change
# that moves logits in their last bits (the CPU lane computing some experts, a kernel adding them in another order, a
# new exllamav3) can flip a near tie, and a greedy answer goes its own way from there: exact text would fail such a
# change, and can't tell it from a real regression. So each short case's reference answer, as token ids, is
# teacher-forced on the run (bench.Bench.replay_steps), and each position's distribution (its bench.REFERENCE_TOP
# likeliest tokens) is compared with the reference's:
# - KL(reference || run), over the tokens either list holds, both renormalised over them. A token missing from one list
#   takes that list's smallest logprob less MISSING_MARGIN: it is less likely than every token listed, by an amount the
#   list doesn't tell; a nat below keeps it in the tail, where it moves the KL little either way;
# - flips: where the run's likeliest token isn't the reference's, how far below its own likeliest the reference rates
#   it. A near tie may go either way; a token the reference ranked well below is another answer;
# - top-1 agreement and how far the logprob of the answer's own token moved (reported only).
# Teacher forcing is one forward pass over the answer: the path a prompt takes, not decode's, so a change to decode
# alone (the CPU lane, the expert copies, MTP) doesn't show in it. The run's own greedy answer is teacher-forced too,
# which shows how decode picked: each token's distance below the forward pass's likeliest at that position.
# Measured on an RTX 5090 (2026-11-07, both quants, docs/catalog.md):
# - fresh runs of one setup gave the reference's distributions exactly (KL 0, every logprob the same): the engine is
#   deterministic, so any drift at all comes from a change;
# - the 3.05 quant against 4.05's answers, a real but small difference: a mean KL of 0.0027-0.015 in three cases
#   (0.00003 on the tool call, an answer the model is sure of). KL_MEAN_MAX fails a change that size;
# - decode against the forward pass: healthy runs took its likeliest token at all but 1-3 near ties an answer, at most
#   0.11 nat below it; with the CPU lane (a legitimate change to decode) at most 0.39 nat, 2% of positions. A pick more
#   than TIE_GAP below, or more than OFF_TOP_MAX of an answer's positions off the top, is a broken decode.
KL_MEAN_MAX = 0.002
TIE_GAP = 1.0
OFF_TOP_MAX = 0.10
MISSING_MARGIN = 1.0
HUB = "https://huggingface.co"
PLATFORMS = {"linux": "linux-x86_64", "windows-wsl": "windows-wsl2"}     # host kinds as the catalog names them
CHECKS = ("compat", "regression")


def today():
    return time.strftime("%Y-%m-%d")


def item(name, ok, detail=""):
    return {"name": name, "ok": ok, "detail": str(detail or "")}


def outcome(items):
    """A check's ok: False if an item failed, None if one couldn't be decided, else True."""
    oks = [i["ok"] for i in items]
    return False if False in oks else None if None in oks else True


def version(v):
    return tuple(int(x) for x in "".join(c if c.isdigit() else " " for c in str(v)).split()[:3])


# ---------------------------------------------------------------------------------------------- what checks read
class Hub:
    """Hugging Face, read through its HTTP API (the manager's environment has no huggingface_hub). A revision or file
    that doesn't exist is None; one that can't be reached raises OSError."""

    def __init__(self, timeout=30):
        self.timeout = timeout

    def get(self, path):
        req = urllib.request.Request(HUB + path, headers={"User-Agent": "repertory-curate"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            with e:
                if e.code in (401, 404):       # the Hub answers 401 for a repository that isn't there
                    return None
                raise OSError(f"Hugging Face answered HTTP {e.code}") from e

    def revision(self, repo, rev):
        """The model's info at a revision (or branch), with every file's hash; None if it doesn't exist."""
        data = self.get(f"/api/models/{urllib.parse.quote(repo)}/revision/{urllib.parse.quote(rev, safe='')}"
                        "?blobs=true")
        return None if data is None else json.loads(data)

    def file(self, repo, rev, name):
        return self.get(f"/{urllib.parse.quote(repo)}/resolve/{urllib.parse.quote(rev, safe='')}/"
                        f"{urllib.parse.quote(name)}")


class LocalFiles:
    """The engine checkout's files, read in place: on Linux the manager and the engine share them."""

    def __init__(self, root):
        self.root = Path(root)

    def read(self, rel):
        try:
            return (self.root / rel).read_text(encoding="utf-8")
        except OSError:
            return None

    def exists(self, rel):
        return (self.root / rel).exists()

    def files(self, rel):
        """Every file under a folder (links followed), relative to it; None if there is no such folder."""
        top = self.root / rel
        if not top.is_dir():
            return None
        return sorted((Path(d) / n).relative_to(top).as_posix() for d, _, names in os.walk(top, followlinks=True)
                      for n in names)

    def commit(self):
        try:
            r = subprocess.run(["git", "-C", str(self.root), "rev-parse", "HEAD"], capture_output=True, text=True,
                               timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return r.stdout.strip() or None if r.returncode == 0 else None


class EngineFiles(LocalFiles):
    """Windows: the engine checkout is inside the WSL distro, read through it (cat, find, test, git); small reads
    only, nothing is hashed."""

    def __init__(self, host):
        self.host = host

    def path(self, rel):
        return f"{self.host.engine_dir}/{rel}"

    def read(self, rel):
        r = self.host.run("cat", "--", self.path(rel))
        return r.stdout if r.returncode == 0 else None

    def exists(self, rel):
        return self.host.run("test", "-e", self.path(rel)).returncode == 0

    def files(self, rel):
        top = self.path(rel).rstrip("/")
        if self.host.run("test", "-d", top).returncode != 0:
            return None
        r = self.host.run("find", "-L", top, "-type", "f", timeout=120)
        return sorted(line[len(top) + 1:] for line in r.stdout.splitlines() if line.startswith(top + "/"))

    def commit(self):
        r = self.host.run("git", "-C", self.host.engine_dir, "rev-parse", "HEAD")
        return r.stdout.strip() or None if r.returncode == 0 else None


def files_for(host):
    return LocalFiles(host.engine_dir) if host.kind == "linux" else EngineFiles(host)


class Machine:
    """This PC as the checks see it: the platform (the host's kind), the GPU (NVML's info), the driver and RAM."""

    def __init__(self, platform, gpu=None, driver=None, ram_total=0.0, os_name=None):
        self.platform, self.gpu, self.driver, self.ram_total, self.os_name = platform, gpu, driver, ram_total, os_name

    @classmethod
    def detect(cls, host):
        import psutil
        try:
            gpu = host.gpu.info() if host.gpu else None
        except OSError:
            gpu = None
        driver = host.nvml.driver_version() if host.nvml else None
        return cls(host.kind, gpu, driver, psutil.virtual_memory().total, host.describe().get("os"))

    def describe(self):
        """What the curation record says about the machine: no host names."""
        g = self.gpu or {}
        out = {"gpu": g.get("name"), "vram_gb": round(g["vram_total"] / GB, 1) if g.get("vram_total") else None,
               "ram_gb": round(self.ram_total / GB, 1) if self.ram_total else None, "platform": self.platform}
        if self.os_name:
            out["os"] = self.os_name
        return out


# ------------------------------------------------------------------------------------------------- compatibility
def compat(data, model_id, machine, files, hub, cfg, schema=None):
    """The compatibility check (no GPU): its items, outcome and numbers."""
    schema = schema or json.loads(catalog.SCHEMA.read_text(encoding="utf-8"))
    out = {"ok": None, "date": today(), "engine_commit": files.commit(), "machine": machine.describe(), "items": []}
    items = out["items"]
    errors = catalog.check(data, schema)
    index = next((i for i, m in enumerate(data.get("models") or []) if m.get("id") == model_id), None)
    mine = [e for e in errors if not e.startswith("$.models[") or e.startswith(f"$.models[{index}]")]
    items.append(item("entry", not mine, "valid against catalog/schema.json" if not mine else "; ".join(mine[:6])))
    if errors:                              # nothing else can be read reliably from an invalid catalog
        out["ok"] = False
        out["summary"] = "the catalog doesn't validate"
        return out
    cat = catalog.Catalog(data)
    model = cat.model(model_id)
    engine = cat.engine_of(model)
    items.append(engine_item(engine, machine))
    pin, info = pin_item(model, hub)
    items.append(pin)
    items.append(architecture_item(model, engine, files, hub))
    fits, numbers = fits_item(model, machine, cfg)
    items.append(fits)
    items.append(files_item(model, files, info))
    out["numbers"] = numbers
    out["ok"] = outcome(items)
    failed = [i["name"] for i in items if i["ok"] is False]
    unknown = [i["name"] for i in items if i["ok"] is None]
    out["summary"] = ("compatible" if out["ok"] else f"failed: {', '.join(failed)}" if failed
                      else f"not decided: {', '.join(unknown)}")
    return out


def engine_item(engine, machine):
    """The engine runs on this platform, GPU and driver; and it is the one the manager runs."""
    req, g = engine["requires"], machine.gpu or {}
    why = []
    if engine["id"] != sizing.ENGINE:
        why.append(f"the manager runs only {sizing.ENGINE}")       # hard-coded: one engine adapter today
    if PLATFORMS.get(machine.platform) not in req["platforms"]:
        why.append(f"not on {machine.platform} (runs on {', '.join(req['platforms'])})")
    gpu = req.get("gpu") or {}
    if gpu.get("vendor") == "nvidia" and not g:
        why.append("no NVIDIA GPU found")
    if g:
        cc, need = g.get("compute_capability"), gpu.get("min_compute_capability")
        if need and (not cc or version(cc) < version(need)):
            why.append(f"compute capability {cc} is below {need}")
        if gpu.get("min_vram_gb") and g.get("vram_total", 0) < gpu["min_vram_gb"] * GIB * 0.95:
            why.append(f"{g.get('vram_total', 0) / GIB:.0f} GB of VRAM is below {gpu['min_vram_gb']} GB")
    if gpu.get("min_driver") and machine.driver and version(machine.driver) < version(gpu["min_driver"]):
        why.append(f"driver {machine.driver} is older than {gpu['min_driver']}")
    have = (f"{g.get('name')} (sm_{str(g.get('compute_capability') or '?').replace('.', '')}, "
            f"{g.get('vram_total', 0) / GIB:.0f} GB), driver {machine.driver}" if g else "no GPU")
    return item("engine", not why, f"{engine['id']} on {machine.platform}: " + ("; ".join(why) if why else have))


def pin_item(model, hub):
    """The pinned revision exists on Hugging Face (and where its branch is now). Returns the item and the revision's
    info (its file list, for the files check), or None."""
    src = model["source"]
    try:
        info = hub.revision(src["huggingface"], src["revision"])
    except (OSError, ValueError) as e:
        return item("pin", None, f"couldn't reach Hugging Face: {e}"), None
    if info is None or info.get("sha") != src["revision"]:
        return item("pin", False, f"{src['huggingface']} has no revision {src['revision']}"), None
    detail = f"{src['huggingface']} at {src['revision'][:12]}"
    if src.get("branch"):
        try:
            head = (hub.revision(src["huggingface"], src["branch"]) or {}).get("sha")
        except (OSError, ValueError):
            head = None
        if head and head != src["revision"]:
            detail += f"; branch {src['branch']} has moved on to {head[:12]} (the pin stays)"
        elif head:
            detail += f" (branch {src['branch']})"
    return item("pin", True, detail), info


def model_config(model, files, hub):
    """config.json from the downloaded folder, else from Hugging Face at the pinned revision: (config, where)."""
    text = files.read(f"{model['paths']['model_dir']}/config.json")
    if text is not None:
        cfg, where = json.loads(text), "the downloaded folder"
    else:
        src = model["source"]
        data = hub.file(src["huggingface"], src["revision"], "config.json")
        if data is None:
            raise ValueError("the pinned revision has no config.json")
        cfg, where = json.loads(data), "Hugging Face at the pinned revision"
    if not isinstance(cfg, dict):
        raise ValueError(f"config.json from {where} is not a JSON object")
    return cfg, where


def architecture_item(model, engine, files, hub):
    """The engine supports the model's architecture and format, and config.json agrees with the entry's facts."""
    try:
        cfg, where = model_config(model, files, hub)
    except (OSError, ValueError) as e:
        return item("architecture", None, f"couldn't read config.json: {e}")
    archs = cfg.get("architectures") or []
    text = cfg.get("text_config") or {}
    quant = cfg.get("quantization_config") or {}
    facts = model["facts"]
    why = []
    if not set(archs) & set(engine["supports"]["architectures"]):
        why.append(f"{engine['id']} doesn't run {', '.join(archs) or 'an unnamed architecture'} "
                   f"(it runs {', '.join(engine['supports']['architectures'])})")
    if quant.get("quant_method") and quant["quant_method"] != model["format"]:
        why.append(f"config.json says {quant['quant_method']}, the entry {model['format']}")
    if quant.get("bits") is not None and abs(float(quant["bits"]) - model["bpw"]) > 0.005:
        why.append(f"config.json says {quant['bits']} bits, the entry {model['bpw']} bpw")
    for key, fact in (("num_experts", "experts_per_layer"), ("n_routed_experts", "experts_per_layer"),
                      ("num_experts_per_tok", "experts_per_token")):
        value = cfg.get(key, text.get(key))
        if value is not None and value != facts[fact]:
            why.append(f"config.json's {key} is {value}, the entry's {fact} {facts[fact]}")
    layers = moe_layers(text or cfg)
    if layers is not None and layers != facts["moe_layers"]:
        why.append(f"config.json has {layers} MoE layers, the entry's moe_layers {facts['moe_layers']}")
    mtp = mtp_layers(text or cfg)
    if mtp is not None and "mtp" in facts and bool(mtp) != facts["mtp"]:
        why.append(f"config.json has {mtp} MTP layer(s), the entry's mtp is {str(facts['mtp']).lower()}")
    detail = (f"{', '.join(archs)} ({cfg.get('model_type')}), {quant.get('quant_method', '?')} "
              f"{quant.get('bits', '?')} bits, from {where}")
    return item("architecture", not why, "; ".join(why) if why else detail)


def moe_layers(text):
    """The main model's MoE layers as config.json's text config describes them (the MTP layer aside): those its
    mlp_layer_types call sparse (GLM-5.3), else every layer past the first_k_dense_replace dense ones (every layer for
    Qwen3.8); None when it doesn't say."""
    n = text.get("num_hidden_layers")
    kinds = text.get("mlp_layer_types")
    if isinstance(kinds, list) and isinstance(n, int):
        return sum(1 for k in kinds[:n] if k == "sparse")
    if not isinstance(n, int):
        return None
    return n - int(text.get("first_k_dense_replace") or 0) - len(text.get("mlp_only_layers") or ())


def mtp_layers(text):
    """The MTP draft layers config.json names (GLM-5.3: num_nextn_predict_layers, Qwen3.8: mtp_num_hidden_layers), or
    None."""
    for key in ("num_nextn_predict_layers", "mtp_num_hidden_layers"):
        if isinstance(text.get(key), int):
            return text[key]
    return None


def fits_item(model, machine, cfg):
    """The model fits this PC: the manager's sizing (hardware.gpu_budget, sizing.fit, as the preset sizes a start)."""
    from . import hardware
    if model["quant"] not in sizing.QUANTS or not machine.gpu or machine.platform not in sizing.OS_IDLE:
        return item("fits", False, "no NVIDIA GPU, or a platform or engine the manager can't size"), {}
    budget = hardware.gpu_budget(machine.gpu, cfg)
    f = sizing.fit(model["quant"], budget, machine.ram_total, machine.platform)
    numbers = {"vram_budget_gb": round(budget / GB, 1), "ram_gb": round(machine.ram_total / GB, 1)}
    if f is None:
        return item("fits", False, f"{budget / GB:.1f} GB of VRAM for the model is too little at any context"), numbers
    numbers.update(context=f["context"], prefill_chunk=f["chunk"], staging=f["staging"], vram_experts=f["slots"],
                   ram_cache_gib=f["ram_cache_gib"], in_memory_pct=round(100 * f["in_memory"], 1))
    detail = (f"{f['context'] // 1024}K context{'' if f['staging'] else ', staging-free'}, {f['slots']:,} experts in "
              f"{budget / GB:.1f} GB of VRAM, a {f['ram_cache_gib']} GiB RAM cache: {f['in_memory']:.0%} of the "
              f"experts in memory")
    if not f["ram_fits"]:
        return item("fits", False, f"not enough RAM ({machine.ram_total / GB:.0f} GB) for {detail}"), numbers
    return item("fits", True, detail), numbers


def files_item(model, files, info):
    """If the model is downloaded: every file checked against the Hub's hash (download.py's .verified.json, compared
    with the pinned revision's file list when the Hub answered) and the expert store packed."""
    folder, store = model["paths"]["model_dir"], model["paths"]["store"]
    listing = files.files(folder)
    if listing is None:
        return item("files", True, "not downloaded: setup checks every file as it downloads")
    try:
        verified = json.loads(files.read(f"{folder}/.verified.json") or "{}")
    except ValueError:
        verified = {}
    partial = [f for f in listing if f.endswith((".part", ".part.json"))]
    # FILE.http-state.json: the resume state QwenFlash's downloader kept beside each file, not part of the model
    present = [f for f in listing if f not in partial and f != ".verified.json" and not f.startswith(".cache/")
               and not f.endswith(".http-state.json")]
    why = []
    if partial:
        why.append(f"{len(partial)} file(s) still downloading")
    unchecked = [f for f in present if f not in verified]
    if unchecked:
        why.append(f"{len(unchecked)} file(s) never checked against the Hub's hash ({', '.join(unchecked[:3])}): "
                   "run the model's setup step (it checks them)")
    if info:
        remote = {s["rfilename"]: (s.get("lfs") or {}).get("sha256") or s.get("blobId")
                  for s in info.get("siblings") or []}
        missing = [n for n in remote if n not in present]
        wrong = [n for n, digest in remote.items() if n in verified and verified[n] != digest]
        if missing:
            why.append(f"{len(missing)} file(s) of the pinned revision missing ({', '.join(missing[:3])})")
        if wrong:
            why.append(f"{len(wrong)} file(s) checked against another revision ({', '.join(wrong[:3])})")
    if not (files.exists(store) and files.exists(str(PurePosixPath(store).with_suffix(".json")))):
        why.append(f"the expert store {store} isn't packed: run the model's setup step")
    against = "the pinned revision's list" if info else "download.py's record (the Hub's list unavailable)"
    return item("files", not why, "; ".join(why) if why else
                f"{len(present)} files, each checked against its hash ({against}); store {store} packed")


# ---------------------------------------------------------------------------------------------------- regression
def load_reference(model_id, folder=None):
    p = (Path(folder) / f"{model_id}.json") if folder else catalog.reference_path(model_id)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def save_reference(ref, folder=None):
    p = (Path(folder) / f"{ref['model']}.json") if folder else catalog.reference_path(ref["model"])
    p.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_json(p, ref, indent=1, flat=True)            # the token data a line a position
    return p


def reference_case(c):
    """A run's case as a reference keeps it: the answer and its timing, then what the comparison needs: the facts the
    document's answer must name, or the other answers' token data."""
    out = {k: c.get(k) for k in ("name",) + bench.REFERENCE_ANSWER + ("ttft_s", "decode_tps", "prefill_tps")}
    if c["name"] in bench.REFERENCE_FACTS:
        out["expect"] = list(bench.REFERENCE_FACTS[c["name"]])
    elif c.get("tokens"):
        out["tokens"] = c["tokens"]
    return out


def replayable(ref):
    """Whether a reference has what the token comparison replays: token data for every case not checked by facts.
    References from before it have none."""
    cases = [c for c in ref.get("cases") or [] if "expect" not in c and c.get("name") not in bench.REFERENCE_FACTS]
    return bool(cases) and all(complete_tokens(c.get("tokens")) for c in cases)


def complete_tokens(t):
    """A case's token data is whole: the prompt's ids, and the answer's with logprobs for each of its positions."""
    t = t or {}
    n = len(t.get("answer") or ())
    return bool(n and t.get("prompt")) and all(len(t.get(k) or ()) == n
                                               for k in ("logprobs", "top_ids", "top_logprobs"))


def run_machine(rec):
    """The machine a bench record ran on, as curation describes it."""
    h = rec.get("host") or {}
    return {"gpu": h.get("gpu"), "vram_gb": rec.get("card_total_gb"), "ram_gb": h.get("ram_total_gb"),
            "platform": h.get("platform") or "linux"}


def same_machine(a, b):
    """Speeds compare only on the machine the reference was measured on: the same GPU, card and platform."""
    return (a.get("gpu") == b.get("gpu") and a.get("platform") == b.get("platform")
            and abs((a.get("vram_gb") or 0) - (b.get("vram_gb") or 0)) < 1.0)


def reference_spec(model, ref):
    """The bench run: the whole card at the reference's context (else the model's largest), with its RAM cache."""
    context = (ref or {}).get("run", {}).get("context") or min(model["facts"]["max_context"], MAXIMUM["context"])
    ram = (ref or {}).get("run", {}).get("ram_gib")
    return f"{model['quant']}:card:{context}" + (f":ram={ram}" if ram else "")


def sanity(cases):
    """What a reference must show before it is recorded: every answer ended by itself, the tool call is the weather
    lookup for Lisbon, the document's facts were found, thinking was on in the last case, and the other answers came
    with their token data."""
    by = {c["name"]: c for c in cases}
    items = []
    capped = [c["name"] for c in cases if c.get("finish") not in ("stop", "tool_calls")]
    items.append(item("natural ends", not capped, "every answer ended by itself" if not capped
                      else f"cut short: {', '.join(capped)}"))
    calls = (by.get("tool") or {}).get("tool_calls") or []
    ok = bool(calls) and calls[0].get("name") == "get_weather" and "lisbon" in str(calls[0].get("arguments")).lower()
    items.append(item("tool call", ok, f"{calls[0]['name']}({calls[0]['arguments']})" if calls else "no tool call"))
    doc = (by.get("document") or {}).get("content", "")
    items.append(item("document", not bench.missing_facts(bench.REFERENCE_FACTS["document"], doc),
                      doc.strip()[:160] or "no answer"))
    items.append(item("thinking", bool((by.get("reasoning") or {}).get("reasoning")),
                      "reasoned before answering" if (by.get("reasoning") or {}).get("reasoning") else "no reasoning"))
    short = [c for c in cases if c["name"] not in bench.REFERENCE_FACTS]
    bare = [c["name"] for c in short if not complete_tokens(c.get("tokens"))]
    n = sum(len(c["tokens"]["answer"]) for c in short if c["name"] not in bare)
    items.append(item("tokens", bool(short) and not bare,
                      f"{n:,} answer tokens with their logprobs" if not bare else f"no token data: {', '.join(bare)}"))
    if short and not bare:                         # a reference from a broken decode would make it the standard
        off = [c["name"] for c in short if not decode_passes(decode_metrics(c["tokens"]))]
        items.append(item("decode", not off, "each answer's decode took the forward pass's likeliest tokens, near "
                                             "ties aside" if not off else f"decode strayed from the forward pass: "
                                                                          f"{', '.join(off)}"))
    return items


def first_difference(a, b):
    """Where two answers part, briefly."""
    if isinstance(a, str) and isinstance(b, str):
        i = next((k for k, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
        return f"at character {i}: {a[max(0, i - 20):i + 40]!r} became {b[max(0, i - 20):i + 40]!r}"
    return f"{json.dumps(a)[:80]} became {json.dumps(b)[:80]}"


def logsumexp(xs):
    top = max(xs)
    return top + math.log(sum(math.exp(x - top) for x in xs))


def position_kl(ref_ids, ref_lps, run_ids, run_lps):
    """KL(reference || run) at one position, from the two top lists: over the tokens either holds, a token missing from
    one at that list's smallest logprob less MISSING_MARGIN, both renormalised."""
    ref, run = dict(zip(ref_ids, ref_lps, strict=True)), dict(zip(run_ids, run_lps, strict=True))
    floor_ref, floor_run = min(ref_lps) - MISSING_MARGIN, min(run_lps) - MISSING_MARGIN
    union = list(ref.keys() | run.keys())
    p, q = [ref.get(t, floor_ref) for t in union], [run.get(t, floor_run) for t in union]
    zp, zq = logsumexp(p), logsumexp(q)
    return max(0.0, sum(math.exp(a - zp) * ((a - zp) - (b - zq)) for a, b in zip(p, q, strict=True)))


def below_top(ids, lps, token):
    """How far below a list's likeliest token `token` is: one the list doesn't hold, at its smallest less
    MISSING_MARGIN, as the KL takes it."""
    lp = dict(zip(ids, lps, strict=True)).get(token, min(lps) - MISSING_MARGIN)
    return max(0.0, lps[0] - lp)


def replay_metrics(tokens, replay):
    """A reference answer replayed on a run (its "replay": logprobs, top_ids, top_logprobs a position) against the
    reference's own ("tokens"): the positions, how many agree on the likeliest token and the share, how far below its
    likeliest the reference rates the run's where they differ (at most), KL(reference || run) mean and max, and how
    far the answer token's logprob moved, mean and max."""
    n = len(tokens["answer"])
    kl = [position_kl(tokens["top_ids"][j], tokens["top_logprobs"][j], replay["top_ids"][j],
                      replay["top_logprobs"][j]) for j in range(n)]
    flips = [below_top(tokens["top_ids"][j], tokens["top_logprobs"][j], replay["top_ids"][j][0]) for j in range(n)
             if replay["top_ids"][j][0] != tokens["top_ids"][j][0]]
    moved = [abs(a - b) for a, b in zip(replay["logprobs"], tokens["logprobs"], strict=True)]
    return {"positions": n, "agreed": n - len(flips), "top1": (n - len(flips)) / n,
            "flip_gap_max": max(flips, default=0.0), "kl_mean": sum(kl) / n, "kl_max": max(kl),
            "logprob_delta_mean": sum(moved) / n, "logprob_delta_max": max(moved)}


def replay_passes(m):
    return m["kl_mean"] <= KL_MEAN_MAX and m["flip_gap_max"] <= TIE_GAP


def decode_metrics(tokens):
    """How a run's greedy decode picked, from its own answer teacher-forced ("tokens": the forward pass's view): the
    positions where it didn't take the forward pass's likeliest token, their share, and how far below it the token it
    took was, at most."""
    gaps = [max(0.0, row[0] - own) for row, own in zip(tokens["top_logprobs"], tokens["logprobs"], strict=True)]
    off = sum(g > 0 for g in gaps)
    return {"decode_off": off, "decode_off_share": off / len(gaps), "decode_gap_max": max(gaps)}


def decode_passes(d):
    return d["decode_gap_max"] <= TIE_GAP and d["decode_off_share"] <= OFF_TOP_MAX


def replay_case(ref_case, case, numbers):
    """A short case: the reference's answer as this run teacher-forced it, against the reference's: (ok, detail). Its
    figures go into numbers as CASE_top1, CASE_kl_mean and so on. A prompt whose tokens changed fails (a new chat
    template or tokenizer): what the model sees is no longer the reference's."""
    name, tokens, replay = ref_case["name"], ref_case["tokens"], case.get("replay") or {}
    if not complete_tokens(dict(replay, prompt=tokens["prompt"], answer=tokens["answer"])):
        return None, "the reference's answer wasn't replayed in this run"
    if not complete_tokens(case.get("tokens")):
        return None, "this run's own answer wasn't teacher-forced, so its decode can't be checked"
    m, d = replay_metrics(tokens, replay), decode_metrics(case["tokens"])
    numbers.update({f"{name}_top1": round(m["top1"], 4), f"{name}_decode_off_share": round(d["decode_off_share"], 4),
                    **{f"{name}_{k}": float(f"{v:.3g}") for k, v in m.items()
                       if k in ("flip_gap_max", "kl_mean", "kl_max", "logprob_delta_mean", "logprob_delta_max")},
                    f"{name}_decode_gap_max": float(f"{d['decode_gap_max']:.3g}")})
    ok = replay_passes(m) and decode_passes(d)
    flips = (f"{m['positions'] - m['agreed']} flipped, at most {m['flip_gap_max']:.2g} nat below the reference's "
             f"likeliest (at most {TIE_GAP:g})" if m["agreed"] < m["positions"] else "none flipped")
    detail = (f"KL mean {m['kl_mean']:.2g} (at most {KL_MEAN_MAX:g}), max {m['kl_max']:.2g}; top-1 agreement "
              f"{m['agreed']}/{m['positions']}, {flips}; decode took the forward pass's likeliest token at "
              f"{len(case['tokens']['answer']) - d['decode_off']}/{len(case['tokens']['answer'])} positions (off at "
              f"most {OFF_TOP_MAX:.0%}), the others at most {d['decode_gap_max']:.2g} nat below it (at most "
              f"{TIE_GAP:g}); the answer token's logprob moved {m['logprob_delta_mean']:.2g} on average, at most "
              f"{m['logprob_delta_max']:.2g}")
    prompt = (case.get("tokens") or {}).get("prompt")
    if prompt is not None and prompt != tokens["prompt"]:
        ok, detail = False, f"its prompt's tokens differ from the reference's (chat template or tokenizer); {detail}"
    return ok, detail


def facts_case(facts, case):
    """A case checked by what its answer must name: (ok, detail)."""
    missing = bench.missing_facts(facts, case.get("content"))
    if missing:
        return False, f"doesn't name {' or '.join(missing)}: {(case.get('content') or '').strip()[:120]!r}"
    return True, f"names {' and '.join(facts)}"


def compare(ref, rec):
    """A run's answers and speeds against the reference: (items, numbers). An answer passes by its tokens (or, for
    the document, its facts); whether its text is identical is said, not judged."""
    now = {c["name"]: c for c in (rec.get("reference") or {}).get("cases") or []}
    items, same = [], 0
    numbers = {"kl_mean_limit": KL_MEAN_MAX, "tie_gap_limit": TIE_GAP, "off_top_limit": OFF_TOP_MAX}
    for c in ref["cases"]:
        n = now.get(c["name"])
        if n is None:
            items.append(item(c["name"], False, "not in this run"))
            continue
        diffs = [k for k in bench.REFERENCE_ANSWER if n.get(k) != c.get(k)]
        same += not diffs
        text = (f"text not identical ({', '.join(diffs)} differ; {diffs[0]} "
                f"{first_difference(c.get(diffs[0]), n.get(diffs[0]))}), which alone doesn't fail" if diffs
                else "text identical")
        facts = c.get("expect") or bench.REFERENCE_FACTS.get(c["name"])
        ok, detail = facts_case(facts, n) if facts else replay_case(c, n, numbers)
        items.append(item(c["name"], ok, f"{detail}; {text}"))
    numbers.update({"identical": f"{same}/{len(ref['cases'])}", "decode_tolerance_pct": round(100 * DECODE_TOLERANCE),
                    "prefill_tolerance_pct": round(100 * PREFILL_TOLERANCE)})
    here, there = run_machine(rec), ref.get("machine") or {}
    for key, tol, name in (("decode_tps", DECODE_TOLERANCE, "decode"), ("prefill_tps", PREFILL_TOLERANCE, "prefill")):
        base, got = (ref.get("speed") or {}).get(key), (rec.get("reference") or {}).get(key)
        numbers.update({f"{name}_tps": got, f"{name}_ref_tps": base})
        if not same_machine(there, here):
            items.append(item(name, None, f"not compared: the reference was measured on {there.get('gpu')} "
                                          f"({there.get('platform')}), this run on {here.get('gpu')}"))
        elif not base or not got:
            items.append(item(name, None, "not measured"))
        else:
            change = (got - base) / base
            numbers[f"{name}_change_pct"] = round(100 * change, 1)
            items.append(item(name, change >= -tol, f"{got:,.1f} tok/s against {base:,.1f} ({change:+.1%}; at most "
                                                    f"{tol:.0%} slower passes)"))
    return items, numbers


def regression(model, runner, record=False, folder=None):
    """The regression check: run the reference requests through the benchmark (runner(specs, workload) -> records),
    then compare them with the model's reference, or with record, write the reference from them."""
    existing = load_reference(model["id"], folder)
    ref = None if record else existing
    out = {"ok": None, "date": today(), "items": []}
    if ref is None and not record:
        out.update(summary="no reference yet: record one with --record, then run this again to compare",
                   items=[item("reference", None, f"no {catalog.reference_path(model['id']).name}")])
        return out
    if ref is not None and ref.get("protocol") != bench.REFERENCE_PROTOCOL:
        out.update(summary=f"the reference is of protocol {ref.get('protocol')}, the requests now "
                           f"{bench.REFERENCE_PROTOCOL}: record it again",
                   items=[item("reference", None, "recorded with other requests")])
        return out
    if ref is not None and not replayable(ref):           # the GPU isn't taken for a run that can't decide
        out.update(summary="the reference predates the token-by-token comparison: record it again with --record, "
                           "then run this again to compare",
                   items=[item("reference", None, f"{ref.get('date')}'s reference has no token data to replay")])
        return out
    # a new reference keeps the configuration of the one it replaces, so speeds stay comparable over time
    records = runner([reference_spec(model, existing)], "reference")
    rec = (records or [None])[0]
    if not rec:
        out.update(summary="the benchmark didn't run", items=[item("run", None, "refused or no result")])
        return out
    out.update(engine_commit=rec.get("engine_commit"), machine=run_machine(rec))
    if rec.get("interrupted") or rec.get("error") == "cancelled":
        out.update(summary=f"cut short ({rec.get('error')}): run it again", items=[item("run", None, rec.get("error"))])
        return out
    if rec.get("error"):
        out.update(ok=False, summary=f"the run failed: {rec['error']}", items=[item("run", False, rec["error"])])
        return out
    run = {"label": rec.get("label"), "context": rec.get("context"), "ram_gib": rec.get("ram_gib"),
           "vram_gb": rec.get("usable_gb"), "vram_experts": (rec.get("engine") or {}).get("slots"),
           "startup_s": rec.get("startup_s")}
    if record:
        items = sanity(rec["reference"]["cases"])
        out["items"] = items
        if outcome(items) is not True:
            out.update(ok=False, summary="not recorded: the answers aren't fit to be a reference")
            return out
        ref = {"model": model["id"], "protocol": bench.REFERENCE_PROTOCOL, "date": today(),
               "machine": run_machine(rec), "engine_commit": rec.get("engine_commit"), "run": run,
               "stack": rec.get("stack"),
               "speed": {k: rec["reference"].get(k) for k in ("decode_tps", "prefill_tps")},
               "cases": [reference_case(c) for c in rec["reference"]["cases"]]}
        path = save_reference(ref, folder)
        out["summary"] = (f"reference recorded in {path.name}: decode {ref['speed']['decode_tps']} tok/s, prefill "
                          f"{ref['speed']['prefill_tps']} tok/s; run the check again to compare against it")
        out["numbers"] = {"decode_tps": ref["speed"]["decode_tps"], "prefill_tps": ref["speed"]["prefill_tps"],
                          "recorded": True, **{k: v for k, v in run.items() if k != "label"}}
        return out
    items, numbers = compare(ref, rec)
    numbers.update({k: v for k, v in run.items() if k != "label"})
    # a whole-card run's VRAM cache follows what the desktop holds at the time: the reference's, to compare
    numbers["vram_experts_ref"] = (ref.get("run") or {}).get("vram_experts")
    out.update(items=items, numbers=numbers, ok=outcome(items))
    bad = [i["name"] for i in items if i["ok"] is False]
    open_ = [i["name"] for i in items if i["ok"] is None]
    answers = f"answers within the tolerance ({numbers['identical']} identical as text)"
    if out["ok"]:
        out["summary"] = f"{answers}, speed within the tolerance"
    elif bad:
        out["summary"] = f"regressed: {', '.join(bad)}"
    elif {c["name"] for c in ref["cases"]}.isdisjoint(open_):
        out["summary"] = f"{answers}; not decided: {', '.join(open_)}"
    else:
        out["summary"] = f"not decided: {', '.join(open_)}"
    return out


# ------------------------------------------------------------------------------------------------------- results
def status_of(checks):
    """curated when compat and regression both passed, failed when one failed, else untested."""
    oks = [c.get("ok") for c in checks.values()]
    if False in oks:
        return "failed"
    if all((checks.get(k) or {}).get("ok") is True for k in CHECKS):
        return "curated"
    return "untested"


def curation(model, results):
    """The entry's new curation record: this run's checks over the ones it didn't run."""
    checks = dict((model.get("curation") or {}).get("checks") or {})
    checks.update(results)
    ran = [r for r in (results.get("regression"), results.get("compat")) if r]     # the regression's first
    rec = {"status": status_of(checks), "date": today()}
    machine = next((r["machine"] for r in ran if r.get("machine")), None)
    if machine:
        rec["machine"] = machine
    rec["engine_commit"] = next((r["engine_commit"] for r in ran if r.get("engine_commit")), None)
    rec["checks"] = {k: checks[k] for k in CHECKS if k in checks}
    return rec


def run(model_id, cfg, checks="all", record=False, write=True, machine=None, files=None, hub=None, runner=None,
        catalog_path=None, reference_dir=None):
    """Curate one model: the checks asked for, and (write) the result into the catalog. Returns the curation record
    and this run's checks."""
    path = Path(catalog_path or catalog.CATALOG)
    data = json.loads(path.read_text(encoding="utf-8"))
    model = next((m for m in data.get("models") or [] if model_id in (m.get("id"), m.get("quant"))), None)
    if model is None:
        raise KeyError(f"no model {model_id!r} in the catalog")
    results = {}
    if checks in ("compat", "all"):
        results["compat"] = compat(data, model["id"], machine, files, hub or Hub(), cfg)
    if checks in ("regression", "all"):
        if results.get("compat", {}).get("ok") is False:
            results["regression"] = None                   # not run: the GPU isn't taken for an entry that can't pass
        else:
            results["regression"] = regression(model, runner, record, reference_dir)
    ran = {k: v for k, v in results.items() if v is not None}
    rec = curation(model, ran)
    if write:
        catalog.save_curation(model["id"], rec, path)
    return rec, results


# ----------------------------------------------------------------------------------------------------------- CLI
def report(model, rec, results, written):
    mark = {True: "ok  ", False: "FAIL", None: "--  "}
    lines = [f"{model['id']} ({model['label']})"]
    for name, title in (("compat", "compatibility"), ("regression", "regression")):
        if name not in results:
            continue
        r = results[name]
        if r is None:
            lines.append(f"{title}: not run (compatibility failed)")
            continue
        lines.append(f"{title}: {r.get('summary', '')}")
        lines += [f"  {mark[i['ok']]} {i['name']:<13} {i.get('detail', '')}" for i in r["items"]]
    lines.append(f"status: {rec['status']}" + (" (written to catalog/catalog.json)" if written else " (not written)"))
    return "\n".join(lines)


def listing(cat):
    rows = [f"{'model':<34} {'quant':<8} {'status':<9} {'checked':<10} offered"]
    for m in cat.models:
        c = m.get("curation") or {}
        rows.append(f"{m['id']:<34} {m['quant']:<8} {catalog.status(m):<9} {c.get('date', '-'):<10} "
                    f"{'yes' if catalog.offered(m) else 'no'}" + ("" if c else " (no curation yet)"))
    return "\n".join(rows)


def main(cfg, a, bench_records):
    """`repertory curate` (repertory/__main__.py): bench_records(cfg, runs, workload) runs the benchmark through the
    running service, or in this process."""
    cat = catalog.load()
    if not a.model:
        print(listing(cat))
        print("curate one with: python -m repertory curate MODEL [--checks compat|regression|all] [--record]")
        return 0
    model = cat.find(a.model)
    if model is None:
        print(f"no model {a.model!r} in the catalog")
        return 2
    if a.record and a.checks == "compat":
        print("--record needs the regression check (--checks regression or all)")
        return 2
    from .host import make_host
    host = make_host(cfg)
    rec, results = run(model["id"], cfg, a.checks, a.record, not a.no_write, Machine.detect(host), files_for(host),
                       Hub(), lambda runs, workload: bench_records(cfg, runs, workload))
    if a.json:
        print(json.dumps({"curation": rec, "results": results}, indent=1))
    else:
        print(report(model, rec, results, not a.no_write))
    return exit_code(results, a.record)


def exit_code(results, record=False):
    """0 when every check this run made passed (a reference recorded counts, if nothing failed), else 1."""
    oks = [r.get("ok") for r in results.values() if r is not None]
    recorded = record and ((results.get("regression") or {}).get("numbers") or {}).get("recorded")
    if None in results.values():                     # the regression wasn't run: compatibility failed
        return 1
    return 0 if all(ok is True for ok in oks) or (recorded and False not in oks) else 1
