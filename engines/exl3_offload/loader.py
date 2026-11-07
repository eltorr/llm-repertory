"""Load an EXL3 MoE model with exllamav3 and attach the glm53-flash-offload engine in its NVMe mode (nv2).

    model, config, cache, tok, pools, tables, draft_model, draft_cache = load(model_dir, store, ...)

The routed experts are never read from the checkpoint: they come from the packed expert store (`store`, made by
glm53-flash-offload/scripts/pack_glm53_store.py) through a pinned RAM tier into a VRAM expert cache, and are computed
by exllamav3's own kernels (with cpu=True, partly by the engine's AVX2 CPU lane). Contract, arguments and tuning flags:
LOADER.md next to this file. The engine is tuned through its own GLM53_* environment variables; load() sets only the
ones its arguments decide and the NVMe mode needs, and keeps every value the caller has set.
"""
import argparse
import importlib.util
import json
import math
import os
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENGINE = REPO / "glm53-flash-offload" / "glm53"
PREFLIGHT = REPO / "glm53-flash-offload" / "docker" / "preflight.py"
EXLLAMAV3 = REPO / "exllamav3"
PAGE = 256   # exllamav3 cache page: the cache holds whole pages

# the NVMe mode's own settings (docker/entrypoint.sh, nvme modes), used where the caller has not set them: prefill
# stages each layer's non-cached experts into VRAM (buffers of one layer's experts) for calls of >= 512 picks
MODE_DEFAULTS = {"GLM53_EC_STAGE_GB": "auto", "GLM53_EC_STAGE_MIN": "512"}
MANIFEST_KEYS = ("record_bytes", "trellis_bytes", "experts_per_layer", "n_layers", "layers", "fields", "vram_slot",
                 "total_bytes")


def read_store(store):
    """The store's manifest (<name>.json next to <name>.bin), checked against the .bin's size."""
    store = Path(store)
    meta = store.with_suffix(".json") if store.suffix == ".bin" else Path(str(store) + ".json")
    if not store.is_file():
        raise FileNotFoundError(f"expert store not found: {store}")
    if not meta.is_file():
        raise FileNotFoundError(f"expert store manifest not found: {meta}")
    man = json.loads(meta.read_text())
    missing = [k for k in MANIFEST_KEYS if k not in man]
    if missing:
        raise ValueError(f"{meta}: not an expert store manifest (missing {missing})")
    size = store.stat().st_size
    if size != man["total_bytes"] or size != man["record_bytes"] * man["experts_per_layer"] * man["n_layers"]:
        raise ValueError(f"{store}: {size} bytes, its manifest says {man['total_bytes']}")
    return man


def expert_bits(man):
    """Bits per weight of the routed experts (EXL3 trellis: last dim = 16 x K)."""
    ks = {f["shape"][-1] // 16 for f in man["fields"] if f["name"].endswith(".trellis")}
    if len(ks) != 1:
        raise ValueError(f"mixed expert bit widths in the store: {sorted(ks)}")
    return ks.pop()


def check_store_matches(model_dir, man):
    """The store was packed from this checkpoint's quant: the first store layer's expert 0 has the same trellis shapes
    in the model (one safetensors header read). Catches a 3.05bpw store next to a 4.05bpw model before the long load."""
    idx = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    key = man["layers"][0]["key"]
    want = {f["name"]: f["shape"] for f in man["fields"] if f["name"].endswith(".trellis")}
    fn = idx.get(f"{key}.experts.0.gate_proj.trellis")
    if fn is None:
        raise ValueError(f"{model_dir} has no {key}.experts.0.gate_proj.trellis: the store is for another model")
    with open(Path(model_dir) / fn, "rb") as f:
        head = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    got = {n: head.get(f"{key}.experts.0.{n}", {}).get("shape") for n in want}
    if got != want:
        raise ValueError(f"store experts {want} != model experts {got}: the store was packed from another quant")


def plan_env(store, man, ram_gb, cpu, environ=None):
    """The GLM53_* variables this load sets ({name: value}); pure, so tests can call it. The arguments decide the NVMe
    mode, the store, the RAM tier and the CPU lane; MODE_DEFAULTS and the CPU layout fill only what is unset."""
    env = dict(os.environ if environ is None else environ)
    if cpu and expert_bits(man) != 3:
        raise ValueError(f"cpu=True needs 3-bit experts (the CPU lane's AVX2 kernels decode K=3 trellises); this store "
                         f"has {expert_bits(man)}-bit experts: load it with cpu=False")
    if env.get("GLM53_ZC_VRAM") or env.get("GLM53_CPU_TIER", "0") not in ("", "0"):
        raise ValueError("GLM53_ZC_VRAM / GLM53_CPU_TIER belong to the engine's all-RAM modes, not the NVMe mode: "
                         "unset them")
    out = {"GLM53_NV": "2", "GLM53_EC": "1", "GLM53_NV_STORE": str(store), "GLM53_NV_CPU": "1" if cpu else "0"}
    if ram_gb is not None:
        out["GLM53_NV_RAM_GB"] = repr(float(ram_gb))
    out.update({k: v for k, v in MODE_DEFAULTS.items() if not env.get(k)})
    # nv2's pinned threads (controller, NVMe readers, CPU lane): the measured layout when this CPU set has it, else
    # one derived from it (docker/preflight.py, as the container entrypoint does); variables already set are kept
    env.update(out)
    spec = importlib.util.spec_from_file_location("glm53_preflight", PREFLIGHT)
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)
    layout, _how = preflight.layout_vars(env)
    out.update({k: v for k, v in layout.items() if not env.get(k)})
    return out


def model_args(model_dir, chunk, cache_tokens, mtp, embed_disk, cache_quant):
    """exllamav3 model_init arguments for this load (its own parser, so every other option keeps its default)."""
    from exllamav3 import model_init
    ap = argparse.ArgumentParser()
    model_init.add_args(ap, cache=True, add_draft_model_args=True, default_chunk_size=chunk)
    argv = ["-m", str(model_dir), "-cs", str(math.ceil(cache_tokens / PAGE) * PAGE), "-chunk_size", str(chunk),
            "-ambs", os.environ.get("GLM53_MAX_BATCH", "1")]
    if cache_quant not in (None, 0, 16):
        argv += ["-cq", str(int(cache_quant))]
    if mtp:
        argv += ["--mtp", "-ndt", str(int(mtp))]
    if not embed_disk:
        argv += ["-ngr"]
    return ap.parse_args(argv)


def load(model_dir, store, chunk=2048, cache_tokens=16384, ram_gb=6.0, cpu=False, mtp=0, embed_disk=True,
         cache_quant=4, torch_limit_bytes=None):
    """Load the model and attach the engine. Returns (model, config, cache, tok, pools, tables, draft_model,
    draft_cache): pools = the VRAM expert cache's pools (expert_cache.Pool, one per GPU), tables = their NVMe tiers
    (nv2.Nv2: store, RAM tier, residency and pointer tables); draft_model / draft_cache are None unless mtp > 0."""
    model_dir, store = Path(model_dir), Path(store)
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"no EXL3 model at {model_dir} (config.json missing)")
    if chunk < 1 or cache_tokens < 1 or mtp < 0:
        raise ValueError(f"chunk {chunk}, cache_tokens {cache_tokens}, mtp {mtp}")
    man = read_store(store)
    check_store_matches(model_dir, man)
    os.environ.update(plan_env(store, man, ram_gb, cpu))

    if str(ENGINE) not in sys.path:
        sys.path.insert(0, str(ENGINE))
    if importlib.util.find_spec("exllamav3") is None:
        sys.path.insert(0, str(EXLLAMAV3))
    import torch
    if torch_limit_bytes:   # before anything is allocated on the device
        dev = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(dev).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, torch_limit_bytes / total), dev)

    import offload_stack
    from exllamav3 import model_init
    offload_stack.pre_load()
    args = model_args(model_dir, chunk, cache_tokens, mtp, embed_disk, cache_quant)
    model, config, cache, tok, draft_model, _draft_config, draft_cache = model_init.init(args, progress=False)
    pools = offload_stack.post_load(model)
    tables = [p.nv2 for p in pools]
    if os.environ.get("GLM53_MAIN_CPUS"):   # the GPU-feeding thread off the engine's pinned cores (as serve.py does)
        import nv2
        os.sched_setaffinity(0, set(nv2._cpus(os.environ["GLM53_MAIN_CPUS"])))
        torch.set_num_threads(int(os.environ.get("GLM53_TORCH_THREADS", "1")))
    return model, config, cache, tok, pools, tables, draft_model, draft_cache
