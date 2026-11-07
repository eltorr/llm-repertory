"""Builds the offload engine's native extensions for this GPU and this PyTorch (setup/30_build.sh runs it).

Each extension would build itself the first time the engine imports it. Building them here, once, keeps that out of
the model's first start, and shows a compiler error during setup rather than at load. TORCH_CUDA_ARCH_LIST picks
the GPU architecture (30_build.sh sets this GPU's); MAX_JOBS, the parallel compile jobs.
"""
import importlib
import sys
import time
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[2] / "glm53-flash-offload" / "glm53"
# (module, the function that builds and loads its extension): the NVMe tier, the VRAM expert cache, the nv2 engine
EXTENSIONS = (("nv_tier", "_ext"), ("expert_cache", "_ext"), ("nv2", "ext"))


def main():
    sys.path.insert(0, str(ENGINE))
    import torch
    if not torch.cuda.is_available():
        print("no CUDA GPU visible: nothing to build for", flush=True)
        return 1
    major, minor = torch.cuda.get_device_capability()
    print(f"PyTorch {torch.__version__} (CUDA {torch.version.cuda}) on {torch.cuda.get_device_name()}, "
          f"sm_{major}{minor}", flush=True)
    for module, build in EXTENSIONS:
        t0 = time.monotonic()
        lib = getattr(importlib.import_module(module), build)()
        print(f"built {module}: {lib.__file__} in {time.monotonic() - t0:.0f} s", flush=True)
    print("offload extensions ready", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
