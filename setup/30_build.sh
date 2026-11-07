#!/bin/bash
# Compile exllamav3 and the offload kernels for the GPU in this machine (its compute capability, e.g. 8.6 for an
# RTX 3090, 8.9 for a 4090, 12.0 for a 5090). Re-run after changing GPUs or PyTorch, or when the setup check says the
# kernels are older than their sources.
source "$(dirname "$0")/lib.sh"
cd "$R" || exit 1
cuda_env
source .venv/bin/activate
CC=$(gpu_cc)
[ -n "$CC" ] || { echo "no GPU visible (nvidia-smi)"; exit 11; }
case "$CC" in
  [0-6].*|7.0|7.2) echo "compute capability $CC is too old: this engine needs Turing (7.5) or newer"; exit 13 ;;
  7.5) echo "note: Turing (7.5) runs the EXL3 kernels' sm_75 fallbacks; this combination is untested" ;;
esac
# one compile job per ~3 GiB of free RAM (nvcc on exllamav3's kernels takes 2-3 GiB each), at most one per CPU
# thread: with the model loaded, one per thread (32) ran the machine out of RAM. MAX_JOBS overrides it.
FREE_GIB=$(awk '/MemAvailable/ { printf "%d", $2 / 1048576 }' /proc/meminfo)
JOBS=$(( (FREE_GIB - 4) / 3 )); [ "$JOBS" -ge 1 ] || JOBS=1; [ "$JOBS" -le "$(nproc)" ] || JOBS=$(nproc)
MAX_JOBS=${MAX_JOBS:-$JOBS}
echo "compiling with $MAX_JOBS jobs ($FREE_GIB GiB RAM free, $(nproc) CPU threads)"
export TORCH_CUDA_ARCH_LIST="$CC" MAX_JOBS
SP=$(site_packages)
DEST="$SP/exllamav3_ext$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
TV=$(sed -n "s/^__version__ = '\(.*\)'/\1/p" "$SP/torch/version.py")
EXL3_SRC=$(bash "$SETUP_DIR/kernel_sources.sh" "$R" exllamav3)
OFFLOAD_SRC=$(bash "$SETUP_DIR/kernel_sources.sh" "$R" offload)
mkdir -p .repertory
# exllamav3 takes minutes: skip it when it was built for this GPU and this PyTorch from these sources (FULL_BUILD=1
# builds it anyway). The offload kernels are always built again.
if [ "${FULL_BUILD:-0}" != 1 ] && { ls "$DEST" || ls exllamav3/exllamav3_ext*.so; } >/dev/null 2>&1 \
   && [ "$(cat .repertory/build-arch 2>/dev/null)" = "$CC" ] && [ "$(cat .repertory/build-torch 2>/dev/null)" = "$TV" ] \
   && [ "$(cat .repertory/build-exl3 2>/dev/null)" = "$EXL3_SRC" ]; then
  echo "exllamav3 is already built for sm_${CC/./} and PyTorch $TV from these sources (FULL_BUILD=1 rebuilds it)"
  BUILT_EXL3=0
else
  # A copy of exllamav3's module installed by an earlier build is found before the new one; built for another
  # PyTorch, it fails to load. Remove it before anything imports exllamav3.
  rm -f "$DEST" .repertory/build-exl3
  echo "building exllamav3 for sm_${CC/./} (a few minutes)..."
  # its dependencies are pinned in setup/requirements.txt; build with this environment's tools, nothing downloaded
  pip install -q --no-deps --no-build-isolation -e exllamav3
  BUILT_EXL3=1
fi
echo "compiling offload extensions..."
rm -rf glm53-flash-offload/build .repertory/build-src
python engines/exl3_offload/build_ext.py
# exllamav3 JIT-builds its extension into torch's cache when the pip build above didn't leave one in exllamav3/, and
# then re-runs ninja's up-to-date check on every start (thousands of header stats; ~140 s on a cold WSL disk). Install
# that module, taken from this PyTorch's own cache folder only.
if [ "$BUILT_EXL3" = 1 ] && ! ls exllamav3/exllamav3_ext*.so >/dev/null 2>&1; then
  CACHE=$(python -c 'from torch.utils.cpp_extension import _get_build_directory as d; print(d("exllamav3_ext", False))' \
          2>/dev/null || true)
  if [ -n "$CACHE" ] && [ -f "$CACHE/exllamav3_ext.so" ]; then
    cp "$CACHE/exllamav3_ext.so" "$DEST"
    echo "installed prebuilt exllamav3_ext into $SP"
  fi
fi
python -c 'import torch, exllamav3.ext as e; print("exllamav3 extension:", e.exllamav3_ext.__file__)'
echo "$CC" > .repertory/build-arch
# the kernels match this PyTorch: a new one (setup/requirements.txt) means building again
echo "$TV" > .repertory/build-torch
# and these sources: the setup check marks the build outdated when they change (kernel_sources.sh)
echo "$EXL3_SRC" > .repertory/build-exl3
echo "$OFFLOAD_SRC" > .repertory/build-src
echo "built for sm_${CC/./}"
