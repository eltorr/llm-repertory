# shellcheck shell=bash disable=SC2034  # sourced by the setup steps, which use its variables
# Shared by the setup steps. They run the same way inside WSL2 (as root, started by the Windows manager) and on
# native Linux (as your user; steps that install packages use sudo).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
SETUP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=${1:-$(cd "$SETUP_DIR/.." && pwd)}           # engine checkout
IN_WSL=0
grep -qi microsoft /proc/version 2>/dev/null && IN_WSL=1
SUDO=""
if [ "$(id -u)" != 0 ]; then
  if sudo -n true 2>/dev/null; then SUDO="sudo"; else SUDO="NEEDS_PASSWORD"; fi
fi
CUDA_VER=13.2        # the CUDA toolkit the kernels are built with; matches PyTorch's build (setup/requirements.in)
DRIVER_MIN=595       # the NVIDIA driver branch for CUDA_VER (the table in NVIDIA's CUDA release notes)
CUDA_HOME_DEFAULT=/usr/local/cuda-$CUDA_VER

need_root() {
  if [ "$SUDO" = NEEDS_PASSWORD ]; then
    echo "This step installs system packages and needs sudo. Run it in a terminal:"
    echo "  sudo bash $SETUP_DIR/$(basename "$0") $R"
    exit 10
  fi
}

cuda_env() {
  if [ -f /etc/profile.d/cuda.sh ]; then source /etc/profile.d/cuda.sh; fi
  # the toolkit this engine builds with (CUDA_VER), even while cuda.sh still names an older one
  if [ -x "$CUDA_HOME_DEFAULT/bin/nvcc" ]; then
    export CUDA_HOME=$CUDA_HOME_DEFAULT PATH=$CUDA_HOME_DEFAULT/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME_DEFAULT/lib64:${LD_LIBRARY_PATH:-}
  fi
}

gpu_cc() { nvidia-smi -i "${QWEN_GPU:-0}" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' '; }

# the newest CUDA the installed driver supports, from nvidia-smi's header ("CUDA Version: 13.2", or in newer
# drivers "CUDA UMD Version: 13.4")
driver_cuda() {
  nvidia-smi 2>/dev/null | sed -n 's/.*CUDA[A-Za-z ]* Version: *\([0-9][0-9.]*\).*/\1/p' | head -1 || true
}

# version_ge A B: A >= B
version_ge() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" = "$2" ]; }

driver_help() {
  if [ "$IN_WSL" = 1 ]; then
    echo "Install NVIDIA driver $DRIVER_MIN or newer on Windows (https://www.nvidia.com/drivers), then run: wsl --shutdown"
  else
    echo "Install NVIDIA driver $DRIVER_MIN or newer, then reboot. On Ubuntu:"
    echo "  sudo apt install nvidia-driver-$DRIVER_MIN-open     (or a newer nvidia-driver-*-open)"
    echo "Other distributions: https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/"
  fi
}

site_packages() { ls -d "$R"/.venv/lib/python3*/site-packages 2>/dev/null | head -1; }
