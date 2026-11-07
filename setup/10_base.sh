#!/bin/bash
# Build tools and the CUDA toolkit (CUDA_VER in lib.sh). Under WSL the Windows NVIDIA driver provides the GPU, so only the toolkit
# is installed (NVIDIA's WSL repository, which has no driver packages). On native Linux the NVIDIA driver must
# already be installed: it needs a reboot and, with Secure Boot, key enrollment, so this step only checks it.
source "$(dirname "$0")/lib.sh"
need_root

if ! nvidia-smi >/dev/null 2>&1; then
  echo "No working NVIDIA driver (nvidia-smi fails)."
  driver_help
  exit 11
fi
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader)"
drv=$(driver_cuda)
if [ -n "$drv" ] && ! version_ge "$drv" "$CUDA_VER"; then
  echo "This driver supports CUDA $drv; the engine is built with CUDA $CUDA_VER."
  driver_help
  exit 11
fi

. /etc/os-release
if ! command -v apt-get >/dev/null; then
  echo "This step automates Ubuntu and Debian ($PRETTY_NAME detected). Install these yourself, then continue:"
  echo "  build tools (gcc/g++, make), git, python3 + venv + headers, ninja, curl; CUDA toolkit $CUDA_VER"
  exit 12
fi
echo "installing build tools..."
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq build-essential git python3-venv python3-dev ninja-build wget curl bc >/dev/null

if [ -x "$CUDA_HOME_DEFAULT/bin/nvcc" ]; then
  echo "CUDA $CUDA_VER already installed"
else
  if [ "$IN_WSL" = 1 ]; then repo=wsl-ubuntu
  elif [ "$ID" = ubuntu ]; then repo=ubuntu${VERSION_ID//./}
  elif [ "$ID" = debian ]; then repo=debian${VERSION_ID%%.*}
  else echo "no CUDA repository mapping for $PRETTY_NAME; install CUDA $CUDA_VER manually"; exit 12; fi
  url=https://developer.download.nvidia.com/compute/cuda/repos/$repo/x86_64
  # NVIDIA builds each CUDA release for some distribution releases only (CUDA 13: not Ubuntu 20.04 or Debian 11)
  if ! listing=$(wget -q -O - "$url/"); then
    echo "can't reach NVIDIA's CUDA repository for $PRETTY_NAME ($url)"; exit 12
  fi
  if ! grep -q "cuda-toolkit-${CUDA_VER/./-}_" <<<"$listing"; then
    echo "NVIDIA has no CUDA $CUDA_VER for $PRETTY_NAME. Use Ubuntu 22.04 or newer, or Debian 12 or newer."
    exit 12
  fi
  # NVIDIA's repository key package, checked against its known hash before it is trusted (it adds a signing key)
  case "$repo" in
    wsl-ubuntu) keyring_sha=eea6cc5f0eaeb99082d054b8c05ae206a378e31e88048df0310d59f651dceed2 ;;
    ubuntu2204) keyring_sha=d93190d50b98ad4699ff40f4f7af50f16a76dac3bb8da1eaaf366d47898ff8df ;;
    ubuntu2404) keyring_sha=d2a6b11c096396d868758b86dab1823b25e14d70333f1dfa74da5ddaf6a06dba ;;
    ubuntu2604) keyring_sha=f7f474b5f6a4adf987aa587920df00e713285958ef6a913dda1945a544a3099e ;;
    debian12)   keyring_sha=e7f219eab6fe4819cdb5c15b98233dc3420302d9c00883219cd3d896857cf48d ;;
    debian13)   keyring_sha=d0d4ef986a44400f9db33c600ef33a985175e7cc63d805a10e1839c7a1e78f5f ;;
    *) echo "no known hash for NVIDIA's $repo keyring; install CUDA $CUDA_VER manually"; exit 12 ;;
  esac
  echo "installing CUDA $CUDA_VER toolkit from NVIDIA's $repo repository (several GB)..."
  tmp=$(mktemp -d)
  wget -q -O "$tmp/keyring.deb" "$url/cuda-keyring_1.1-1_all.deb"
  echo "$keyring_sha  $tmp/keyring.deb" | sha256sum -c --quiet - || { echo "NVIDIA's keyring package failed its hash check"; exit 12; }
  $SUDO dpkg -i "$tmp/keyring.deb" >/dev/null
  rm -rf "$tmp"
  $SUDO apt-get update -qq
  # the toolkit only: never pull a driver package (WSL must not have one; native keeps the one you installed)
  $SUDO apt-get install -y -qq "cuda-toolkit-${CUDA_VER/./-}" >/dev/null
fi

extra=""
[ "$IN_WSL" = 1 ] && extra=":/usr/lib/wsl/lib"
$SUDO tee /etc/profile.d/cuda.sh >/dev/null <<EOF
export CUDA_HOME=$CUDA_HOME_DEFAULT
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64$extra:\${LD_LIBRARY_PATH:-}
EOF
cuda_env
nvcc --version | tail -1
python3 --version
