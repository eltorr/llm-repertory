#!/bin/bash
# Engine code and its Python environment.
#   20_env.sh ENGINE_DIR [SOURCE_REPO]
# With SOURCE_REPO (the Windows manager passes its own checkout), ENGINE_DIR becomes a clone of it that is updated
# to the source's current commit. Uncommitted changes in ENGINE_DIR are stashed first, never lost.
# Without it (native Linux), ENGINE_DIR is the checkout itself.
source "$(dirname "$0")/lib.sh"
SRC=${2:-}
BRANCH=main             # the branch of the Windows checkout that the WSL clone follows

if [ -n "$SRC" ] && [ "$(readlink -f "$SRC")" != "$(readlink -f "$R")" ]; then
  # git refuses repositories owned by another user (the Windows checkout, seen through /mnt) unless trusted
  git config --global --get-all safe.directory | grep -qxF "$SRC" || git config --global --add safe.directory "$SRC"
  want=$(git -C "$SRC" rev-parse "$BRANCH")
  if [ ! -d "$R/.git" ]; then
    echo "cloning the engine from $SRC into $R..."
    git clone -q --branch "$BRANCH" "$SRC" "$R"
  else
    cd "$R" || exit 1
    if [ "$(git rev-parse HEAD)" = "$want" ]; then
      echo "engine code is current (${want:0:7})"
    else
      if [ -n "$(git status --porcelain --untracked-files=normal)" ]; then
        git stash push -u -q -m "repertory setup: local changes before sync $(date '+%F %T')"
        echo "stashed local changes in $R (git stash list)"
      fi
      git remote get-url repertory >/dev/null 2>&1 || git remote add repertory "$SRC"
      git remote set-url repertory "$SRC"
      git fetch -q repertory "$BRANCH"
      git checkout -q -B "$BRANCH" FETCH_HEAD
      echo "engine code updated to $(git log --oneline -1)"
    fi
  fi
fi

cd "$R" || exit 1
cuda_env
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
# file checks only: importing torch from a cold WSL page cache takes minutes
have_env() {
  grep -q "+cu${CUDA_VER/./}" "$1/torch/version.py" 2>/dev/null || return 1   # PyTorch built for this CUDA
  for m in fastapi uvicorn safetensors jinja2 numpy ninja; do [ -e "$1/$m" ] || return 1; done
}
SP=$(site_packages)
ready=0
[ -n "$SP" ] && have_env "$SP" && ready=1
# Exactly the files in setup/requirements.txt (pip checks each one's hash), so every install gets the tested set and a
# tampered or swapped download is refused. PyTorch's CUDA build comes from its own index, the rest from PyPI.
echo "installing the Python packages pinned in setup/requirements.txt..."
pip install -q --require-hashes -r setup/requirements.txt \
  --index-url "https://download.pytorch.org/whl/cu${CUDA_VER/./}" --extra-index-url https://pypi.org/simple
# A PyTorch built for another CUDA leaves its CUDA packages behind (pip keeps them: their names differ). Remove them, so
# nothing can depend on a runtime that a fresh install doesn't have. Some share folders with this CUDA's packages
# (nvidia/cudnn, nccl, ...), so removing them deletes files those need: reinstall this CUDA's packages afterwards.
stale=$(pip list --format=freeze 2>/dev/null | sed -n 's/^\(nvidia-[a-z0-9-]*-cu[0-9][0-9]*\)==.*/\1/p' |
        grep -v -- "-cu${CUDA_VER%%.*}$" || true)
if [ -n "$stale" ]; then
  echo "removing CUDA packages left by an older PyTorch: $(echo "$stale" | tr '\n' ' ')"
  # shellcheck disable=SC2086 # one package name per word
  pip uninstall -q -y $stale
  nv=$(mktemp)
  awk '/^[^ #]/{keep = /^nvidia-/} keep' setup/requirements.txt > "$nv"
  echo "reinstalling this CUDA's packages, whose shared files went with them..."
  pip install -q --force-reinstall --no-deps --require-hashes -r "$nv" \
    --index-url "https://download.pytorch.org/whl/cu${CUDA_VER/./}" --extra-index-url https://pypi.org/simple
  rm -f "$nv"
  ready=0                                         # so the check below imports torch
fi
if [ "$ready" = 1 ]; then
  echo "Python environment ready: torch $(sed -n "s/^__version__ = '\(.*\)'/\1/p" "$SP/torch/version.py")"
  exit 0
fi
python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU", torch.cuda.get_device_name(0),
      "| sm_%d%d" % torch.cuda.get_device_capability(0))
x = torch.randn(2048, 2048, device="cuda"); (x @ x).sum().item(); print("GPU matmul OK")
PY
