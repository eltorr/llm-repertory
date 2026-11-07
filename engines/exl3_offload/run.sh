#!/bin/bash
# Starts the exl3_offload engine's model server on any NVIDIA GPU (native Linux or WSL2), sized from the hardware
# unless overridden. Paths are relative to the repo root, whatever the current folder.
#
#   engines/exl3_offload/run.sh server    OpenAI-compatible API on $HOST:$PORT (default 127.0.0.1:30000),
#                                         context window $CONTEXT tokens (default 262144)
#
# Overrides (environment): QUANT (3.05 | 4.05), RAM_GB, MODEL_DIR, STORE, HOST, PORT,
#   QWEN_VRAM_LIMIT_BYTES  device-wide VRAM the expert cache may grow into (includes other apps' VRAM)
#   QWEN_TORCH_LIMIT_BYTES this process's torch allocator cap
#   QWEN_VRAM_GUARD_BYTES  device-wide usage at which the server exits (76) instead of running out mid-request
#   MEM_HIGH, MEM_MAX, QWEN_RAM_LIMIT_BYTES   RAM limits of the server's memory cgroup / RAM guard
# The Repertory manager computes all of these from what is free right now; the defaults below suit a manual run.
set -euo pipefail
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/../.."
[ -f /etc/profile.d/cuda.sh ] && source /etc/profile.d/cuda.sh
[ -d /usr/lib/wsl/lib ] && case ":$PATH:" in *:/usr/lib/wsl/lib:*) ;; *) PATH="$PATH:/usr/lib/wsl/lib" ;; esac
GB=1000000000
GIB=1073741824
calc() { awk "BEGIN { printf \"%d\", $1 }"; }

QUANT="${QUANT:-3.05}"
case "$QUANT" in
  # every expert: 48 layers x 512 experts x (1.843 MB | 2.457 MB)
  3.05) : "${MODEL_DIR:=./model}"      "${STORE:=./store/qwen.bin}";    EXPERT_BYTES=45290000000 ;;
  4.05) : "${MODEL_DIR:=./model-4.05}" "${STORE:=./store/qwen405.bin}"; EXPERT_BYTES=60380000000 ;;
  *) echo "unknown QUANT=$QUANT (use 3.05 or 4.05)"; exit 2 ;;
esac
[ -d "$MODEL_DIR" ] || { echo "model not found at $MODEL_DIR (download it: setup/50_model.sh . $QUANT)"; exit 2; }
[ -f "$STORE" ]     || { echo "expert store not found at $STORE (setup/50_model.sh . $QUANT packs it)"; exit 2; }

# --- GPU: sizes from the card that is actually here ---------------------------------------------------------------
# The offload engine is single-GPU (docs/research/multi-gpu.md): with several cards visible, pin it to one.
GPU="${QWEN_GPU:-0}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$(nvidia-smi -L | wc -l)" -gt 1 ]; then export CUDA_VISIBLE_DEVICES="$GPU"; fi
read -r VRAM_TOTAL_MIB VRAM_USED_MIB CC DISP < <(nvidia-smi -i "$GPU" \
  --query-gpu=memory.total,memory.used,compute_cap,display_active --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
VRAM_TOTAL=$(calc "$VRAM_TOTAL_MIB * 1048576"); VRAM_USED=$(calc "$VRAM_USED_MIB * 1048576")
# keep ~10% of the card (1.2-3.2 GB) free for the desktop and other apps; 0.6 GB minimum when no display is attached
# (sizing.headroom_bytes)
# (nvidia-smi says Disabled under WSL even for the desktop's card)
MIN_HEADROOM=1.2; [ "${DISP:-}" = Disabled ] && [ ! -d /usr/lib/wsl ] && MIN_HEADROOM=0.6
HEADROOM=$(calc "($VRAM_TOTAL * 0.10 < $MIN_HEADROOM * $GB) ? $MIN_HEADROOM * $GB : (($VRAM_TOTAL * 0.10 > 3.2 * $GB) ? 3.2 * $GB : $VRAM_TOTAL * 0.10)")
: "${QWEN_VRAM_LIMIT_BYTES:=$(calc "$VRAM_TOTAL - $HEADROOM")}"
: "${QWEN_TORCH_LIMIT_BYTES:=$(calc "$QWEN_VRAM_LIMIT_BYTES - $VRAM_USED + 0.5 * $GB")}"
: "${QWEN_VRAM_GUARD_BYTES:=$VRAM_TOTAL}"
export QWEN_VRAM_LIMIT_BYTES QWEN_TORCH_LIMIT_BYTES QWEN_VRAM_GUARD_BYTES
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$CC}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- plan: chunk size, staging and cache from sizing.py (the manager passes its own plan in QWEN_CHUNK & co.) -----
CONTEXT="${CONTEXT:-262144}"
QWEN_RAM_ALL_GIB=""
if [ -z "${QWEN_CHUNK:-}" ]; then
  BUDGET_GB=$(awk "BEGIN { printf \"%.2f\", ($QWEN_VRAM_LIMIT_BYTES - $VRAM_USED) / $GB }")
  PLAN=$(python3 engines/exl3_offload/sizing.py env --quant "$QUANT" --budget-gb "$BUDGET_GB" --context "$CONTEXT") || {
    eval "$PLAN"; echo "[run.sh] lower CONTEXT (now $CONTEXT) or free VRAM"; exit 3; }
  eval "$PLAN"
fi
export QWEN_CHUNK

# --- RAM: hold every expert that won't fit in VRAM, within what the host can spare --------------------------------
if [ -z "${RAM_GB:-}" ]; then
  AVAIL=$(awk '/MemAvailable/ { print $2 * 1024 }' /proc/meminfo)
  NEED=${QWEN_RAM_ALL_GIB:-$(calc "$EXPERT_BYTES / $GIB")}
  SPARE=$(calc "($AVAIL - 14 * $GB) / $GIB")          # 8 GB for the host + ~6.5 GB server overhead
  RAM_GB=$(( NEED < SPARE ? NEED : SPARE )); [ "$RAM_GB" -ge 4 ] || RAM_GB=4
fi
: "${MEM_HIGH:=$(( RAM_GB + 16 ))G}"
: "${MEM_MAX:=$(( RAM_GB + 19 ))G}"
: "${QWEN_RAM_LIMIT_BYTES:=$(calc "(${MEM_MAX%G} * $GIB) - 1.5 * $GB")}"
export QUANT MODEL_DIR STORE RAM_GB MEM_HIGH MEM_MAX QWEN_RAM_LIMIT_BYTES

# --- memory cgroup: gives the RAM guard something to measure (WSL's root cgroup has no memory.current) ------------
if [ -z "${QWEN_IN_SCOPE:-}" ] && command -v systemd-run >/dev/null 2>&1; then
  if [ "$(id -u)" = 0 ]; then SR=(systemd-run --quiet --scope); else SR=(systemd-run --quiet --user --scope); fi
  if "${SR[@]}" --unit="repertory-probe-$$" true >/dev/null 2>&1; then
    exec "${SR[@]}" --unit="repertory-$$" -p MemoryHigh="$MEM_HIGH" -p MemoryMax="$MEM_MAX" -p MemorySwapMax=0 \
         env QWEN_IN_SCOPE=1 bash "$SELF" "$@"
  fi
  echo "[run.sh] no systemd scope available: RAM limits not enforced"
fi
CG="/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1)"
[ -f "$CG/memory.current" ] && export QWEN_MEMORY_CGROUP="$CG"

[ -f .venv/bin/activate ] || { echo "no .venv here: run setup/20_env.sh first"; exit 2; }
source .venv/bin/activate

MODE="${1:-}"; shift || true
echo "[run.sh] quant=$QUANT mode=$MODE gpu=${VRAM_TOTAL_MIB}MiB sm_${CC/./} vram_limit=$(calc "$QWEN_VRAM_LIMIT_BYTES / $GB").$(calc "($QWEN_VRAM_LIMIT_BYTES % $GB) / 100000000")GB ram_gb=$RAM_GB cgroup=${QWEN_MEMORY_CGROUP:-none} (high $MEM_HIGH, max $MEM_MAX)"

# WSL's virtual disk serves the page-by-page reads of mapping big shared libraries very slowly (cold `import torch`
# ~150 s vs 3 s warm). Reading them sequentially first takes ~2 s. Skipped on native Linux, which doesn't need it.
if grep -qi microsoft /proc/version 2>/dev/null; then
  t0=$(date +%s%N)
  find .venv/lib exllamav3 glm53-flash-offload -type f \( -name '*.so' -o -name '*.so.*' \) -print0 2>/dev/null \
    | xargs -0 -r -P8 -n4 cat > /dev/null
  find .venv/lib exllamav3 glm53-flash-offload engines -type f \( -name '*.py' -o -name '*.pyc' \) -print0 2>/dev/null \
    | xargs -0 -r -P16 -n200 cat > /dev/null
  echo "[run.sh] pre-read engine libraries in $(( ($(date +%s%N) - t0) / 1000000 )) ms"
fi

case "$MODE" in
  server) exec python engines/exl3_offload/server.py --model "$MODEL_DIR" --store "$STORE" --ram-gb "$RAM_GB" \
                 --cache "$CONTEXT" --chunk "$QWEN_CHUNK" --host "${HOST:-127.0.0.1}" --port "${PORT:-30000}" "$@" ;;
  *) echo "usage: [QUANT=3.05|4.05] $0 server [args]"; exit 2 ;;
esac
