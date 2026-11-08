#!/bin/bash
# Download one model and pack its direct-I/O expert store. Resumable: re-run to continue.
#   50_model.sh ENGINE_DIR QUANT [--dir DIR --store STORE --branch BRANCH --repo REPO --revision SHA] [OLD_ENGINE_DIR]
# Setup passes the options from the catalog (catalog/catalog.json: the model's paths and source pin). Called with only
# QUANT, as before the catalog drove setup, 3.05 and 4.05 pick Qwen's (the same values, hard-coded below).
# OLD_ENGINE_DIR: QwenFlash's checkout. A model it downloaded and packed there is linked, not fetched again.
source "$(dirname "$0")/lib.sh"
Q=${2:?quant (e.g. 3.05)}
shift 2
D="" S="" B="" REPO="" SHA="" OLD=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dir) D=${2:?--dir needs a value}; shift 2 ;;
    --store) S=${2:?--store needs a value}; shift 2 ;;
    --branch) B=${2:?--branch needs a value}; shift 2 ;;
    --repo) REPO=${2:?--repo needs a value}; shift 2 ;;
    --revision) SHA=${2:?--revision needs a value}; shift 2 ;;
    -*) echo "unknown option $1"; exit 2 ;;
    *) OLD=$1; shift ;;
  esac
done
if [ -z "$D" ]; then
  case "$Q" in
    3.05) D=model;      B=3.05bpw_h5_ng5; S=store/qwen.bin ;;
    4.05) D=model-4.05; B=4.05bpw_h6_ng6; S=store/qwen405.bin ;;
    *) echo "unknown quant $Q (pass --dir, --store and --branch or --revision)"; exit 2 ;;
  esac
fi
[ -n "$S" ] || { echo "--store is missing"; exit 2; }
for p in "$D" "$S"; do
  case "/$p/" in //*|*/../*|*[!A-Za-z0-9._/-]*) echo "refusing a path outside the engine checkout: $p"; exit 2 ;; esac
done
SD=$(dirname "$S")
N=$(basename "$S" .bin)
cd "$R" || exit 1
if [ -n "$OLD" ] && [ ! -e "$D" ] && [ -f "$OLD/$D/config.json" ] && [ -f "$OLD/$SD/$N.bin" ] \
    && [ -f "$OLD/$SD/$N.json" ]; then
  mkdir -p "$SD"
  ln -s "$OLD/$D" "$D" && ln -s "$OLD/$SD/$N.bin" "$SD/$N.bin" && ln -s "$OLD/$SD/$N.json" "$SD/$N.json"
  echo "using the $Q model and expert store QwenFlash downloaded in $OLD (linked, not downloaded again)"
  ls -lh "$SD/$N.bin"
  exit 0
fi
source .venv/bin/activate
echo "free space: $(df -h --output=avail . | tail -1 | tr -d ' ')"
fs=$(stat -f -c %T .)
case "$fs" in
  ext2/ext3|xfs|btrfs) ;;
  *) echo "warning: the expert store needs O_DIRECT reads; '$fs' may not support them" ;;
esac
get=(--model-dir "./$D")
[ -n "$B" ] && get+=(--branch "$B")
[ -n "$SHA" ] && get+=(--sha "$SHA")
[ -n "$REPO" ] && get+=(--repo "$REPO")
echo "downloading ${B:-$SHA} into $D (resumable)..."
python engines/exl3_offload/download.py "${get[@]}" 2>&1 \
  | awk '/^PROGRESS/{n++; if (n%5) next} {print; fflush()}'
if [ -f "$SD/$N.bin" ] && [ -f "$SD/$N.json" ]; then
  echo "expert store $SD/$N.bin already packed"
else
  mkdir -p "$SD"
  echo "packing expert store (about a minute)..."
  python glm53-flash-offload/scripts/pack_glm53_store.py pack --model "./$D" --out-dir "./$SD" --name "$N"
fi
ls -lh "$SD/$N.bin"
