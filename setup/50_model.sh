#!/bin/bash
# Download one quant and pack its direct-I/O expert store. Resumable: re-run to continue.
#   50_model.sh ENGINE_DIR 3.05|4.05 [OLD_ENGINE_DIR]
# OLD_ENGINE_DIR: QwenFlash's checkout. A quant it downloaded and packed there is linked, not fetched again.
source "$(dirname "$0")/lib.sh"
Q=${2:?quant (3.05 or 4.05)}
case "$Q" in
  3.05) D=model;      B=3.05bpw_h5_ng5; N=qwen ;;
  4.05) D=model-4.05; B=4.05bpw_h6_ng6; N=qwen405 ;;
  *) echo "unknown quant $Q"; exit 2 ;;
esac
cd "$R" || exit 1
OLD=${3:-}
if [ -n "$OLD" ] && [ ! -e "$D" ] && [ -f "$OLD/$D/config.json" ] && [ -f "$OLD/store/$N.bin" ]    && [ -f "$OLD/store/$N.json" ]; then
  mkdir -p store
  ln -s "$OLD/$D" "$D" && ln -s "$OLD/store/$N.bin" "store/$N.bin" && ln -s "$OLD/store/$N.json" "store/$N.json"
  echo "using the $Q model and expert store QwenFlash downloaded in $OLD (linked, not downloaded again)"
  ls -lh "store/$N.bin"
  exit 0
fi
source .venv/bin/activate
echo "free space: $(df -h --output=avail . | tail -1 | tr -d ' ')"
fs=$(stat -f -c %T .)
case "$fs" in
  ext2/ext3|xfs|btrfs) ;;
  *) echo "warning: the expert store needs O_DIRECT reads; '$fs' may not support them" ;;
esac
echo "downloading $B into $D (resumable)..."
python engines/exl3_offload/download.py --model-dir "./$D" --branch "$B" 2>&1 \
  | awk '/^PROGRESS/{n++; if (n%5) next} {print; fflush()}'
if [ -f "store/$N.bin" ] && [ -f "store/$N.json" ]; then
  echo "expert store store/$N.bin already packed"
else
  mkdir -p store
  echo "packing expert store (about a minute)..."
  python glm53-flash-offload/scripts/pack_glm53_store.py pack --model "./$D" --out-dir ./store --name "$N"
fi
ls -lh "store/$N.bin"
