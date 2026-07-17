#!/usr/bin/env bash
# Run on a Palmetto LOGIN / DTN node (compute nodes are air-gapped).
# Downloads base Monet-7B weights and pre-caches the V* + HR-Bench 4k datasets
# into the shared HF cache under /scratch/haizhow.
set -euo pipefail

export HF_HOME=/scratch/haizhow/cache/huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME"

WEIGHTS_DIR=/scratch/haizhow/weights/Monet-7B

echo "== downloading NOVAglow646/Monet-7B -> $WEIGHTS_DIR =="
huggingface-cli download NOVAglow646/Monet-7B \
  --local-dir "$WEIGHTS_DIR" --local-dir-use-symlinks False

echo "== pre-caching eval datasets (login node has network) =="
python - <<'PY'
from datasets import load_dataset
print("V* ..."); load_dataset("lmms-lab/vstar-bench", split="test")
print("HR-Bench 4k ..."); load_dataset("DreamMr/HR-Bench", "hrbench_4k", split="test")
print("done")
PY

echo "Weights at: $WEIGHTS_DIR"
echo "HF cache at: $HF_HOME"
echo "Now inspect schemas on the login node before a full run:"
echo "  python -m evaluation.capimagine.datasets --inspect vstar"
echo "  python -m evaluation.capimagine.datasets --inspect hrbench_4k"
