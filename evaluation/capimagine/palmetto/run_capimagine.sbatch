#!/usr/bin/env bash
#SBATCH --job-name=capimagine
#SBATCH --output=/scratch/%u/logs/capimagine_%j.out
#SBATCH --error=/scratch/%u/logs/capimagine_%j.err
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#
# CapImagine do(Z) accuracy harness for base Monet-7B.
# Runs the clean (capture) pass, then both corruption passes; the corruption
# passes print Delta accuracy vs clean and the text-diff guard.
#
# Override via --export, e.g.:
#   sbatch --export=ALL,DATASET=hrbench_4k evaluation/capimagine/palmetto/run_capimagine.sbatch
set -euo pipefail

# MODEL_TAG keeps each model's clean baseline + mu/sigma stats in a SEPARATE
# OUT_DIR. Do NOT run two models into the same OUT_DIR: the corrupt passes read
# <dataset>_clean.json / <dataset>_stats.pt from OUT_DIR, so a shared dir makes
# the second model silently overwrite the first and Delta comes out garbage.
MODEL="${MODEL:-/scratch/$USER/monet_weights/Monet-7B}"
MODEL_TAG="${MODEL_TAG:-model}"
DATASET="${DATASET:-vstar}"
OUT_DIR="${OUT_DIR:-/scratch/$USER/results/capimagine/${MODEL_TAG}/${DATASET}}"
LIMIT_FLAG=""; [ -n "${LIMIT:-}" ] && LIMIT_FLAG="--limit ${LIMIT}"

mkdir -p "/scratch/$USER/logs" "$OUT_DIR"

source ~/.bashrc || true
conda activate monet

# Repo root must be on PYTHONPATH so `inference.*` / `evaluation.*` import.
cd "${REPO_DIR:-$SLURM_SUBMIT_DIR}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Monet latent config + offline (compute node is air-gapped after login-node cache).
export LATENT_SIZE=10
# Must match the HF_HOME used when the datasets were pre-cached on the DTN,
# or the offline job below can't find them.
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "MODEL=$MODEL DATASET=$DATASET OUT_DIR=$OUT_DIR"

for MODE in clean corrupt_gauss corrupt_mean; do
  echo "======== mode=$MODE ========"
  python -m evaluation.capimagine.harness \
    --model "$MODEL" --dataset "$DATASET" --mode "$MODE" \
    --out-dir "$OUT_DIR" --tp 1 $LIMIT_FLAG
done

echo "Results in $OUT_DIR"
