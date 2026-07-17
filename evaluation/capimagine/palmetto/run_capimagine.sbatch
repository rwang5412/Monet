#!/usr/bin/env bash
#SBATCH --job-name=capimagine
#SBATCH --output=/scratch/haizhow/logs/capimagine_%j.out
#SBATCH --error=/scratch/haizhow/logs/capimagine_%j.err
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

MODEL="${MODEL:-/scratch/haizhow/weights/Monet-7B}"
DATASET="${DATASET:-vstar}"
OUT_DIR="${OUT_DIR:-/scratch/haizhow/results/capimagine/${DATASET}}"
LIMIT_FLAG=""; [ -n "${LIMIT:-}" ] && LIMIT_FLAG="--limit ${LIMIT}"

mkdir -p /scratch/haizhow/logs "$OUT_DIR"

source ~/.bashrc || true
conda activate monet

# Repo root must be on PYTHONPATH so `inference.*` / `evaluation.*` import.
cd "${REPO_DIR:-$SLURM_SUBMIT_DIR}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Monet latent config + offline (compute node is air-gapped after login-node cache).
export LATENT_SIZE=10
export HF_HOME=/scratch/haizhow/cache/huggingface
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
