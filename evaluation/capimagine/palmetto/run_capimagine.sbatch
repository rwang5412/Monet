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
#
# NOTE: no `-u` (nounset). Sourcing /etc/bashrc + `conda activate` reference
# unbound vars (e.g. BASHRCSOURCED), which under set -u aborts the whole job
# before anything runs. All our own vars use ${VAR:-default}, so -u adds nothing.
set -eo pipefail

# MODEL_TAG keeps each model's clean baseline + mu/sigma stats in a SEPARATE
# OUT_DIR. Do NOT run two models into the same OUT_DIR: the corrupt passes read
# <dataset>_clean.json / <dataset>_stats.pt from OUT_DIR, so a shared dir makes
# the second model silently overwrite the first and Delta comes out garbage.
MODEL="${MODEL:-/scratch/$USER/monet_weights/Monet-7B}"
MODEL_TAG="${MODEL_TAG:-model}"
DATASET="${DATASET:-vstar}"
OUT_DIR="${OUT_DIR:-/scratch/$USER/results/capimagine/${MODEL_TAG}/${DATASET}}"
# V* has high-res samples that overflow 8192 (one hits ~8250, worst case ~16200),
# and vLLM fails the whole batch on a single oversized prompt -- keep this above it.
PROMPT_STYLE="${PROMPT_STYLE:-monet}"   # paper_eval = the paper's VLMEvalKit prompt
MAX_MODEL_LEN="${MAX_MODEL_LEN:-18432}"
# Greedy (temp=0) gives the cleanest do(Z) Delta. If the clean pass emits too few
# latents, resubmit with TEMPERATURE=0.1 (near-deterministic, but lets latents fire).
TEMPERATURE="${TEMPERATURE:-0}"
LIMIT_FLAG=""; [ -n "${LIMIT:-}" ] && LIMIT_FLAG="--limit ${LIMIT}"

mkdir -p "/scratch/$USER/logs" "$OUT_DIR"

source ~/.bashrc || true
conda activate monet

# Repo root must be on PYTHONPATH so `inference.*` / `evaluation.*` import.
cd "${REPO_DIR:-$SLURM_SUBMIT_DIR}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Monet latent config + offline (compute node is air-gapped after login-node cache).
# LATENT_SIZE is the test-time latent count. Default 10 (do(Z) runs). For an
# accuracy repro, override to 8 (the SFT-trained size; paper sweeps {8,10,12,16}).
# The value is both shell-exported (so it reaches the vLLM worker) AND passed as
# --latent-size (harness uses setdefault, so the shell export wins consistently).
export LATENT_SIZE="${LATENT_SIZE:-10}"
# Must match the HF_HOME used when the datasets were pre-cached on the DTN,
# or the offline job below can't find them.
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "MODEL=$MODEL DATASET=$DATASET OUT_DIR=$OUT_DIR"

for MODE in clean corrupt_gauss corrupt_mean; do
  echo "======== mode=$MODE ========"
  # SHELL-export the hook config so it reaches the vLLM WORKER. The Monet runner
  # patch loads via sitecustomize at Python startup, so vLLM snapshots the worker
  # env BEFORE the harness sets os.environ["MONET_LATENT_*"] at runtime -- those
  # never reach the worker (capture writes no stats, corruptions no-op -> Delta=0).
  # LATENT_SIZE works only because it is shell-exported; do the same here.
  if [ "$MODE" = "clean" ]; then export MONET_LATENT_MODE=capture; else export MONET_LATENT_MODE="$MODE"; fi
  export MONET_LATENT_STATS="$OUT_DIR/${DATASET}_stats.pt"
  export MONET_LATENT_SEED=0
  python -m evaluation.capimagine.harness \
    --model "$MODEL" --dataset "$DATASET" --mode "$MODE" \
    --out-dir "$OUT_DIR" --tp 1 --prompt-style "$PROMPT_STYLE" \
    --latent-size "$LATENT_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" --temperature "$TEMPERATURE" $LIMIT_FLAG
done

echo "Results in $OUT_DIR"
