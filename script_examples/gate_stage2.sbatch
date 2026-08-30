#!/usr/bin/env bash
#SBATCH --job-name=gate_s2
#SBATCH --output=/scratch/%u/logs/gate_s2_%j.out
#SBATCH --error=/scratch/%u/logs/gate_s2_%j.err
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#
# Stage-2 promotion gate + collapse probe on a trained checkpoint.
# Inference-only (one model on GPU at a time, bf16) -- a 40GB A100 is enough,
# and the a100:1 queue is far shorter than nodeai. No DeepSpeed, no JIT, no
# CUDA module needed.
#
# Runs a 3-sample smoke check under CUDA_LAUNCH_BLOCKING first (so any indexing
# device-assert reports at its TRUE line in seconds), then the full n=300 gate
# only if the smoke check exits clean.
#
#   sbatch script_examples/gate_stage2.sbatch                      # pilot ckpt
#   sbatch --export=ALL,STUDENT=/path/to/ckpt script_examples/gate_stage2.sbatch
#   sbatch --export=ALL,SKIP_SMOKE=1 script_examples/gate_stage2.sbatch   # full only
set -eo pipefail

STUDENT="${STUDENT:-/scratch/$USER/monet_ckpts/sft2_pilot500}"
TEACHER="${TEACHER:-/scratch/$USER/monet_weights/Monet-SFT-7B/stage1}"
DATA="${DATA:-/scratch/$USER/monet_data/Monet-SFT-125K}"
ALIGN_IDX="${ALIGN_IDX:-20,21,22,23,24,25,26,27,28}"
LATENT_SIZE="${LATENT_SIZE:-8}"

source ~/.bashrc || true
conda activate monet
cd "${REPO_DIR:-$SLURM_SUBMIT_DIR}"
echo "=== CODE: $(git rev-parse --short HEAD 2>/dev/null || echo unknown) ($(git log -1 --format=%cd --date=short 2>/dev/null)) on $(hostname), job ${SLURM_JOB_ID:-none} ==="
if git fetch -q origin 2>/dev/null && [ "$(git rev-parse HEAD)" != "$(git rev-parse '@{u}')" ]; then
  echo "WARN: checkout is BEHIND origin -- this job is running STALE code (git pull --ff-only before sbatch)"
fi
mkdir -p "/scratch/$USER/logs"

# Pull the latest gate fix (harmless if already current). Never abort the job on
# a pull hiccup -- the on-disk code may already be up to date.
if git pull --ff-only 2>/dev/null; then echo "=== after pull: $(git rev-parse --short HEAD) ==="; else echo "WARN: git pull skipped (not fast-forwardable); using on-disk code"; fi

run_gate () {  # $1 = n, $2 = extra label, remaining = env prefix handled by caller
  python -m src.gate_stage2 \
    --student "$STUDENT" \
    --teacher "$TEACHER" \
    --data_path "$DATA/Visual_CoT/train.json" \
    --dataset_root "$DATA" \
    --latent_size "$LATENT_SIZE" \
    --n "$1" \
    --alignment_layer_indices "$ALIGN_IDX" \
    --out "/scratch/$USER/logs/gate_$(basename "$STUDENT")_${2}_${SLURM_JOB_ID}.json"
}

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  echo "=== SMOKE: n=3 under CUDA_LAUNCH_BLOCKING (true-line asserts) ==="
  if ! CUDA_LAUNCH_BLOCKING=1 run_gate 3 smoke; then
    echo "FATAL: n=3 smoke check failed -- NOT launching the full n=300. See trace above."
    exit 1
  fi
  echo "=== SMOKE PASSED -- proceeding to full gate ==="
fi

echo "=== FULL: n=${N:-300} ==="
run_gate "${N:-300}" full
echo "Gate done -> /scratch/$USER/logs/gate_$(basename "$STUDENT")_full_${SLURM_JOB_ID}.json"
