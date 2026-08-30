#!/usr/bin/env bash
#SBATCH --job-name=harvest_s3
#SBATCH --output=/scratch/%u/logs/harvest_s3_%j.out
#SBATCH --error=/scratch/%u/logs/harvest_s3_%j.err
#SBATCH --gpus-per-node=a100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=750G
#SBATCH --time=12:00:00
#
# STEP 1 of modified Stage 3: harvest the TARGET latents h*_latent from our
# MODIFIED Stage-2 checkpoint. Forward-only (no optimizer) -> lighter than
# training. Writes one latent file per sample; --resume skips existing.
# These become --teacher_latent_dir for the Stage-3 run.
#
#   sbatch --gpus-per-node=h100:8 script_examples/harvest_stage3_targets.sbatch
set -eo pipefail

DATA="${DATA:-/scratch/$USER/monet_data/Monet-SFT-125K}"
STAGE2="${STAGE2:-/scratch/$USER/monet_ckpts/sft_stage2_residual0.05_ground1.0_latent8}"
SAVE="${SAVE:-/scratch/$USER/monet_ckpts/teacher_latents_modified}"
LATENT_SIZE="${LATENT_SIZE:-8}"

source ~/.bashrc || true
conda activate monet
cd "${REPO_DIR:-$SLURM_SUBMIT_DIR}"
echo "=== CODE: $(git rev-parse --short HEAD 2>/dev/null || echo unknown) ($(git log -1 --format=%cd --date=short 2>/dev/null)) on $(hostname), job ${SLURM_JOB_ID:-none} ==="
if git fetch -q origin 2>/dev/null && [ "$(git rev-parse HEAD)" != "$(git rev-parse '@{u}')" ]; then
  echo "WARN: checkout is BEHIND origin -- this job is running STALE code (git pull --ff-only before sbatch)"
fi
mkdir -p "/scratch/$USER/logs" "$SAVE"

CUDA_MODULE="${CUDA_MODULE:-cuda/12.3.0}"
module load "$CUDA_MODULE" 2>/dev/null || module load cuda 2>/dev/null || true
GCC_MODULE="${GCC_MODULE:-gcc/12.3.0}"
module load "$GCC_MODULE" 2>/dev/null || module load gcc 2>/dev/null || true
command -v nvcc >/dev/null 2>&1 || { echo "FATAL: nvcc not found"; exit 1; }
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
export TRITON_CACHE_DIR="/scratch/$USER/.triton_cache"; mkdir -p "$TRITON_CACHE_DIR"
export TORCH_EXTENSIONS_DIR="/scratch/$USER/torch_extensions"; mkdir -p "$TORCH_EXTENSIONS_DIR"
export MONET_OPTIM="${MONET_OPTIM:-adamw_torch}"
export MONET_NUM_WORKERS="${MONET_NUM_WORKERS:-2}"
export MONET_PIN_MEMORY="${MONET_PIN_MEMORY:-0}"

torchrun --nproc-per-node="${NPROC:-8}" --master-port="${PORT:-29521}" -m src.precompute_teacher_latents \
  --bsz 1 \
  --data_path \
    "$DATA/Visual_CoT/train.json" \
    "$DATA/CogCoM/train.json" \
    "$DATA/ReFocus/train.json" \
    "$DATA/Zebra_CoT_count/train.json" \
    "$DATA/Zebra_CoT_visual_search/train.json" \
    "$DATA/Zebra_CoT_geometry/train.json" \
  --load_model_path "$STAGE2" \
  --save_model_path "$SAVE" \
  --dataset_root "$DATA" \
  --deepspeed "${DS_CONFIG:-./deepspeed/ds_zero2_offload_nojit.json}" \
  --latent_size "$LATENT_SIZE" \
  --output_hidden_states \
  --resume

echo "Harvest done -> $SAVE"
