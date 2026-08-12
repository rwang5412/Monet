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
#   sbatch script_examples/gate_stage2.sbatch                      # pilot ckpt
#   sbatch --export=ALL,STUDENT=/path/to/ckpt script_examples/gate_stage2.sbatch
set -eo pipefail

STUDENT="${STUDENT:-/scratch/$USER/monet_ckpts/sft2_pilot500}"
TEACHER="${TEACHER:-/scratch/$USER/monet_weights/Monet-SFT-7B/stage1}"
DATA="${DATA:-/scratch/$USER/monet_data/Monet-SFT-125K}"

source ~/.bashrc || true
conda activate monet
cd "${REPO_DIR:-$SLURM_SUBMIT_DIR}"
mkdir -p "/scratch/$USER/logs"

python -m src.gate_stage2 \
  --student "$STUDENT" \
  --teacher "$TEACHER" \
  --data_path "$DATA/Visual_CoT/train.json" \
  --dataset_root "$DATA" \
  --latent_size "${LATENT_SIZE:-8}" \
  --n "${N:-300}" \
  --alignment_layer_indices "${ALIGN_IDX:-20,21,22,23,24,25,26,27,28}" \
  --out "/scratch/$USER/logs/gate_$(basename "$STUDENT")_${SLURM_JOB_ID}.json"
