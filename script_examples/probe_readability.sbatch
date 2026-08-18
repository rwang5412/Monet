#!/usr/bin/env bash
#SBATCH --job-name=probe_read
#SBATCH --output=/scratch/%u/logs/probe_read_%j.out
#SBATCH --error=/scratch/%u/logs/probe_read_%j.err
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#
# Linear-vs-MLP readability probe on a trained Stage-2 checkpoint (inference +
# closed-form ridge + tiny MLP fit). Same short a100:1 queue as the gate.
set -eo pipefail

STUDENT="${STUDENT:-/scratch/$USER/monet_ckpts/sft_stage2_residual0.05_ground1.0_latent8}"
DATA="${DATA:-/scratch/$USER/monet_data/Monet-SFT-125K}"

source ~/.bashrc || true
conda activate monet
cd "${REPO_DIR:-$SLURM_SUBMIT_DIR}"
mkdir -p "/scratch/$USER/logs"

python -m src.probe_readability \
  --student "$STUDENT" \
  --data_path "$DATA/Visual_CoT/train.json" \
  --dataset_root "$DATA" \
  --latent_size "${LATENT_SIZE:-8}" \
  --n "${N:-300}" --n_fit "${N_FIT:-200}" \
  --out "/scratch/$USER/logs/probe_read_$(basename "$STUDENT")_${SLURM_JOB_ID}.json"
