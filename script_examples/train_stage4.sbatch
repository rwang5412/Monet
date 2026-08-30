#!/usr/bin/env bash
#SBATCH --job-name=monet_stage4
#SBATCH --output=/scratch/%u/logs/stage4_%j.out
#SBATCH --error=/scratch/%u/logs/stage4_%j.err
#SBATCH --gpus-per-node=a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#
# Stage 4: causal latent training. Arms (design doc §7):
#   arm0: WD=0 WN=0 WS=0 WNEC=0            (byte-identical baseline)
#   arm1: WD=1.0 WN=0.2 WS=0 WNEC=0        (writer only)
#   arm2: WD=1.0 WN=0.2 WS=1.0 WNEC=0.1    (full stack)
#   arm2b: as arm2 with SWAP_OBS_ONLY=1    (attribution — wire before running)
#
#   sbatch --export=ALL,ARM=arm2,BASE=/scratch/$USER/monet_weights/Monet-SFT-7B/stage3 \
#     scripts/train_stage4.sbatch
#
# NOTE: ~4x vanilla activation memory (4 grad-relevant forwards). Run the 5-step
# memory check first (STEPS=5). If it does not fit, drop F4 first (WNEC=0).
set -eo pipefail

ARM="${ARM:-arm0}"
# From-scratch default: BASE Qwen2.5-VL (pre-download to scratch; compute nodes are
# air-gapped). NEW_TOK=1 is REQUIRED with a base-Qwen BASE (trains the added special
# tokens' embedding/lm_head rows); leave 0 for Monet checkpoints.
BASE="${BASE:-/scratch/$USER/monet_weights/Qwen2.5-VL-7B-Instruct}"
NEW_TOK="${NEW_TOK:-1}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/monet_data/Monet-SFT-125K}"
CF_PAIRS="${CF_PAIRS:-/scratch/$USER/monet_data/cf/pairs.jsonl}"
LATENT_SIZE="${LATENT_SIZE:-8}"     # SFT-trained K
EPOCHS="${EPOCHS:-1}"
SAVE="/scratch/$USER/monet_ckpts/stage4_${ARM}"

case "$ARM" in
  arm0)  WD=0;   WN=0;   WS=0;   WNEC=0   ;;
  arm1)  WD=1.0; WN=0.2; WS=0;   WNEC=0   ;;
  arm2)  WD=1.0; WN=0.2; WS=1.0; WNEC=0.1 ;;
  *) echo "unknown ARM=$ARM"; exit 1 ;;
esac

source ~/.bashrc || true
conda activate monet
cd "${REPO_DIR:-$SLURM_SUBMIT_DIR}"
echo "=== CODE: $(git rev-parse --short HEAD 2>/dev/null || echo unknown) ($(git log -1 --format=%cd --date=short 2>/dev/null)) on $(hostname), job ${SLURM_JOB_ID:-none} ==="
if git fetch -q origin 2>/dev/null && [ "$(git rev-parse HEAD)" != "$(git rev-parse '@{u}')" ]; then
  echo "WARN: checkout is BEHIND origin -- this job is running STALE code (git pull --ff-only before sbatch)"
fi
mkdir -p "/scratch/$USER/logs" "$SAVE"

torchrun --nproc-per-node="${NPROC:-2}" --master-port=29517 -m src.main \
  --epochs "$EPOCHS" --bsz 1 --grad_accum_steps 16 \
  --stage sft_stage4 \
  --data_path \
    "$DATA_ROOT/Visual_CoT/train.json" \
    "$DATA_ROOT/Zebra_CoT_visual_search/train.json" \
  --load_model_path "$BASE" \
  --save_model_path "$SAVE" \
  --dataset_root "$DATA_ROOT" \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --latent_size "$LATENT_SIZE" \
  --lr 1e-5 \
  --decode_weight "$WD" --nce_weight "$WN" \
  --swap_weight "$WS" --necessity_weight "$WNEC" \
  --necessity_margin "${MARGIN:-1.0}" \
  --max_steps "${STEPS:--1}" \
  $([ "$NEW_TOK" = "1" ] && echo --train_new_token_embeddings) \
  --lora_r "${LORA_R:-128}" \
  --cf_pairs_path "$CF_PAIRS" \
  --wandb_name "stage4_${ARM}"

echo "Stage-4 $ARM done -> $SAVE"
