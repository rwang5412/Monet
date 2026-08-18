#!/usr/bin/env bash
#SBATCH --job-name=sft3_pilot
#SBATCH --output=/scratch/%u/logs/sft3_pilot_%j.out
#SBATCH --error=/scratch/%u/logs/sft3_pilot_%j.err
#SBATCH --gpus-per-node=a100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=750G
#SBATCH --time=8:00:00
#
# Modified SFT Stage 3 (PILOT): re-init from the Stage-1 warm-up, align generated
# latents to the harvested MODIFIED-Stage-2 targets, PLUS the decode-CE writer
# loss L_dec (the causal lever; --decode_weight). Aux images are removed (stock
# Stage 3). Decoder is discarded at inference.
#
# PILOT = a data subset for a fast (~4h) read before the full run. Judge on
# free-gen do(Z) + V*, NOT another teacher-forced gate.
#
#   sbatch --gpus-per-node=h100:8 script_examples/sft_stage3_pilot.sbatch
#   sbatch --export=ALL,DECODE_WEIGHT=0 ...   # ablation: stock Stage 3 (no L_dec)
set -eo pipefail

DATA="${DATA:-/scratch/$USER/monet_data/Monet-SFT-125K}"
STAGE1="${STAGE1:-/scratch/$USER/monet_weights/Monet-SFT-7B/stage1}"
TARGETS="${TARGETS:-/scratch/$USER/monet_ckpts/teacher_latents_modified}"
LATENT_SIZE="${LATENT_SIZE:-8}"
ALIGNMENT_WEIGHT="${ALIGNMENT_WEIGHT:-2.0}"
CE_EMPHASIZE_FACTOR="${CE_EMPHASIZE_FACTOR:-4.0}"
DECODE_WEIGHT="${DECODE_WEIGHT:-1.0}"        # writer-side redundancy breaker
SWAP_WEIGHT="${SWAP_WEIGHT:-0.2}"            # reader-side redirect; LOW to preserve accuracy
SWAP_MARGIN="${SWAP_MARGIN:-0.15}"           # causality dial (nats); target modest do(Z)
EPOCHS="${EPOCHS:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-32000}"          # pilot subset; unset/large for full run

SAVE_CKPT="${SAVE_CKPT:-sft_stage3_decode${DECODE_WEIGHT}_latent${LATENT_SIZE}_pilot}"
SAVE="/scratch/$USER/monet_ckpts/${SAVE_CKPT}"

source ~/.bashrc || true
conda activate monet
cd "${REPO_DIR:-$SLURM_SUBMIT_DIR}"
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

# Fail fast if the harvested targets are missing.
first=$(find "$TARGETS" -maxdepth 1 -name '*.pt' -print -quit 2>/dev/null)
[ -z "$first" ] && { echo "FATAL: no target latents in $TARGETS (run harvest_stage3_targets first)"; exit 1; }
echo "targets OK: $TARGETS (found $first)"

# Recentered alignment (mod A): compute the dataset-mean target once (cheap CPU),
# then pass it to training. RECENTER=0 disables.
RECENTER="${RECENTER:-1}"
ALIGN_MEAN="$TARGETS/align_mean.pt"
if [ "$RECENTER" = "1" ]; then
  [ -f "$ALIGN_MEAN" ] || python -m src.compute_target_mean --target_dir "$TARGETS" --out "$ALIGN_MEAN"
  RECENTER_ARG="--align_recenter_path $ALIGN_MEAN"
else
  RECENTER_ARG=""
fi

torchrun --nproc-per-node="${NPROC:-8}" --master-port="${PORT:-29531}" -m src.main \
  --epochs "$EPOCHS" \
  --bsz 1 \
  --grad_accum_steps 16 \
  --stage sft_stage3 \
  --data_path \
    "$DATA/Visual_CoT/train.json" \
    "$DATA/CogCoM/train.json" \
    "$DATA/ReFocus/train.json" \
    "$DATA/Zebra_CoT_count/train.json" \
    "$DATA/Zebra_CoT_visual_search/train.json" \
    "$DATA/Zebra_CoT_geometry/train.json" \
  --log_file "/scratch/$USER/logs/sft3_pilot_train.log" \
  --load_model_path "$STAGE1" \
  --save_model_path "$SAVE" \
  --dataset_root "$DATA" \
  --deepspeed "${DS_CONFIG:-./deepspeed/ds_zero2_offload_nojit.json}" \
  --latent_size "$LATENT_SIZE" \
  --alignment_weight "$ALIGNMENT_WEIGHT" \
  --ce_emphasize_factor "$CE_EMPHASIZE_FACTOR" \
  --alignment_layer all_layers \
  --teacher_latent_dir "$TARGETS" \
  --decode_weight "$DECODE_WEIGHT" \
  --swap_weight "$SWAP_WEIGHT" \
  --swap_margin "$SWAP_MARGIN" \
  $RECENTER_ARG \
  --num_samples "$NUM_SAMPLES" \
  ${WANDB_NAME:+--wandb_name "$WANDB_NAME"}

echo "Stage-3 pilot done -> $SAVE"
