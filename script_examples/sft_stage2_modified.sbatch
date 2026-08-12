#!/usr/bin/env bash
#SBATCH --job-name=sft2_mod
#SBATCH --output=/scratch/%u/logs/sft2_mod_%j.out
#SBATCH --error=/scratch/%u/logs/sft2_mod_%j.err
#SBATCH --gpus-per-node=a100:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=750G
#SBATCH --time=72:00:00
# Palmetto 2 forbids --mem=0; request an explicit high value. Job 14946762
# host-OOM'd at ~step 190 under --mem=320G (32 pinned dataloader workers +
# DeepSpeed pinned CPU offload). 750G is ~2.3x that; raise via CLI --mem if the
# node has more (sinfo -o "%n %m" -N). See MONET_NUM_WORKERS / MONET_PIN_MEMORY.
#
# Modified SFT Stage 2 on the ORIGINAL Monet stage-1 warm-up checkpoint.
# Two changes vs the paper's stage 2 (both default-ON here):
#   Change 1 - residual obs alignment: student obs states must be closer to the
#     WITH-aux-image teacher (h_pos) than the WITHOUT-aux-image teacher (h_neg)
#     by a margin; requires BOTH precomputed caches (see sft_stage2.sh step 1,
#     and the same command + --no_aux_images for h_neg).
#   Change 2 - latent grounding InfoNCE: each latent block must match its own
#     aux image's post-merger tokens against a 4096-slot memory queue.
#   Prerequisite mask (do NOT drop): --observation_tokens_cannot_see_question_image
#     blocks obs tokens from reading the question image directly, so the only
#     path to the visual residual runs through the latents.
#
# The precompute jobs MUST be finished first -- the trainer hard-fails on any
# missing rep file. Override via --export, e.g.:
#   sbatch --export=ALL,EPOCHS=1 script_examples/sft_stage2_modified.sbatch
set -eo pipefail

DATA="${DATA:-/scratch/$USER/monet_data/Monet-SFT-125K}"
STAGE1="${STAGE1:-/scratch/$USER/monet_weights/Monet-SFT-7B/stage1}"   # student init AND the teacher used for the caches
REPS_POS="${REPS_POS:-/scratch/$USER/monet_ckpts/teacher_reps_pos}"
REPS_NEG="${REPS_NEG:-/scratch/$USER/monet_ckpts/teacher_reps_neg}"

# Paper hyperparameters (Table 7) -- unchanged.
LATENT_SIZE="${LATENT_SIZE:-8}"
CE_EMPHASIZE_FACTOR="${CE_EMPHASIZE_FACTOR:-4.0}"
ALIGNMENT_WEIGHT="${ALIGNMENT_WEIGHT:-2.0}"
EMPHASIZE_LATENT_WEIGHT="${EMPHASIZE_LATENT_WEIGHT:-2.0}"
EPOCHS="${EPOCHS:-2}"
# Modification hyperparameters.
# MARGIN is bounded by the TEACHERS' own separation: max achievable s_pos-s_neg =
# 1-cos(h_pos,h_neg) ~= 0.04-0.09 on the kept layers (layer_sel probe, n=500).
# 0.2 would be unsatisfiable (hinge_active_frac pinned at 1). 0.05 ~= the mean.
MARGIN="${MARGIN:-0.05}"
GROUNDING_WEIGHT="${GROUNDING_WEIGHT:-1.0}"
GROUNDING_QUEUE="${GROUNDING_QUEUE:-4096}"
# From the layer_sel probe (upper third by separation); caches must be built with
# the same list via --keep_layers.
ALIGN_IDX="${ALIGN_IDX:-20,21,22,23,24,25,26,27,28}"
# Holdout: train on the first N processed rows so the file's tail (~500 raw rows,
# covering gate_stage2's last-300 eval slice) is never seen in training.
NUM_SAMPLES="${NUM_SAMPLES:-118000}"

SAVE_CKPT="${SAVE_CKPT:-sft_stage2_residual${MARGIN}_ground${GROUNDING_WEIGHT}_latent${LATENT_SIZE}}"
SAVE="/scratch/$USER/monet_ckpts/${SAVE_CKPT}"

source ~/.bashrc || true
conda activate monet
cd "${REPO_DIR:-$SLURM_SUBMIT_DIR}"
mkdir -p "/scratch/$USER/logs" "$SAVE"

# DeepSpeed JIT-compiles CUDA ops at engine init and needs the CUDA toolkit
# (CUDA_HOME/nvcc) -- not provided by the conda env. The toolkit MAJOR version
# must match torch's CUDA build (12.x): Palmetto's default module is cuda/13,
# so we pin 12.3.0 unless CUDA_MODULE overrides.
CUDA_MODULE="${CUDA_MODULE:-cuda/12.3.0}"
module load "$CUDA_MODULE" 2>/dev/null || module load cuda 2>/dev/null || true
# C++ compiler for JIT extensions (cpu_adam): no c++ on the default PATH at all
# ('c++: command not found' -- the real cause of the cpu_adam build failures).
GCC_MODULE="${GCC_MODULE:-gcc/12.3.0}"
module load "$GCC_MODULE" 2>/dev/null || module load gcc 2>/dev/null || true
command -v c++ >/dev/null 2>&1 || echo "WARNING: no c++ after module load; prebuilt cpu_adam.so required"
command -v nvcc >/dev/null 2>&1 || { \
  echo "FATAL: nvcc not found (tried $CUDA_MODULE). Run 'module avail cuda' and resubmit with CUDA_MODULE=<name>."; exit 1; }
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
echo "CUDA_HOME=$CUDA_HOME ($(nvcc --version | tail -1))"
# Triton autotune cache off NFS (silences the DeepSpeed warning, avoids exit hangs)
export TRITON_CACHE_DIR="/scratch/$USER/.triton_cache"
mkdir -p "$TRITON_CACHE_DIR"
# Torch JIT extensions (DeepSpeedCPUAdam) on scratch; PREBUILD once on the login
# node (see repo notes) -- multi-rank JIT races on NFS home produce a broken
# cpu_adam.so ('cannot open shared object file', jobs 14912695/6).
export TORCH_EXTENSIONS_DIR="/scratch/$USER/torch_extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR"
# Plan B default: no-JIT offload (torch AdamW does the CPU step). cpu_adam JIT is
# unbuildable on the H100 nodes (gcc 12.3 emits AVX512-FP16 the system assembler
# doesn't know: 'no such instruction: vmovw'). Client optimizer must be the
# non-fused torch AdamW since its states live on CPU.
export MONET_OPTIM="${MONET_OPTIM:-adamw_torch}"

# Host-RAM controls (job 14946762 OOM). Fewer, unpinned dataloader workers cut
# the pinned-memory footprint that stacked on DeepSpeed's CPU-offload buffers.
export MONET_NUM_WORKERS="${MONET_NUM_WORKERS:-2}"
export MONET_PIN_MEMORY="${MONET_PIN_MEMORY:-0}"

# Fail fast if the caches are missing (cheaper than dying mid-epoch).
# NOTE: no `ls | head` here -- with 118K files head's early exit SIGPIPEs ls and
# pipefail kills the job (exit 141). find -quit stops after the first match.
for d in "$REPS_POS" "$REPS_NEG"; do
  first=$(find "$d" -maxdepth 1 -name 'rep_*.pt' -print -quit 2>/dev/null)
  if [ -z "$first" ]; then echo "FATAL: teacher rep cache empty/missing: $d"; exit 1; fi
  echo "cache OK: $d (found $first)"
done

torchrun --nproc-per-node="${NPROC:-8}" --master-port="${PORT:-29511}" -m src.main \
  --epochs "$EPOCHS" \
  --bsz 1 \
  --grad_accum_steps 16 \
  --stage sft_stage2 \
  --data_path "$DATA/Visual_CoT/train.json" \
  --log_file "/scratch/$USER/logs/sft2_mod_train.log" \
  --load_model_path "$STAGE1" \
  --save_model_path "$SAVE" \
  --dataset_root "$DATA" \
  --deepspeed "${DS_CONFIG:-./deepspeed/ds_zero2_offload_nojit.json}" \
  --latent_size "$LATENT_SIZE" \
  --alignment_weight "$ALIGNMENT_WEIGHT" \
  --ce_emphasize_factor "$CE_EMPHASIZE_FACTOR" \
  --emphasize_latent_weight "$EMPHASIZE_LATENT_WEIGHT" \
  --alignment_layer all_layers \
  --teacher_reps_dir "$REPS_POS" \
  --teacher_reps_neg_dir "$REPS_NEG" \
  --alignment_layer_indices "$ALIGN_IDX" \
  --num_samples "$NUM_SAMPLES" \
  --obs_residual_margin "$MARGIN" \
  --grounding_weight "$GROUNDING_WEIGHT" \
  --grounding_queue_size "$GROUNDING_QUEUE" \
  --observation_tokens_cannot_see_question_image \
  --save_freq "${SAVE_FREQ:-150}" \
  ${RESUME:+--resume_from_checkpoint} \
  ${WANDB_NAME:+--wandb_name "$WANDB_NAME"}

echo "Modified stage-2 done -> $SAVE"
