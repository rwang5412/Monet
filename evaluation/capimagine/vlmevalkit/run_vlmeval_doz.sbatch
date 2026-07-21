#!/usr/bin/env bash
#SBATCH --job-name=vlmeval_doz
#SBATCH --output=/scratch/%u/logs/vlmeval_doz_%j.out
#SBATCH --error=/scratch/%u/logs/vlmeval_doz_%j.err
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#
# CapImagine do(Z) through VLMEvalKit: three passes in order (capture writes the
# mu/sigma stats the corruptions read), each scored by VLMEvalKit's own extractor.
#   capture       -> clean baseline accuracy (hook is pass-through, records stats)
#   corrupt_mean  -> every latent replaced with the global mean mu   (collapse)
#   corrupt_gauss -> every latent replaced with N(mu, sigma)         (destroy)
# do(Z) result = corrupt_acc - clean_acc  (< 0 => latents load-bearing; ~0 => cosmetic).
# The hook config is SHELL-exported before each run so it reaches the vLLM workers.
#
#   sbatch --export=ALL,MODEL=Monet-SFT-7B,LATENT_SIZE=16 \
#     evaluation/capimagine/vlmevalkit/run_vlmeval_doz.sbatch
set -eo pipefail

VLME="${VLME:-/scratch/$USER/VLMEvalKit}"
MODEL="${MODEL:-Monet-SFT-7B}"
DATA="${DATA:-VStarBench}"
export LATENT_SIZE="${LATENT_SIZE:-16}"
export LMUData="${LMUData:-/scratch/$USER/LMUData}"
STATS="${STATS:-/scratch/$USER/results/vlmeval_doz/${DATA}_stats.pt}"

mkdir -p "/scratch/$USER/logs" "$(dirname "$STATS")"
source ~/.bashrc || true
conda activate monet

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="$VLME:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"

cd "$VLME"
echo "VLME=$VLME MODEL=$MODEL LATENT_SIZE=$LATENT_SIZE STATS=$STATS"

for MODE in capture corrupt_mean corrupt_gauss; do
  echo "======== do(Z) mode=$MODE ========"
  export MONET_LATENT_MODE="$MODE"
  export MONET_LATENT_STATS="$STATS"
  export MONET_LATENT_SEED=0
  python run.py --data "$DATA" --model "$MODEL" --work-dir "outputs/doz_${MODE}" --verbose
done

echo "=== do(Z) accuracy summary ==="
for MODE in capture corrupt_mean corrupt_gauss; do
  echo "--- $MODE ---"
  cat outputs/doz_${MODE}/${MODEL}/*_acc.csv 2>/dev/null || echo "(no acc.csv)"
done
