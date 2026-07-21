#!/usr/bin/env bash
#SBATCH --job-name=vlmeval
#SBATCH --output=/scratch/%u/logs/vlmeval_%j.out
#SBATCH --error=/scratch/%u/logs/vlmeval_%j.err
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#
# Reproduce a Monet benchmark number through VLMEvalKit (accuracy only; no do(Z)).
# The VStarBench TSV MUST already be cached in $LMUData -- run this once on the
# LOGIN node first (compute nodes are air-gapped):
#   export LMUData=/scratch/$USER/LMUData
#   python -c "from vlmeval.dataset import build_dataset; build_dataset('VStarBench')"
#
# Then submit, e.g.:
#   sbatch --export=ALL,MODEL=Monet-SFT-7B,LATENT_SIZE=16 \
#     evaluation/capimagine/vlmevalkit/run_vlmeval.sbatch
set -eo pipefail

VLME="${VLME:-/scratch/$USER/VLMEvalKit}"
MODEL="${MODEL:-Monet-SFT-7B}"
DATA="${DATA:-VStarBench}"
export LATENT_SIZE="${LATENT_SIZE:-16}"
export LMUData="${LMUData:-/scratch/$USER/LMUData}"

mkdir -p "/scratch/$USER/logs"
source ~/.bashrc || true
conda activate monet

# vLLM V1 workers must spawn; PYTHONPATH must include $VLME so sitecustomize.py is
# auto-imported at startup (in the driver AND spawned workers) and swaps the runner.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="$VLME:${PYTHONPATH:-}"
# Offline: the TSV must already be cached in $LMUData.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"

# --- keep the log clean ---
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"   # quiet vLLM INFO spam
# Exact-match scoring only. Compute nodes are air-gapped, so the default gpt-4o-mini
# judge spams 401 / KeyError:'choices' for every sample. Run the Claude/DeepSeek
# judge separately on the login node with --reuse if you want the +2-3 pts.
JUDGE="${JUDGE:-exact_matching}"
touch "$VLME/.env" 2>/dev/null || true   # silence the repeated "no .env file" error

# Latent-emission log: the patched Qwen wrapper appends "1"/"0" per sample here
# (1 = the output contained the latent-start token 151666). Cleared before each run.
export MONET_EMIT_LOG="/scratch/$USER/results/emit/${MODEL}_${DATA}_emit.log"
mkdir -p "$(dirname "$MONET_EMIT_LOG")"; rm -f "$MONET_EMIT_LOG"

cd "$VLME"
echo "VLME=$VLME MODEL=$MODEL DATA=$DATA LATENT_SIZE=$LATENT_SIZE JUDGE=$JUDGE"
python run.py --data "$DATA" --model "$MODEL" --judge "$JUDGE"

echo "Results under $VLME/outputs/$MODEL/"
if [ -f "$MONET_EMIT_LOG" ]; then
  total=$(wc -l < "$MONET_EMIT_LOG")
  emit=$(grep -c '^1$' "$MONET_EMIT_LOG" || true)
  echo "=== latent emission: ${emit} / ${total} samples emitted <abs_vis_token> (151666) ==="
fi
