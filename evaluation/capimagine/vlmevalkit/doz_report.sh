#!/usr/bin/env bash
# Quietly DeepSeek-judge every do(Z) pass and print a clean CapImagine report.
# VLMEvalKit's per-call logging (which echoes your API key) is redirected to log
# files, so nothing sensitive hits the terminal. Run on the LOGIN node (needs the
# DeepSeek key in $VLME/.env for the judge).
#
#   MONET=/scratch/$USER/Monet VLME=/scratch/$USER/VLMEvalKit \
#     bash evaluation/capimagine/vlmevalkit/doz_report.sh
set -eo pipefail

VLME="${VLME:-/scratch/$USER/VLMEvalKit}"
MONET="${MONET:-/scratch/$USER/Monet}"
DATA="${DATA:-VStarBench}"
MODELS="${MODELS:-Monet-7B-readme Monet-SFT-7B-readme}"
LOGDIR="/scratch/$USER/results/vlmeval_doz/judge_logs"
mkdir -p "$LOGDIR"

cd "$VLME"
set -a; [ -f .env ] && . ./.env; set +a
if [ -z "${DEEPSEEK_API_KEY:-}${OPENAI_API_KEY:-}" ]; then
  echo "FATAL: no DEEPSEEK_API_KEY/OPENAI_API_KEY in $VLME/.env -- VLMEvalKit would silently"
  echo "       fall back to exact matching (job logs: 'API Key: ' empty, KeyError 'choices')."; exit 1
fi
export PYTHONPATH="$MONET:${PYTHONPATH:-}"
echo "Judging do(Z) passes with DeepSeek (logs -> $LOGDIR) ..."
for MODEL in $MODELS; do
  for M in ${MODES:-capture corrupt_mean corrupt_gauss swap}; do
    [ -d "outputs/doz_$M/$MODEL" ] || continue
    printf "  %-22s %-14s ... " "$MODEL" "$M"
    # Drop the latent-step junk tokens the runner leaks into the text; otherwise
    # every extractor reads the stray leading letter as the answer (~chance).
    python -m evaluation.capimagine.vlmevalkit.strip_latent_prefix \
      "outputs/doz_$M/$MODEL/${MODEL}_${DATA}.xlsx" 2>/dev/null | tr '\n' ' ' || true
    find "outputs/doz_$M/$MODEL" -name "*_acc.csv" -o -name "*result.pkl" \
      -o -name "*result.xlsx" 2>/dev/null | xargs -r rm -f
    if python run.py --data "$DATA" --model "$MODEL" --judge deepseek-chat \
         --reuse --work-dir "outputs/doz_$M" >"$LOGDIR/${MODEL}_${M}.log" 2>&1; then
      echo "done"
    else
      echo "FAILED (see $LOGDIR/${MODEL}_${M}.log)"
    fi
  done
done

echo
export PYTHONPATH="$MONET:${PYTHONPATH:-}"
for MODEL in $MODELS; do
  python -m evaluation.capimagine.vlmevalkit.score_doz \
    --vlme "$VLME" --model "$MODEL" --data "$DATA"
  echo
done
