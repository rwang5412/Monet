#!/usr/bin/env bash
# Install ALL of VLMEvalKit's dependencies into the current (monet) conda env in one
# shot, WITHOUT disturbing the Monet inference stack. VLMEvalKit's requirements can
# otherwise pull a different torch/vllm/transformers and break the latent runner, so
# we pin those (and a few friends) to their currently-installed versions via a pip
# constraints file, then install VLMEvalKit's deps under that constraint.
#
#   conda activate monet
#   VLME=/scratch/$USER/VLMEvalKit bash evaluation/capimagine/vlmevalkit/install_deps.sh
set -eo pipefail

VLME="${VLME:-/scratch/$USER/VLMEvalKit}"
CONSTRAINTS="$(mktemp)"

echo "== Pinning the Monet stack to its current versions (constraints) =="
python - "$CONSTRAINTS" <<'PY'
import sys, importlib.metadata as m
protect = ["torch", "torchvision", "torchaudio", "vllm", "transformers",
           "huggingface-hub", "tokenizers", "qwen-vl-utils", "numpy",
           "pillow", "flash-attn", "xformers", "accelerate"]
lines = []
for p in protect:
    try:
        v = m.version(p)
        lines.append(f"{p}=={v}")
        print(f"  {p}=={v}")
    except m.PackageNotFoundError:
        pass
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
PY

echo "== Installing VLMEvalKit + its declared deps (constrained) =="
cd "$VLME"
pip install -e . -c "$CONSTRAINTS"

# Belt-and-suspenders: also honor a requirements.txt if the repo ships one.
if [ -f requirements.txt ]; then
  echo "== Installing requirements.txt (constrained) =="
  pip install -r requirements.txt -c "$CONSTRAINTS"
fi

rm -f "$CONSTRAINTS"

echo "== Verifying the protected stack is unchanged =="
python -c "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"

echo "== Checking the import chain (no GPU) =="
if python run.py --help >/dev/null 2>&1; then
  echo "OK: run.py imports cleanly. You can submit the sbatch."
else
  echo "run.py still fails to import -- rerun it directly to see which module:"
  echo "    cd $VLME && python run.py --help"
fi
