"""Place this file in the VLMEvalKit repo ROOT, named EXACTLY `sitecustomize.py`.

Python auto-imports the module named `sitecustomize` at interpreter startup in
EVERY process -- the driver AND every spawned vLLM worker. (The Monet README calls
it `sitecustomized.py`; that name is NOT auto-imported and the runner swap silently
never happens. Use `sitecustomize.py`.)

It swaps vLLM's gpu_model_runner for the Monet runner, so when the model emits the
latent-start token (151666) decoding switches to latent mode. The copied runner
also carries the CapImagine do(Z) hook, so setting MONET_LATENT_MODE=capture|
corrupt_mean|corrupt_gauss runs the intervention inside VLMEvalKit's eval.

Prereqs (see SETUP.md): copy BOTH files into VLMEvalKit/Monet_models/
    cp <MONET>/inference/vllm/monet_gpu_model_runner.py  VLMEvalKit/Monet_models/
    cp <MONET>/inference/vllm/monet_latent_hook.py        VLMEvalKit/Monet_models/
"""
import importlib
import os
import sys

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

# Repo root on sys.path so `Monet_models.*` resolves as a namespace package, in the
# driver and (via PYTHONPATH) in spawned workers.
workspace = os.path.abspath(".")
if workspace not in sys.path:
    sys.path.insert(0, workspace)
_old = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = f"{workspace}:{_old}" if _old else workspace

# Monet latent token ids (fixed by the checkpoint's tokenizer).
os.environ.setdefault("LATENT_START_ID", "151666")
os.environ.setdefault("LATENT_END_ID", "151667")
# Test-time latent count K. The runner reads this at construction; if it is 0 the
# model emits ZERO latents (silent no-latent eval). Default to 16 (V* peaks high);
# override by exporting LATENT_SIZE before `python run.py`.
os.environ.setdefault("LATENT_SIZE", "16")

# Swap the runner. Monet_models/ must contain monet_gpu_model_runner.py AND
# monet_latent_hook.py (the runner imports the hook from Monet_models as a fallback).
sys.modules["vllm.v1.worker.gpu_model_runner"] = importlib.import_module(
    "Monet_models.monet_gpu_model_runner")
print("[Monet sitecustomize] swapped vLLM gpu_model_runner; "
      f"LATENT_SIZE={os.environ['LATENT_SIZE']} "
      f"MONET_LATENT_MODE={os.environ.get('MONET_LATENT_MODE', 'off')}")
