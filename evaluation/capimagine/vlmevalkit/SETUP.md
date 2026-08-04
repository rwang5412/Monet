# Running Monet (accuracy + CapImagine do(Z)) inside VLMEvalKit

VLMEvalKit handles the V* prompt/image/scoring correctly (which our custom harness
did not fully reproduce), and the Monet runner we copy in carries the do(Z) latent
hook. So the same setup gives BOTH a faithful accuracy number AND the do(Z)
intervention -- just toggle `MONET_LATENT_MODE`.

Paths below assume: Monet repo at `$MONET` (e.g. `/scratch/$USER/Monet`), VLMEvalKit
at `$VLME` (e.g. `/scratch/$USER/Monet/VLMEvalKit`), weights under
`/scratch/$USER/monet_weights`. Run everything in the `monet` env (vllm==0.10.0).

## 1. One-time setup

```bash
MONET=/scratch/$USER/Monet
VLME=$MONET/VLMEvalKit          # adjust if you cloned it elsewhere

mkdir -p $VLME/Monet_models
cp $MONET/inference/vllm/monet_gpu_model_runner.py $VLME/Monet_models/
cp $MONET/inference/vllm/monet_latent_hook.py      $VLME/Monet_models/
cp $MONET/evaluation/capimagine/vlmevalkit/sitecustomize.py $VLME/sitecustomize.py
```

Register Monet as a model: open `$VLME/vlmeval/config.py`, find `qwen2vl_series = {`
(~line 1901), and add these two entries inside that dict:

```python
    "Monet-SFT-7B": partial(
        vlm.Qwen2VLChat,
        model_path="/scratch/YOUR_USER/monet_weights/Monet-SFT-7B/stage3",
        min_pixels=256 * 28 * 28,
        max_pixels=8192 * 28 * 28,
        system_prompt="You are an expert multimodal large language model designed to reason with latent visual embeddings.",
        use_vllm=True,
        max_new_tokens=4096,
        post_process=False,
    ),
    "Monet-7B": partial(   # the RL (SFT+VLPO) model
        vlm.Qwen2VLChat,
        model_path="/scratch/YOUR_USER/monet_weights/Monet-7B",
        min_pixels=256 * 28 * 28,
        max_pixels=8192 * 28 * 28,
        system_prompt="You are an expert multimodal large language model designed to reason with latent visual embeddings.",
        use_vllm=True,
        max_new_tokens=4096,
        post_process=False,
    ),
```

Notes:
- The README's alternative system prompt is
  `"You are a helpful multimodal assistant. You are required to answer the question based on the image provided. Put your final answer in \\boxed{}."`
  The two sources conflict; try the expert one first, swap if it undershoots.
- `VStarBench` downloads its TSV to `$LMUData`. On air-gapped compute nodes, run
  once on the login node first (`export LMUData=/scratch/$USER/LMUData`) to cache it.

## 2. Accuracy run (no do(Z))

`MONET_LATENT_MODE` unset -> hook never loads -> plain Monet eval.

```bash
cd $VLME
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LMUData=/scratch/$USER/LMUData
LATENT_SIZE=16 python run.py --data VStarBench --model Monet-SFT-7B --verbose
# result CSV under $VLME/outputs/Monet-SFT-7B/...  target: V* = 82.20
```

This is the ground-truth reference. If it lands ~82, the bug was in our custom
harness; if it is also ~57, the weights genuinely can't hit 82 and we stop blaming
the harness.

## 3. CapImagine do(Z) inside VLMEvalKit

Run VLMEvalKit three times, shell-EXPORTING the hook config so it reaches the
spawned workers (setting it from Python is too late -- vLLM snapshots worker env at
startup). Compare the three accuracy numbers: clean - corrupt = do(Z) Delta,
scored by VLMEvalKit's robust judge.

```bash
cd $VLME
export VLLM_WORKER_MULTIPROC_METHOD=spawn LMUData=/scratch/$USER/LMUData
STATS=/scratch/$USER/results/vlmeval_dozstats/vstar_stats.pt
mkdir -p "$(dirname "$STATS")"

# clean pass (captures mu/sigma AND gives the clean accuracy)
MONET_LATENT_MODE=capture MONET_LATENT_STATS=$STATS LATENT_SIZE=16 \
  python run.py --data VStarBench --model Monet-SFT-7B --work-dir outputs/doz_clean

# collapse: every latent -> global mean
MONET_LATENT_MODE=corrupt_mean MONET_LATENT_STATS=$STATS LATENT_SIZE=16 \
  python run.py --data VStarBench --model Monet-SFT-7B --work-dir outputs/doz_mean

# destroy: every latent -> N(mu, sigma)
MONET_LATENT_MODE=corrupt_gauss MONET_LATENT_STATS=$STATS MONET_LATENT_SEED=0 LATENT_SIZE=16 \
  python run.py --data VStarBench --model Monet-SFT-7B --work-dir outputs/doz_gauss
```

`clean_acc - corrupt_acc`:
- Delta < 0  -> latents are load-bearing (destroying them hurts the answer).
- Delta ~ 0  -> latents are cosmetic (the CapImagine disconnect).

Guard: `corrupt_*` output text MUST differ from clean (the hook fired). If the three
runs are identical, the env didn't reach the worker -- check the sitecustomize print
line shows `MONET_LATENT_MODE=corrupt_*`.
