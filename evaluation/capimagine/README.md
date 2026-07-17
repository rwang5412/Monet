# CapImagine do(Z) harness — base Monet-7B

Reproduces the **CapImagine** causal test on Monet: does the answer actually
depend on the continuous latent "visual imagination" tokens Z, or does the model
route around them through the raw image (Z is cosmetic)? We intervene on Z during
**real generation** (`do(Z)`) and measure the change in task accuracy.

- `Δ ≈ 0` → latents are cosmetic (the expected finding on the base model).
- `Δ < 0` (accuracy drops under corruption) → latents are load-bearing.

Scope of this build: **do(Z) accuracy only**, on **V\*** and **HR-Bench 4k**,
against **base Monet-7B** (`NOVAglow646/Monet-7B`). The teacher-forced
NLL/mediation harness and the probe-R²/ROI metrics are intentionally *not* here —
those need ROI supervision targets the benchmark datasets don't carry.

## How it works

Monet does its latent reasoning inside **vLLM**: when the model samples the
latent-start token `151666`, the runner feeds each step's last hidden state back
as the next input embedding for `LATENT_SIZE` (=10) steps. The single
intervention point is that fed-back tensor in
[`inference/vllm/monet_gpu_model_runner.py`](../../inference/vllm/monet_gpu_model_runner.py)
(`st["pending"] = last_token_h[i].detach()`).

[`inference/vllm/monet_latent_hook.py`](../../inference/vllm/monet_latent_hook.py)
wraps that tensor, driven by env vars so it survives the vLLM driver→worker
process split:

| `MONET_LATENT_MODE` | effect |
|---|---|
| `off` (default) | pass-through; normal inference |
| `capture` | accumulate μ/σ of Z, dump to `MONET_LATENT_STATS`; model unperturbed (this **is** the clean do(Z) baseline) |
| `corrupt_mean` | replace every latent with the global mean μ (collapse) |
| `corrupt_gauss` | replace every latent with `N(μ, σ)` (destroy) |

Corruptions are matched to the empirical μ/σ, so a drop reflects lost latent
*information*, not an out-of-distribution embedding shock.

## Running on Palmetto

**1. Login/DTN node (has network) — download weights + pre-cache datasets:**
```bash
conda activate monet
bash evaluation/capimagine/palmetto/download_monet_weights.sh
# then confirm the dataset schemas resolve:
python -m evaluation.capimagine.datasets --inspect vstar
python -m evaluation.capimagine.datasets --inspect hrbench_4k
```

**2. Smoke test FIRST — does base Monet-7B emit latents on these datasets?**
The whole harness is void if the model never enters latent mode (emission is
gated by the system prompt).
```bash
srun --gpus-per-node=a100:1 --mem=60G --time=00:30:00 --pty \
  python -m evaluation.capimagine.smoke_latent_emission --model /scratch/haizhow/weights/Monet-7B --dataset vstar
```
Expect a non-zero latent-emission rate. If it's 0, stop and fix the prompt /
`LATENT_SIZE` / checkpoint before going further.

**3. Full harness (clean → corrupt_gauss → corrupt_mean):**
```bash
sbatch --export=ALL,DATASET=vstar     evaluation/capimagine/palmetto/run_capimagine.sbatch
sbatch --export=ALL,DATASET=hrbench_4k evaluation/capimagine/palmetto/run_capimagine.sbatch
```
Progress → `.err`, accuracies/Δ → `.out`; per-example JSON + `*_stats.pt` under
`/scratch/haizhow/results/capimagine/<dataset>/`.

## Reading the output

Each corruption pass prints, over the **clean-latent-emitting** samples only
(do(Z) is undefined where no latent was emitted; `N` is reported):

```
clean_acc   = ...
corrupt_acc = ...
DELTA       = ±...   (<0 => load-bearing; ~0 => cosmetic)
frac_text_changed = ...   (GUARD)
```

**The guard is the key sanity check.** Under a destructive corruption the
*generated text* must change. If `frac_text_changed ≈ 0` while Δ ≈ 0, the hook
isn't firing — that's a **bug**, not the CapImagine finding. Only trust a Δ ≈ 0
result when the text is genuinely changing.

## Gotchas to check on first run

- **Verify the V\* gold field before trusting any accuracy number.** The loader
  falls back to grabbing the first A–H character if it can't find a clean answer
  column, which can yield a confident *wrong* letter. Run
  `python -m evaluation.capimagine.datasets --inspect vstar` and confirm the gold
  answer is a bare option letter (or that options render into a matchable block).
- **Low/zero latent emission?** The system prompt is sent as a `system` role per
  the README. If the smoke test shows little emission, try prepending it to the
  user text instead — chat-template handling of system turns is the likely cause.
- **HR-Bench 4k is high-res.** If it errors or truncates, raise `--max-model-len`
  (default 8192); vision tokens + question + latents + answer can overflow.

## Deferred (not in this build)

- Teacher-forced NLL / `proportion_mediated` (needs an `override_latent_embeds`
  hook in the HF forward and ROI targets).
- `gauss_add` one-shot corruption (needs per-example clean-latent replay across
  passes / req_id matching in vLLM).
- Probe-R², effective-rank, directed-flip representation metrics (need ROI
  supervision targets absent from these benchmarks).
