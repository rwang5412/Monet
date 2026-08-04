# CapImagine results

## 1. Latent-emission rate: SFT vs RL (V*, matched conditions)

**Setup (identical for both models):** dataset V* (`lmms-lab/vstar-bench`),
prompt style `monet` (the exact RL training prompt from
`RL/examples/format_prompt/monet_format.jinja`, default system message),
`temperature=0.5` (the RL rollout temperature from `RL/examples/vlpo_train.sh`),
n=50 samples, `LATENT_SIZE=10`.

| Model | Emission rate | Count |
|---|---|---|
| SFT (`Monet-SFT-7B/stage3`) | ~0.60 | 30/50 |
| RL  (`Monet-7B`)            | ~0.08 | 4/50 |

95% CIs are non-overlapping (SFT ≈ [0.45, 0.74], RL ≈ [0.02, 0.19]).

**Finding:** RL training sharply reduces how often the model emits latent
"visual imagination" tokens (`<abs_vis_token>`, id 151666) — from ~60% to ~8%
on V*. The RL model still *narrates* the zoom ("...I will zoom in on the
relevant part") but usually does not emit the latent token.

### Confounds ruled out before concluding
- **Prompt.** The original harness used a custom system prompt neither model
  trained on; the RL model emitted 0%. Both models trained with the default
  system message + inline instruction (SFT: `src/task.py`; RL:
  `config_monet.yaml` -> `monet_format.jinja`). Under the correct prompt the
  RL rate rose only to ~6%.
- **Decoding temperature.** Greedy (temp=0) can starve the "sometimes" latent
  token. Raising to the rollout temp 0.5 moved RL from ~6% to ~8–12% — a
  marginal bump, not a rescue. SFT stays high across prompts and temperatures.

### Caveats / open items
- Verify this matches the paper's own inference setup for Monet-7B (prompt,
  temperature, task mix). We used the most in-distribution prompt available
  (the RL training format), so low emission under it is meaningful, but the
  paper's reported numbers may use a different configuration.
- n=50; a full-191 (V*) and HR-Bench 4k run would tighten the rate.

## 2. do(Z) accuracy (clean vs corrupt) — TODO
Not yet run. Note RL's low emission means do(Z) scores over few samples on
V* alone (~4/50); pool V* + HR-Bench 4k for adequate N on the RL side.
