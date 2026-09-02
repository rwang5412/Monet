# CapImagine / Monet causal-latent project — research log

Persistent record of what we set out to do, what we measured, what we found, and
what we changed. Written so that a reader (or a fresh context window) can pick up
the project without re-deriving anything.

Repo: fork `rwang5412/Monet`, branch `capimagine-harness`.
Cluster: Palmetto (`haizhow`), scratch at `/scratch/haizhow`.
Last updated: 2026-09-01.

---

## 1. Goal

Make Monet's latent visual tokens **causally load-bearing** — ablating or swapping
them should change the model's answer — **while preserving benchmark accuracy**
(V* ≈ 80). This is an intervention project, not just a diagnosis.

---

## 2. Baseline: the released checkpoints have cosmetic latents

Reproduced Monet's numbers through VLMEvalKit (README prompt + DeepSeek-V3.1
judge — the two go together; the Appendix-C "expert" prompt makes the model answer
tersely and measures different behavior):

| Model | V* (ours) | V* (paper) |
|---|---|---|
| Monet-SFT-7B (stage 3) | 80.63 | 82.20 |
| Monet-7B (RL) | 82.72 | 83.25 |

**do(Z) intervention result** (3-pass capture / corrupt_mean / corrupt_gauss over
clean-emitting samples):

| Model | do(Z) Δ | guard (frac. of outputs changed) |
|---|---|---|
| Monet-SFT-7B | **0.0** (0/91) | **1.000** |
| Monet-7B (RL) | −1.8% (1/56) | **1.000** |

`guard = 1.000` means the intervention provably fired — 100% of output text
changed — yet the **answers did not move**. The latents are cosmetic: Monet
answers from text CoT plus the full image, not from its "visual thoughts".
The released latents are also collapsed: effective rank ~3 of 3584, ~0.94
pairwise cosine.

**Key baseline fact for interpreting everything below:** changing generated text
does NOT imply changing answers. Any objective that only makes observations
latent-dependent can still leave do(Z) at zero.

---

## 3. Metric definitions (gate + training)

Gate (`src/gate_stage2.py`, held-out last 300 rows, teacher-forced):

- **content gap** = `nll_donor − nll_real` — splice in ANOTHER sample's real
  latents; does predicting THIS sample's observation get worse? Measures whether
  the LM reads *what is in* the latents. **This is the promotion criterion.**
- **presence gap** = `nll_zero − nll_real` — zero the latents entirely. Measures
  whether the LM uses the latents *at all*.
- **textmask gap** = NLL with the question TEXT attention-masked, minus real.
  Large ⇒ the content the model relies on lives in the question text.
- **effective rank** = `exp(entropy of normalized squared singular values)` of the
  latent cloud. Released Monet ≈ 3; want ≫ 10.
- **cross_sample_sim / within_block_sim** — pairwise cosine across samples, and
  across the K slots inside one sample.
- NLL, not accuracy, is the sensitive read: argmax saturates (content can move
  probability mass without flipping the top token).

Promotion rule: `cross_sim < 0.9 AND eff_rank > 10 AND content_gap > 0.02`.

---

## 4. Stage-2 modifications (ours)

Two changes on top of Monet's stage 2:

- **Change 1 — residual observation alignment.** Replace the absolute cosine
  `1 − cos(h*_obs, h_obs)` (which starts near its optimum, since observation
  states are dominated by token identity and context) with a margin:
  `relu(margin − cos(ĥ,h_pos) + cos(ĥ,h_neg))`, where `h_pos` is a frozen stage-1
  teacher WITH the aux image and `h_neg` the same teacher WITHOUT. Everything the
  two teachers share cancels, isolating the visual residual. Layers 20-28.
- **Change 2 — latent grounding InfoNCE.** Pooled latent block → student-side
  projector → match its own aux image's post-merger visual tokens against a
  4096-slot ring queue (MoCo pattern).
- **3-pass mask parity.** Student observation tokens cannot see the question
  image, so the latents are the only visual route. Teachers get a matching mask.

Note: DeepSpeed is incompatible with Monet's Eq.9 latent-only surrogate (the inner
`autograd.grad` crashes the ZeRO reducer), so we use direct backprop; the masks
preserve latent-only routing architecturally.

---

## 5. Stage-2 results — the full run FAILED its gate

**Pilot** (job 14946764, 500 steps): nce_top1 0.72, residual_gap +0.021,
cross_sample_sim drifted 0.5 → 0.81 and never plateaued.

**Full run** (job 14965283, 2 epochs, ckpt
`/scratch/haizhow/monet_ckpts/sft_stage2_residual0.05_ground1.0_latent8`):
training log looked healthy — nce_top1 0.67, residual_gap +0.058, obs_acc 0.68.

**Gate (job 15102023) — FAIL:**

| metric | pilot-500 | full run | verdict |
|---|---|---|---|
| content gap | +0.032 | **+0.0070** | FAIL (need > 0.02) |
| effective rank | 46.9 | 54.4 | pass |
| cross_sample_sim | 0.714 | 0.837 | pass |
| within_block_sim | 0.761 | **0.925** | slots are near-clones |
| presence gap (obs) | +1.18 | +1.42 | — |
| ans content gap | +0.008 | **−0.00002** | answers exactly latent-independent |
| s_pos − s_neg | +0.041 | +0.047 | no mask leak |

The LM depends on latent **presence** ~200× more than on latent **content**.
**More training made causality worse**: the 2-epoch run is worse than its own
500-step pilot on every causal metric. The similarity drift we tracked during the
pilot was the visible symptom of real degradation.

### Root causes (all three found by audit, 2026-09-01)

1. **Shared-direction shortcut.** A large part of `h_pos − h_neg` is a direction
   common to every sample ("I looked at a crop"). The student can earn margin by
   shifting all observation states along it without encoding anything
   sample-specific. Fingerprint: cross_sim climbing 0.5 → 0.867 monotonically
   while content collapsed.
2. **The content objective was numerically negligible.** At end of training the
   residual term contributed `2.0 × 0.018 = 0.037` versus grounding's
   `1.0 × 1.73` — **47× smaller**. The one term driving sample-specific content
   was drowned out by a term that only makes latents retrievable by a separate
   trained projector. (Retrievable ≠ readable by the LM.)
3. **Margin calibrated against a wrong ceiling.** `MARGIN=0.05` came from an early
   probe estimating teacher separation at 0.04-0.09. Measured over 10,000 real
   pairs (`src/compute_residual_mean`), the ceiling is **0.125** (0.085 at layer
   20 → 0.165 at layer 28). The run hit its too-easy target (gap 0.058) and
   stopped pushing — `hinge_active_frac` falling to 0.496 was that happening in
   plain sight. Recentering barely lowers the ceiling (0.121), so it is near-free.

---

### Stage 2 v3 (recentered) — job 15440094, COMPLETED 10h03m

Config: `RECENTER_PATH` on, `MARGIN=0.10`, `ALIGNMENT_WEIGHT=8.0`,
`EMPHASIZE_LATENT_WEIGHT=1.0`, `EPOCHS=1`. Code `681b7f2` (contains `188fc34`).
Checkpoint: `/scratch/haizhow/monet_ckpts/sft_stage2_residual0.10rc_ground1.0_latent8`.

| training metric | v2 (failed gate) | v3 |
|---|---|---|
| residual_gap | +0.058 | **+0.075** |
| hinge_active_frac | 0.496 | **0.698** |
| within_block_sim | 0.938 | **0.722** |
| cross_sample_sim | 0.867 | **0.824** |
| nce_top1 | 0.674 | **0.706** |
| teacher_ce | 0.857 | 1.358 (1 epoch vs 2) |
| obs_token_acc | 0.677 | 0.664 (1 epoch vs 2) |

All three fixes behaved as designed: the objective stayed unsatisfied (hinge 0.70,
not coasting at 0.50), the gap passed the old plateau, and the residual term ran at
`8.0 × 0.043 = 0.34` vs grounding's 2.80 — **8:1 instead of 47:1**.
CAVEAT: v2's training log also looked healthy and still failed the gate. Only
`content_nll_gap` decides promotion. Confound: v3 is 1 epoch, v2 was 2.

Gate result: _pending_.

### Stage 3 with L_dec alive — job 15439209, COMPLETED 12h01m

Ckpt `sft_stage3_decode1.0_latent8_full`. **Interpretability is limited**: it ran
with the upstream slot-axis alignment bug (§6) on targets from the FAILED v2 gate.
Still informative about L_dec itself:
- `decode_loss` 12.13 → **5.32** — L_dec trains (first run where it was ever live).
- `observation_token_acc` **0.879** — the writer loss cost NO accuracy, which was
  the main risk of `decode_weight=1.0`.
- `swap_gap` **0.004** vs a 0.15 margin — the reader lever stayed inert, as in the
  pilot.
- `decode_gap` 0.0 — this run predates the tripwire fix (`0ec50fd`).

---

## 6. Bug ledger — provenance matters

### Inherited from upstream Monet (verified via `git show <pre-branch>:file`)

These exist in the original code, before any of our work. **Every Monet stage-3
run ever performed has had them, including the ones that produced the released
checkpoints.**

1. **The stage-3 alignment loss reduces cosine over the SLOT axis, not features.**
   `modeling_qwen2_5_vl_monet.py:247` called `F.cosine_similarity(t, s)` with no
   `dim=`, so it used the default `dim=1` — on `[num_layer, K, dim]` tensors that
   is the K=8 slot axis. For every (layer, feature) pair it cosined the length-8
   across-slot profile instead of the length-3584 content vector. The objective is
   **invariant to the content it exists to distill**, and it **rewards the slots
   looking alike** — actively training the within-block redundancy (0.925) that
   L_dec was added to break. Manifests as an `alignment_loss` that looks alive but
   sits flat (0.973 → 0.974 → 0.973 in job 15439209).
   Proof it was unintended: the sibling `obs_residual_loss` passes `dim=-1`
   explicitly, and this function's own `dim()==1` branch passes `0` explicitly.
   Fixed: commit `a2d8179`.
2. **Latent slots are gradient-isolated from each other.**
   `modeling_qwen2_5_vl_monet.py:1983` feeds the LM `latent_embed.detach()`, so
   slot *k+1* is produced from slot *k* only through a detached copy. Nothing can
   push a slot to differ from the one before it — an architectural cause of
   within-block redundancy. **NOT fixed** (would require changing Monet's latent
   generation loop).
3. **Harvest image budget mismatch.** `precompute_teacher_latents.py:132`
   hardcoded `1000/500` while stage-2 training uses `1500/1280` and the stage-3
   student sees `2000/2000` — three resolutions in one distillation chain, with
   the checkpoint run at a resolution it never trained at. The slot-count assert
   only checks K, so nothing caught it. Fixed: commit `07daff5`.

**These three together are a coherent explanation for why released Monet's latents
are collapsed and cosmetic**: the distillation objective could not transfer
content, the architecture could not differentiate slots, and the targets were
harvested off-distribution. This is a finding about the paper, not our pipeline.

### Ours (in code we added)

1. **L_dec was silently dead for two full 8×H100 jobs** (15237561, 15427523).
   `decode_loss.py` did `self.proj(z.float())` while the module is attached as
   `.to(model.dtype)` = bf16 → `mat1 and mat2 must have the same dtype`. The
   trainer's `try/except` warned 5× then went silent, and since `decode_loss` was
   only logged when the counter was > 0, **no log key revealed it**. Detected by
   arithmetic: the logged `loss` equalled `student_ce + 2.0×alignment +
   0.2×swap` with nothing left over. Fixed: `e1bfcd6`.
2. **`decode_gap` was structurally 0.0.** It built its shuffled control with
   `torch.roll(z, 1, dims=0)`, and stage 3 trains at bsz=1 where that is a no-op.
   Slot-shuffling is not a fix either — cross-attention over the memory has no
   positional encoding, so the decoder is permutation-invariant across slots.
   Fixed: `0ec50fd` (prefers a real donor from the DonorBank; falls back to
   matched-moment noise, and now logs which control was used).
3. **`emphasize_latent_weight` was a no-op in every stage-2 run.** Direct mode
   (always selected under DeepSpeed) built `loss = teacher_ce + latent_routed`,
   dropping the multiplier the surrogate branch applies. The paper's 2.0 ran the
   aux losses at 1×. Fixed: `188fc34`.
4. **The obs-count-mismatch guard would have killed the run it protects.** The
   modeling accumulator starts as a Python float; at bsz=1 one bad row leaves it a
   float and `alignment_loss.item()` raises `AttributeError`. Fixed: `188fc34`.
5. **The stage-3 CE-only fallback crashed differently.** With `loss_type=['ce']`
   the modeling never sets `loss_dict['alignment']`, so the bare `[]` lookup
   raised `KeyError`. Fixed: `a2d8179`.
6. **Fail-loud guards sat inside the conditions they protect**, so a permanently
   false precondition (decoder not found, latents unstashed, donor bank starved)
   incremented nothing and raised nothing. Moved to method scope: "never
   attempted" now fails as loudly as "always failed". Fixed: `a2d8179`.
7. **Metrics only logged when their objective fired**, so a dead loss produced no
   key and absence was the only signal. Now every enabled objective logs
   unconditionally (0.0 when dead) plus `*_fired` fractions. Fixed: `a2d8179`.
8. Training-log `nce_top1` is measured against a queue that already contains its
   own positive; training-log `within_block_sim` actually measures *across*
   blocks, not within one. The **gate's** within-block number is computed
   independently and is trustworthy. NOT fixed (metric quality only).

**Pattern:** our bugs wasted GPU time (a silently dead loss burns hours and
produces nothing); the upstream bugs matter scientifically.

---

## 7. Stage-3 modifications (ours)

- **L_dec (writer side)** — the K latents alone must reconstruct the observation
  sentence, through a small cross-attention decoder discarded at inference.
  `slot_dropout` hides slots so no single one is the sole carrier. This is the
  designed fix for within-block redundancy.
- **L_swap (reader side)** — splice a random different-answer donor's latents in
  and require the span NLL to get worse by a margin;
  `relu(margin − (nll_donor − nll_real))` with `nll_real` detached.
- **Mod A — recentered alignment** — subtract the dataset-mean target latent so
  the cosine budget goes to the content subspace.
- `--swap_span {obs,answer,both}` added (`3cad952`), default `obs`.

### Why L_swap is supervised on OBSERVATIONS, not answers

The original justification ("teacher-forced answer NLL ~0.11 is too small to give
gradient") is **wrong** — `nll_real` is detached, so gradient flows through
`nll_donor`, which can rise however confident the model is.

But the conclusion is right for a different reason: under teacher forcing the gold
CoT precedes the answer, so the answer is already determined by context
(`ans_nll_real = 0.114`) and no latent intervention can move it. The observation
span is the only place where latents are the dominant available evidence — which
is exactly why obs gap is nonzero (0.007) and ans gap is exactly zero (−0.00002).
**Answer-level causality must come via the free-generation chain
latents → informative observation → answer**, not by supervising teacher-forced
answers.

---

## 8. Infrastructure lessons

- **Always `git pull --ff-only && sbatch` as one command.** A 6-day-queued gate job
  (14962190) crashed on a bug that had already been fixed, because the cluster
  checkout was stale. Slurm also **spools the batch script at submit time**, so
  merging after submitting does not update the launcher. All launchers now print
  `=== CODE: <hash> ===` as their first log line (`c713b8b`).
- The cluster checkout is on `main`, consuming the fork branch via merge commits
  that are never pushed, so the launchers' "BEHIND origin" warning is **spurious
  there** — trust the `CODE:` hash line.
- Hardware: `nodeai01-07` = 8×H100 80GB, `nodeai08-10` = 8×H200. 8-GPU A100 nodes
  are 40GB (too small); 80GB A100 nodes have only 2 GPUs (ZeRO-2 optimizer state
  alone is ~42GB/rank at 2 ranks).
- No C++ compiler on any default PATH → `cpu_adam` is unbuildable → Plan B:
  `ds_zero2_offload_nojit.json` + `MONET_OPTIM=adamw_torch`.
- Short inference jobs (gate, probes) should request short walltime
  (`--time=1:30:00`) so Slurm backfill runs them in minutes.

---

## 9. Current state (2026-09-01)

**Running:** stage 2 v3 — recentered, submitted with
`RECENTER_PATH=/scratch/haizhow/monet_ckpts/residual_mean.pt MARGIN=0.10
ALIGNMENT_WEIGHT=8.0 EMPHASIZE_LATENT_WEIGHT=1.0 EPOCHS=1`.

Each flag targets one diagnosed cause: recentering kills the shortcut, `MARGIN=0.10`
sits just under the true 0.121 recentered ceiling so the objective keeps demanding
more, `ALIGNMENT_WEIGHT=8.0` moves the residual term from 47× weaker than
grounding to ~3-10×, `EMPHASIZE_LATENT_WEIGHT=1.0` pins the newly-fixed flag so
nothing else shifts, `EPOCHS=1` because 2 epochs demonstrably degraded causality.

**Cancelled:** 15439209 (stage 3 on failed-gate targets with the broken alignment
axis), 15440018 (stage 2 launched without the margin/weight fixes).

**Artifacts on scratch:**
- `residual_mean.pt` — μ `(9, 3584)` over 10,000 pairs. Computed from the frozen
  teacher caches, so it does **not** need regenerating when the student retrains.
- `teacher_reps_pos/`, `teacher_reps_neg/` — 117,895 files each.
- `teacher_latents_modified/` — 124,166 stage-3 targets, **stale**: harvested from
  the failed checkpoint at the wrong image budget. Must be re-harvested.

**Sequence from here:** stage 2 v3 finishes (~10h) → gate it (1 GPU, <1h,
short walltime) → only on PASS re-harvest targets (~12h, now at the corrected
budget) → stage 3 with the fixed alignment axis → free-generation do(Z) + V*
through VLMEvalKit.

**Watch in stage 2 v3, ~1h in:** `residual_gap` should climb past 0.058 rather
than plateauing; `hinge_active_frac` should stay ≥0.8 rather than falling to 0.5;
`cross_sample_sim` should stay well below 0.867.

---

## 10. Open questions

1. Does removing the shortcut + raising the margin/weight actually lift the
   content gap above 0.02? Untested — this is what stage 2 v3 answers.
2. Will within-block similarity fall, or is the upstream slot gradient-isolation
   (§6 upstream #2) the binding constraint? If v3 passes its gate with
   within-block still ~0.9, that is the next thing to confront.
3. Does obs-level causality transfer to answer-level do(Z) in free generation?
   The baseline (guard 1.000, Δ 0.0) is direct evidence that it does not happen
   automatically. This is the project's central open question.
4. Emission rate on the final checkpoint — do(Z) is scored only over
   clean-emitting samples, so a changed emission rate changes what Δ means.
5. `gate_stage2.py` itself and the VLMEvalKit do(Z) path have **not** been
   audited. They matter at verdict time.

---

## 11. Reproducing the eval

Register the checkpoint in the **Palmetto clone** `/scratch/haizhow/VLMEvalKit/vlmeval/config.py`
(not the local Desktop copy) inside `qwen2vl_series`, using the README system
prompt. Then:

```bash
sbatch --export=ALL,MODEL=Monet-S3-ours,LATENT_SIZE=8,DUMP=1 \
  evaluation/capimagine/vlmevalkit/run_vlmeval_doz.sbatch
```

`LATENT_SIZE=8` (our K), not the released model's 16. `DUMP=1` saves individual
latents so `probe_collapse.py` can report effective rank on free-generation
latents — a stronger check than the teacher-forced gate. API keys go in
`$VLME/.env` or a shell export, never in a file that gets committed.
