"""Modified Stage-3 trainer: paper Stage-3 (latent alignment + NTP) PLUS two
accuracy-preserving levers for latent causality.

  L_dec  (writer side) -- the K latents alone must reconstruct the observation
    sentence. Breaks within-block redundancy (0.92) that pooled InfoNCE permits;
    discarded at inference, so zero accuracy cost.

  L_swap (reader side) -- REDIRECT, don't remove. Splice a RANDOM other sample's
    latents (donor) into this row and require the answer to get slightly worse:
        L_swap = relu(margin - (nll_donor - nll_real)),   nll_real detached.
    No counterfactual pairs, no VLM, no bbox -- the donor is any recent latent
    block. The MARGIN is the causality dial: small margin (~0.1-0.2 nats) + low
    weight asks the answer to depend on the latents *a little*, so accuracy is
    preserved and do(Z) rises modestly (target 2-10%). Nothing is masked, so the
    image/text accuracy pathways are untouched -- the model is only pushed to
    *also* use the latents.

Gate verdict this addresses: content is present (rank 54, retrievable) but the
LM's reader doesn't USE it (obs content gap ~0). L_swap trains the reader; L_dec
keeps the writer's slots distinct so there is something to use.
"""
import logging

import torch
import torch.nn.functional as F

from src.trainer import CustomTrainerSFT_STAGE3
from src.train.span_nll import nll_on_positions


class DonorBank:
    """Ring of recent detached latent blocks [K, d], each tagged with an answer
    key, to draw swap donors from. bsz=1 has no in-batch donor, so we keep a
    small cross-step bank. Draws are DIFFERENT-ANSWER only: a same-answer donor
    asks the model to become wrong about an answer the donor's latents still
    support (self-cancelling). No RNG -- the draw index is derived from the step."""
    def __init__(self, size: int = 64):
        self.size = size
        self.buf = []   # list of (z_cpu, answer_key)

    def push(self, z: torch.Tensor, answer_key):
        self.buf.append((z.detach().to("cpu"), answer_key))
        if len(self.buf) > self.size:
            self.buf.pop(0)

    def draw(self, step: int, shape, answer_key) -> torch.Tensor | None:
        cand = [b for (b, k) in self.buf[:-1]
                if b.shape == shape and k != answer_key]   # same shape, diff answer
        if not cand:
            return None
        return cand[step % len(cand)]


class CustomTrainerSFT_STAGE3_Decode(CustomTrainerSFT_STAGE3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decode_weight = float(getattr(self.args, 'decode_weight', 1.0))
        self.slot_dropout = int(getattr(self.args, 'slot_dropout', 2))
        # Recentered alignment (mod A): subtract the dataset-mean target latent
        # from the loaded targets so the cosine budget goes to the content
        # subspace, not the shared mean. Path set via --align_recenter_path.
        self._align_mean = None
        _amp = getattr(self.args, 'align_recenter_path', None)
        if _amp:
            try:
                self._align_mean = torch.load(_amp, map_location="cpu")["mean"].float()
                logging.info(f"recentered alignment ON: mean {tuple(self._align_mean.shape)} from {_amp}")
            except Exception as e:
                logging.warning(f"align_recenter_path load failed ({e!r}); alignment NOT recentered")
        self.swap_weight = float(getattr(self.args, 'swap_weight', 0.0))
        self.swap_margin = float(getattr(self.args, 'swap_margin', 0.15))
        self.swap_every = int(getattr(self.args, 'swap_every', 1))
        # Which span L_swap is supervised on: 'obs' (default, original design),
        # 'answer' (where do(Z) is actually measured), or 'both'.
        self.swap_span = str(getattr(self.args, 'swap_span', 'obs'))
        self._bank = DonorBank(size=int(getattr(self.args, 'swap_bank', 64)))
        self._dec_loss_cum = self._dec_gap_cum = 0.0
        self._dec_steps = 0
        self._dec_ok_total = 0   # cumulative successes (log() resets _dec_steps)
        self._swap_loss_cum = self._swap_gap_cum = 0.0
        self._swap_steps = 0
        self._swap_ok_total = 0  # cumulative successes (log() resets _swap_steps)
        self._dec_gap_donor = 0  # decode_gap comparisons that used a REAL donor
        # L_swap gap bucketed by whether the question image was VISIBLE to the
        # observation tokens this step (set by the collator under
        # --obs_image_dropout). The visible bucket is the inference-relevant
        # signal: at eval the image is always present, so only a gap that
        # survives image visibility predicts a do(Z) effect. With the mask on,
        # the gap is trivially large (latents are the only route) and teaches
        # nothing transferable.
        self._swap_gap_vis_cum = 0.0; self._swap_vis_steps = 0
        self._swap_gap_msk_cum = 0.0; self._swap_msk_steps = 0
        self._obs_masked_steps = 0
        self._log_window = 0     # compute_loss calls since the last log() flush
        self._cl_calls = 0       # total compute_loss calls (fail-loud horizon)
        self._dec_last_err = self._swap_last_err = None

    def _obs_target_ids(self, inputs):
        obs_poss = inputs.get('observation_poss', [None])[0]
        ids = inputs['student_input_ids'][0]
        if not obs_poss:
            return None
        poss = torch.tensor([p for p in obs_poss if 0 <= p < ids.numel()],
                            device=ids.device, dtype=torch.long)
        return ids.index_select(0, poss) if poss.numel() else None

    def _obs_positions(self, inputs):
        """Observation-span positions for the swap NLL. The swap is SUPERVISED on
        observations (answer NLL is ~0.11 under teacher-forcing -> no gradient);
        the answer-level causality claim is VERIFIED separately by free-gen do(Z)."""
        obs_poss = inputs.get('observation_poss', [None])[0]
        if not obs_poss:
            return None
        # NOTE: at the top of compute_loss the parent has not yet created
        # inputs['input_ids'] (it aliases student_input_ids inside its body), so
        # read the student key here.
        L = inputs['student_input_ids'][0].numel()
        return [p for p in obs_poss if 0 < p < L]

    def _answer_positions(self, inputs):
        """Labeled response positions that are NOT observation tokens (reasoning +
        final answer).

        do(Z) is an ANSWER-level measurement, so this is the span the causality
        claim actually lives on. The original design supervised L_swap on
        observations only, reasoning that teacher-forced answer NLL (~0.11) is too
        small to give gradient -- but nll_real is detached and the gradient flows
        through nll_donor, which is free to rise however confident the model is.
        A small nll_real lowers the margin TARGET; it does not remove the signal.
        """
        labels = inputs.get('student_labels', inputs.get('labels'))
        if labels is None:
            return None
        L = inputs['student_input_ids'][0].numel()
        obs = set(self._obs_positions(inputs) or [])
        lab = (labels[0] != -100).nonzero(as_tuple=False).flatten().tolist()
        return [p for p in lab if 0 < p < L and p not in obs]

    def _swap_positions(self, inputs, obs_pos):
        """Span L_swap is supervised on, per --swap_span."""
        if self.swap_span == 'answer':
            return self._answer_positions(inputs)
        if self.swap_span == 'both':
            return sorted(set((obs_pos or []) + (self._answer_positions(inputs) or [])))
        return obs_pos

    def _answer_key(self, inputs):
        """A cheap key for 'same final answer': the tail labeled token ids. Used to
        draw DIFFERENT-answer donors (same-answer donors are self-cancelling)."""
        labels = inputs.get('student_labels', inputs.get('labels'))
        ids = inputs['student_input_ids'][0]     # 'input_ids' not aliased yet at top
        if labels is None:
            return None
        lab_pos = (labels[0] != -100).nonzero(as_tuple=False).flatten()
        if lab_pos.numel() == 0:
            return None
        tail = lab_pos[-6:]                       # last few response tokens ~ the boxed answer
        return tuple(ids.index_select(0, tail).tolist())

    def _span_nll(self, model, inputs, z, positions):
        """Span NLL of this row under latents z (teacher-forced)."""
        fwd = dict(
            input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
            pixel_values=inputs['pixel_values'], image_grid_thw=inputs['image_grid_thw'],
            ce_patch_pos=inputs['ce_patch_pos'], ce_patch_vec=[z],
            labels=inputs['labels'], loss_type=['ce'], latent_mode=False, return_dict=True)
        if 'attention_mask_4d' in inputs:
            fwd['attention_mask_4d'] = inputs['attention_mask_4d']
        out = model(**fwd)
        return nll_on_positions(out.logits[0], inputs['input_ids'][0], positions)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Set by the stage-3 collator under --obs_image_dropout; popped so it
        # never reaches the model forward. None => masking not in use.
        obs_masked = inputs.pop('obs_image_masked', None)
        if obs_masked:
            self._obs_masked_steps += 1
        obs_ids = self._obs_target_ids(inputs)
        obs_pos = self._obs_positions(inputs)
        ans_key = self._answer_key(inputs)
        base = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                    num_items_in_batch=num_items_in_batch)
        loss = base[0] if isinstance(base, tuple) else base
        z = getattr(self, '_last_latents', None)

        # ---- L_dec: writer-side redundancy breaker ----
        dec = getattr(_unwrap(model), 'latent_obs_decoder', None)
        if dec is not None and z is not None and obs_ids is not None and self.decode_weight > 0:
            try:
                zt = z.unsqueeze(0) if z.dim() == 2 else z
                tgt = obs_ids.unsqueeze(0)[:, :dec.max_len]
                l_dec = dec(zt.to(next(dec.parameters()).dtype), tgt,
                            slot_dropout=self.slot_dropout)
                loss = loss + self.decode_weight * l_dec
                self._dec_loss_cum += float(l_dec.detach().item())
                # Real other-sample latents make the best comparison Z; the bank
                # is empty for the first few steps, and decode_gap falls back to
                # matched-moment noise then (bsz=1 has no in-batch alternative).
                _don = self._bank.draw(int(getattr(self.state, 'global_step', 0) or 0),
                                       zt[0].shape, ans_key)
                self._dec_gap_cum += dec.decode_gap(
                    zt.detach(), tgt, donor=None if _don is None else _don.unsqueeze(0))
                self._dec_gap_donor += (1 if _don is not None else 0)
                self._dec_steps += 1
                self._dec_ok_total += 1
            except Exception as e:
                self._dec_err = getattr(self, '_dec_err', 0) + 1
                if self._dec_err <= 5:
                    logging.warning(f"L_dec skipped: {e!r}")
                # Tolerate a sporadic bad row, but NEVER a permanently dead
                # objective: a bf16 dtype error was swallowed here for two entire
                # training jobs (15237561, 15427523) -- L_dec contributed nothing
                # and the logs showed no decode_loss key to notice it by.
                self._dec_last_err = repr(e)

        # ---- L_swap: reader-side redirect (different-answer donor) ----
        step = int(getattr(self.state, 'global_step', 0) or 0)
        swap_pos = self._swap_positions(inputs, obs_pos)
        if (self.swap_weight > 0 and z is not None and swap_pos
                and 'ce_patch_pos' in inputs and (step % max(1, self.swap_every) == 0)):
            try:
                zt = z if z.dim() == 2 else z[0]
                donor = self._bank.draw(step, zt.shape, ans_key)
                if donor is not None:
                    donor = donor.to(zt.device, zt.dtype)
                    with torch.no_grad():
                        nll_real = self._span_nll(model, inputs, zt, swap_pos)
                    nll_donor = self._span_nll(model, inputs, donor, swap_pos)
                    # push nll_donor UP toward nll_real + margin (nll_real detached)
                    l_swap = F.relu(self.swap_margin - (nll_donor - nll_real.detach()))
                    loss = loss + self.swap_weight * l_swap
                    self._swap_loss_cum += float(l_swap.detach().item())
                    _g = float((nll_donor - nll_real).detach().item())
                    self._swap_gap_cum += _g
                    self._swap_steps += 1
                    self._swap_ok_total += 1
                    if obs_masked is not None:
                        if obs_masked:
                            self._swap_gap_msk_cum += _g; self._swap_msk_steps += 1
                        else:
                            self._swap_gap_vis_cum += _g; self._swap_vis_steps += 1
                self._bank.push(zt, ans_key)
            except Exception as e:
                self._swap_err = getattr(self, '_swap_err', 0) + 1
                if self._swap_err <= 5:
                    logging.warning(f"L_swap skipped: {e!r}")
                # Same fail-loud rule as L_dec: a sporadic bad row is fine, an
                # objective that has NEVER once contributed is a dead run.
                self._swap_last_err = repr(e)
        elif self.swap_weight > 0 and z is not None:
            self._bank.push(z if z.dim() == 2 else z[0], ans_key)

        # ---- fail-loud, OUTSIDE every guard ----
        # The old checks lived inside `if dec is not None and ...` / the donor
        # branch, so a permanently-FALSE precondition (decoder not found by
        # _unwrap, latents never stashed, donor bank starved) incremented no
        # counter, raised nothing, and merely omitted the log key. That is the
        # exact signature of the two jobs L_dec was dead in. "Never attempted"
        # must be as loud as "always failed".
        self._cl_calls += 1
        self._log_window += 1
        if self._cl_calls == 200:
            for name, weight, ok, err, last in (
                    ("L_dec", self.decode_weight, self._dec_ok_total,
                     getattr(self, '_dec_err', 0), self._dec_last_err),
                    ("L_swap", self.swap_weight, self._swap_ok_total,
                     getattr(self, '_swap_err', 0), self._swap_last_err)):
                if weight > 0 and ok == 0:
                    raise RuntimeError(
                        f"{name} is ENABLED (weight={weight}) but contributed to 0 of "
                        f"{self._cl_calls} steps -- the objective is dead and this run "
                        f"cannot test it. Caught errors: {err}; last: {last}. "
                        f"(err=0 means the loss was never even attempted: check that the "
                        f"module is attached, latents are stashed, and spans are non-empty.)")
        return (loss, None) if return_outputs else loss

    def log(self, logs, start_time=None):
        """Emit every enabled objective's key UNCONDITIONALLY.

        The old guards (`if self._dec_steps > 0`) meant a dead objective produced
        no key at all, and the ABSENCE of a key was the only evidence -- which is
        exactly how L_dec stayed dead for two full jobs. Now an enabled-but-never-
        firing loss logs 0.0 next to a *_fired fraction, so it is visible in the
        first log line instead of inferable from a missing dict entry.
        """
        merged = dict(logs)
        if self.decode_weight > 0:
            merged["decode_loss"] = (round(self._dec_loss_cum / self._dec_steps, 6)
                                     if self._dec_steps > 0 else 0.0)
            merged["decode_gap"] = (round(self._dec_gap_cum / self._dec_steps, 6)
                                    if self._dec_steps > 0 else 0.0)
            # 0.0 => L_dec is enabled but contributed NOTHING this window.
            merged["decode_fired"] = round(self._dec_steps / max(self._log_window, 1), 3)
            # 'donor' is the real interchange control; 'noise' is the weak
            # fallback used when the DonorBank is empty (always so when
            # swap_weight == 0), against which any decoder scores well.
            merged["decode_gap_ctrl"] = ("donor" if self._dec_gap_donor > 0 else "noise")
            self._dec_loss_cum = self._dec_gap_cum = 0.0
            self._dec_steps = self._dec_gap_donor = 0
        if self.swap_weight > 0:
            merged["swap_loss"] = (round(self._swap_loss_cum / self._swap_steps, 6)
                                   if self._swap_steps > 0 else 0.0)
            # swap_gap = nll_donor - nll_real; this is the training-time causality
            # signal we want to see RISE toward swap_margin (do(Z) proxy).
            merged["swap_gap"] = (round(self._swap_gap_cum / self._swap_steps, 6)
                                  if self._swap_steps > 0 else 0.0)
            # 0.0 => enabled but never fired (e.g. the donor bank never yields a
            # same-shape different-answer block, which raises no exception).
            merged["swap_fired"] = round(self._swap_steps / max(self._log_window, 1), 3)
            self._swap_loss_cum = self._swap_gap_cum = 0.0
            self._swap_steps = 0
        # Bucketed by image visibility -- only present under --obs_image_dropout.
        # swap_gap_visible is THE number: causality the model keeps when it can
        # see the image, i.e. what do(Z) will measure. swap_gap_masked should be
        # large and is not evidence of anything transferable.
        if self._swap_vis_steps + self._swap_msk_steps + self._obs_masked_steps > 0:
            merged["swap_gap_visible"] = (round(self._swap_gap_vis_cum / self._swap_vis_steps, 6)
                                          if self._swap_vis_steps > 0 else 0.0)
            merged["swap_gap_masked"] = (round(self._swap_gap_msk_cum / self._swap_msk_steps, 6)
                                         if self._swap_msk_steps > 0 else 0.0)
            merged["obs_masked_frac"] = round(self._obs_masked_steps / max(self._log_window, 1), 3)
            self._swap_gap_vis_cum = self._swap_gap_msk_cum = 0.0
            self._swap_vis_steps = self._swap_msk_steps = self._obs_masked_steps = 0
        self._log_window = 0
        return super().log(merged, start_time)


def _unwrap(model):
    m = model
    for _ in range(4):
        if hasattr(m, 'latent_obs_decoder'):
            return m
        m = getattr(m, 'module', None)
        if m is None:
            break
    return model
