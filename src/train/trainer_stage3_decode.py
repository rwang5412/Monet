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
    """Ring of recent detached latent blocks [K, d] to draw swap donors from.
    bsz=1 has no in-batch donor, so we keep a small cross-step bank (no RNG:
    donor index is derived from the global step, so runs are reproducible)."""
    def __init__(self, size: int = 64):
        self.size = size
        self.buf = []

    def push(self, z: torch.Tensor):
        self.buf.append(z.detach().to("cpu"))
        if len(self.buf) > self.size:
            self.buf.pop(0)

    def draw(self, step: int, shape) -> torch.Tensor | None:
        # need at least one prior block of the SAME shape (same K, d)
        cand = [b for b in self.buf[:-1] if b.shape == shape]
        if not cand:
            return None
        return cand[step % len(cand)]


class CustomTrainerSFT_STAGE3_Decode(CustomTrainerSFT_STAGE3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decode_weight = float(getattr(self.args, 'decode_weight', 1.0))
        self.swap_weight = float(getattr(self.args, 'swap_weight', 0.0))
        self.swap_margin = float(getattr(self.args, 'swap_margin', 0.15))
        self.swap_every = int(getattr(self.args, 'swap_every', 1))
        self._bank = DonorBank(size=int(getattr(self.args, 'swap_bank', 64)))
        self._dec_loss_cum = self._dec_gap_cum = 0.0
        self._dec_steps = 0
        self._swap_loss_cum = self._swap_gap_cum = 0.0
        self._swap_steps = 0

    def _obs_target_ids(self, inputs):
        obs_poss = inputs.get('observation_poss', [None])[0]
        ids = inputs['student_input_ids'][0]
        if not obs_poss:
            return None
        poss = torch.tensor([p for p in obs_poss if 0 <= p < ids.numel()],
                            device=ids.device, dtype=torch.long)
        return ids.index_select(0, poss) if poss.numel() else None

    def _answer_positions(self, inputs):
        """Labeled response positions (obs + reasoning + answer) for the swap NLL."""
        labels = inputs.get('labels', inputs.get('student_labels'))
        if labels is None:
            return None
        lab = labels[0]
        return (lab != -100).nonzero(as_tuple=False).flatten().tolist()

    def _span_nll(self, model, inputs, z, positions):
        """Answer-span NLL of this row under latents z (teacher-forced)."""
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
        obs_ids = self._obs_target_ids(inputs)
        ans_pos = self._answer_positions(inputs)
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
                l_dec = dec(zt.to(next(dec.parameters()).dtype), tgt)
                loss = loss + self.decode_weight * l_dec
                self._dec_loss_cum += float(l_dec.detach().item())
                self._dec_gap_cum += dec.decode_gap(zt.detach(), tgt)
                self._dec_steps += 1
            except Exception as e:
                if getattr(self, '_dec_err', 0) < 5:
                    self._dec_err = getattr(self, '_dec_err', 0) + 1
                    logging.warning(f"L_dec skipped: {e!r}")

        # ---- L_swap: reader-side redirect (random donor, accuracy-preserving) ----
        step = int(getattr(self.state, 'global_step', 0) or 0)
        if (self.swap_weight > 0 and z is not None and ans_pos
                and 'ce_patch_pos' in inputs and (step % max(1, self.swap_every) == 0)):
            try:
                zt = z if z.dim() == 2 else z[0]
                donor = self._bank.draw(step, zt.shape)
                if donor is not None:
                    donor = donor.to(zt.device, zt.dtype)
                    with torch.no_grad():
                        nll_real = self._span_nll(model, inputs, zt, ans_pos)
                    nll_donor = self._span_nll(model, inputs, donor, ans_pos)
                    # push nll_donor UP toward nll_real + margin (nll_real detached)
                    l_swap = F.relu(self.swap_margin - (nll_donor - nll_real.detach()))
                    loss = loss + self.swap_weight * l_swap
                    self._swap_loss_cum += float(l_swap.detach().item())
                    self._swap_gap_cum += float((nll_donor - nll_real).detach().item())
                    self._swap_steps += 1
                self._bank.push(zt)
            except Exception as e:
                if getattr(self, '_swap_err', 0) < 5:
                    self._swap_err = getattr(self, '_swap_err', 0) + 1
                    logging.warning(f"L_swap skipped: {e!r}")
        elif self.swap_weight > 0 and z is not None:
            self._bank.push(z if z.dim() == 2 else z[0])

        return (loss, None) if return_outputs else loss

    def log(self, logs, start_time=None):
        merged = dict(logs)
        if self._dec_steps > 0:
            merged["decode_loss"] = round(self._dec_loss_cum / self._dec_steps, 6)
            merged["decode_gap"] = round(self._dec_gap_cum / self._dec_steps, 6)
            self._dec_loss_cum = self._dec_gap_cum = 0.0
            self._dec_steps = 0
        if self._swap_steps > 0:
            merged["swap_loss"] = round(self._swap_loss_cum / self._swap_steps, 6)
            # swap_gap = nll_donor - nll_real; this is the training-time causality
            # signal we want to see RISE toward swap_margin (do(Z) proxy).
            merged["swap_gap"] = round(self._swap_gap_cum / self._swap_steps, 6)
            self._swap_loss_cum = self._swap_gap_cum = 0.0
            self._swap_steps = 0
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
