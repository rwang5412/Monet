"""Modified Stage-3 trainer: paper Stage-3 (recentered latent alignment + NTP)
PLUS the decode-CE writer loss L_dec.

Rationale (from the Stage-2 gate verdict + review):
  * Stage-2 latents are content-inert to the LM: presence-causal (+1.42 nats)
    but content-inert (+0.007). rank 54 / nce 0.77 bought nothing causally.
  * within_block_sim = 0.92 -- the pooled InfoNCE objective is satisfied by
    eight redundant slots, so it PERMITS the second collapse axis.
  * L_dec forces the K latents ALONE to reconstruct the observation sentence.
    A mean/redundant block cannot decode thousands of distinct sentences, so it
    is instance-specific, forces slot differentiation, and its target lives in
    text (the closest proxy for "LM-readable"). No latent/image swapping.

L_dec runs on the SAME generated latents the alignment loss sees (the latent
forward's ce_patch_vec), against the observation-span token ids. The decoder is
a small, separate module (attached to `model.latent_obs_decoder` in main.py so
its parameters land in the optimizer) and is DISCARDED at inference.

decode_gap (shuffled-Z minus real-Z decode loss) is logged every step as the
tripwire: if it drifts toward zero the decoder is writing from language priors
and L_dec is a no-op.
"""
import logging

import torch

from src.trainer import CustomTrainerSFT_STAGE3, load_offline_tensor


class CustomTrainerSFT_STAGE3_Decode(CustomTrainerSFT_STAGE3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Weight on the decode-CE writer loss; 0.0 recovers stock modified Stage 3.
        self.decode_weight = float(getattr(self.args, 'decode_weight', 1.0))
        # Optional recentering vector for the alignment target (modification A):
        # a [d] tensor of the dataset-mean latent, subtracted from both sides
        # before the cosine so the alignment budget goes to the content subspace.
        # Loaded/attached in main.py; None => stock alignment.
        self._align_recenter = getattr(self.args, '_align_recenter_vec', None)
        self._dec_loss_cum = 0.0
        self._dec_gap_cum = 0.0
        self._dec_steps = 0

    def _obs_target_ids(self, inputs):
        """Observation-span token ids [T] for this (bsz=1) row, for L_dec."""
        obs_poss = inputs.get('observation_poss', [None])[0]
        ids = inputs['student_input_ids'][0]
        if not obs_poss:
            return None
        poss = torch.tensor([p for p in obs_poss if 0 <= p < ids.numel()],
                            device=ids.device, dtype=torch.long)
        if poss.numel() == 0:
            return None
        return ids.index_select(0, poss)   # [T]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # The parent runs the latent forward + CE + alignment and returns the
        # combined loss; we re-run its body but capture the generated latents so
        # we can add L_dec. To avoid duplicating the (long) parent body, call it
        # and read the latents it stashed.
        obs_ids = self._obs_target_ids(inputs)
        base = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                    num_items_in_batch=num_items_in_batch)
        loss = base[0] if isinstance(base, tuple) else base

        # The parent stashed the latent-forward output's latents on the trainer
        # via _last_latents (added below). Guard if absent (e.g. CE-only row).
        z = getattr(self, '_last_latents', None)
        dec = getattr(unwrap(model), 'latent_obs_decoder', None)
        if dec is not None and z is not None and obs_ids is not None:
            try:
                zt = z.unsqueeze(0) if z.dim() == 2 else z          # [1,K,d]
                tgt = obs_ids.unsqueeze(0)[:, :dec.max_len]         # [1,T]
                l_dec = dec(zt.to(next(dec.parameters()).dtype), tgt)
                loss = loss + self.decode_weight * l_dec
                self._dec_loss_cum += float(l_dec.detach().item())
                self._dec_gap_cum += dec.decode_gap(zt.detach(), tgt)
                self._dec_steps += 1
            except Exception as e:
                if getattr(self, '_dec_err', 0) < 5:
                    self._dec_err = getattr(self, '_dec_err', 0) + 1
                    logging.warning(f"L_dec skipped: {e!r}")
        return (loss, None) if return_outputs else loss

    def log(self, logs, start_time=None):
        merged = dict(logs)
        if self._dec_steps > 0:
            merged["decode_loss"] = round(self._dec_loss_cum / self._dec_steps, 6)
            merged["decode_gap"] = round(self._dec_gap_cum / self._dec_steps, 6)
            self._dec_loss_cum = self._dec_gap_cum = 0.0
            self._dec_steps = 0
        return super().log(merged, start_time)


def unwrap(model):
    """Peel DeepSpeed/DDP wrappers to reach the attached decoder."""
    m = model
    for _ in range(4):
        if hasattr(m, 'latent_obs_decoder'):
            return m
        m = getattr(m, 'module', None)
        if m is None:
            break
    return model
