"""Stage-4 trainer: causal latent training (writer L_dec+L_nce, reader L_swap+L_nec).

    L = L_answer
      + λ_dec  · L_dec      (writer, primary)
      + λ_nce  · L_nce      (writer, helper)
      + λ_swap · L_swap     (reader, primary)
      + λ_nec  · L_nec      (reader, helper)

Forwards per Pool-A example (batched with its twin by CFCollator):
    F1 clean(row)            L_answer, L_dec(Z→obs), L_nce           [Z has grad]
    F2 clean(twin), no_grad  → Z' (detached; a cache would go stale)
    F3 spliced(row + Z')     L_swap: CE(obs span→obs') + CE(ans→y')
    F4 ablated(row + donor)  L_nec hinge (nll_with detached from F1)

Pool-B rows (and rows without a verified twin) run F1 only.

Stage 4 deliberately drops the Stage-2/3 cosine alignment loss: pure attraction to
targets is the degenerate objective that produced the collapse (effective rank
2.9/3584 measured on the released SFT checkpoint).

Zero-weight guarantee: with all four weights at 0 this compute_loss runs F1 alone
and returns before ANY RNG consumption or bank access — the byte-identical
baseline of arm 0.
"""
import gc
from typing import Optional

import torch
from trl import SFTTrainer

from .decode_loss import LatentObsDecoder
from .nce_loss import NegativeBank, info_nce, pool_latents, pool_obs_embeddings
from .necessity_loss import DonorBank, necessity_hinge
from .span_nll import nll_on_positions
from .swap_loss import swap_loss


class CustomTrainerSFT_STAGE4(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.exp_name = kwargs.pop('exp_name')
        self.latent_pad_id = kwargs.pop('latent_pad_id')
        aux_decoder: Optional[LatentObsDecoder] = kwargs.pop('aux_decoder', None)
        super().__init__(*args, **kwargs)
        a = self.args
        self.w_dec = float(getattr(a, 'decode_weight', 0.0))
        self.w_nce = float(getattr(a, 'nce_weight', 0.0))
        self.w_swap = float(getattr(a, 'swap_weight', 0.0))
        self.w_nec = float(getattr(a, 'necessity_weight', 0.0))
        self.margin = float(getattr(a, 'necessity_margin', 1.0))
        self.aux_decoder = aux_decoder
        if self.w_dec > 0:
            assert self.aux_decoder is not None, "decode_weight>0 needs aux_decoder"
        self.donor_bank = DonorBank(capacity=64)
        self.neg_bank = NegativeBank(capacity=64)
        # rolling logs
        self._acc = {k: 0.0 for k in
                     ("l_answer", "l_dec", "l_nce", "l_swap", "l_swap_obs",
                      "l_swap_ans", "l_nec", "nll_with", "nll_ablated", "decode_gap")}
        self._n = {k: 0 for k in self._acc}
        self._z_ring = []  # pooled Z for effective-rank logging

    # ------------------------------------------------------------ optimizer
    def create_optimizer(self):
        """Two param groups: the aux decoder (new module, full training, its own
        LR ~1e-4) and everything else trainable (LoRA adapter, base LR ~1e-5)."""
        if self.optimizer is not None:
            return self.optimizer
        opt_model = self.model
        dec_lr = float(getattr(self.args, 'decoder_lr', 1e-4))
        dec_params, rest_params = [], []
        for n, p in opt_model.named_parameters():
            if not p.requires_grad:
                continue
            (dec_params if 'aux_latent_decoder' in n else rest_params).append(p)
        groups = [{"params": rest_params, "lr": self.args.learning_rate}]
        if dec_params:
            groups.append({"params": dec_params, "lr": dec_lr})
        cls, kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        kwargs.pop("lr", None)
        self.optimizer = cls(groups, **kwargs)
        return self.optimizer

    # ------------------------------------------------------------- helpers
    def _log_add(self, key, val):
        self._acc[key] += float(val)
        self._n[key] += 1

    def _latent_positions(self, input_ids: torch.Tensor):
        return (input_ids[0] == self.latent_pad_id).nonzero(as_tuple=False).flatten().tolist()

    def _forward_ce(self, model, batch, labels, ce_pos=None, ce_vec=None,
                    want_logits=False):
        """One latent_mode=False forward over student tensors. labels=None skips
        the full-vocab fp32 CE upcast (span NLL is computed on sliced logits)."""
        inputs = dict(
            latent_mode=False,
            input_ids=batch["student_input_ids"].to(model.device),
            attention_mask=batch["student_attention_mask"].to(model.device),
            pixel_values=batch["student_pixel_values"].to(model.device),
            image_grid_thw=batch["student_image_grid_thw"].to(model.device),
            loss_type=['ce'] if labels is not None else [],
        )
        if labels is not None:
            inputs["labels"] = labels.to(model.device)
        if ce_pos is not None:
            inputs["ce_patch_pos"] = [ce_pos]
            inputs["ce_patch_vec"] = [ce_vec]
        out = model(**inputs, return_dict=True)
        return out

    def _forward_latent(self, model, batch, no_grad: bool):
        """latent_mode=True forward: autoregressively generates the K latents and
        returns (ce_patch_pos, ce_patch_vec) for this sample."""
        inputs = dict(
            latent_mode=True,
            input_ids=batch["student_input_ids"].to(model.device),
            attention_mask=batch["student_attention_mask"].to(model.device),
            pixel_values=batch["student_pixel_values"].to(model.device),
            image_grid_thw=batch["student_image_grid_thw"].to(model.device),
            loss_type=[],
            output_hidden_states=False,
        )
        model.gradient_checkpointing_disable()  # latent fwd uses use_cache
        if no_grad:
            with torch.no_grad():
                out = model(**inputs, return_dict=True)
        else:
            out = model(**inputs, return_dict=True)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        return out.ce_patch_pos[0], out.ce_patch_vec[0]

    # --------------------------------------------------------- compute_loss
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        f1 = inputs["f1"]
        f1_spans = inputs["f1_spans"]
        aux_on = (self.w_dec > 0 or self.w_nce > 0 or self.w_swap > 0 or self.w_nec > 0)

        # ---- F1: clean forward (always) -------------------------------------
        pos1, z1 = self._forward_latent(model, f1, no_grad=False)
        out1 = self._forward_ce(model, f1, labels=f1["student_labels"],
                                ce_pos=pos1, ce_vec=z1)
        l_answer = out1.loss
        self._log_add("l_answer", l_answer.item())

        # ---- arm-0 early exit: BEFORE any RNG/bank access -------------------
        if not aux_on:
            return (l_answer, None) if return_outputs else l_answer

        loss = l_answer
        ids1 = f1["student_input_ids"][0]

        # nll_with on the gold answer span (detached — L_answer owns F1)
        nll_with = nll_on_positions(out1.logits[0], ids1, f1_spans["ans_positions"]).detach()
        self._log_add("nll_with", nll_with.item())

        # ---- writer losses on Z (grad flows into the generator) -------------
        if self.w_dec > 0:
            obs_ids = torch.as_tensor(f1_spans["obs_ids"], device=z1.device)[None]
            l_dec = self.aux_decoder(z1[None], obs_ids)
            loss = loss + self.w_dec * l_dec
            self._log_add("l_dec", l_dec.item())

        is_pool_a = bool(inputs.get("is_pool_a"))
        z_prime = None
        if is_pool_a:
            # ---- F2: twin latents, detached by construction -----------------
            _, z_prime = self._forward_latent(model, inputs["f2"], no_grad=True)
            z_prime = z_prime.detach()

        if self.w_nce > 0:
            embed_w = model.get_input_embeddings().weight.detach()
            obs_emb = pool_obs_embeddings(embed_w, f1["student_input_ids"].to(embed_w.device),
                                          [f1_spans["obs_positions"]])
            zq = pool_latents([z1])
            if z_prime is not None:
                obs2 = pool_obs_embeddings(embed_w,
                                           inputs["f2"]["student_input_ids"].to(embed_w.device),
                                           [inputs["f2"]["observation_poss"][0]])
                zq = torch.cat([zq, pool_latents([z_prime]).to(zq.device)])
                obs_emb = torch.cat([obs_emb, obs2])
            l_nce = info_nce(zq, obs_emb.to(zq.device),
                             extra_negatives=self.neg_bank.tensor(zq.device))
            loss = loss + self.w_nce * l_nce
            self._log_add("l_nce", l_nce.item())
            self.neg_bank.add(obs_emb[0])

        # ---- reader losses (Pool A only) ------------------------------------
        if is_pool_a and self.w_swap > 0:
            f3, f3s = inputs["f3"], inputs["f3_spans"]
            pos3 = self._latent_positions(f3["student_input_ids"])
            assert len(pos3) == z_prime.shape[0], \
                f"F3 latent positions ({len(pos3)}) != Z' rows ({z_prime.shape[0]})"
            out3 = self._forward_ce(model, f3, labels=None, ce_pos=pos3, ce_vec=z_prime)
            l_swap, l_obs, l_ans = swap_loss(
                out3.logits[0],
                f3s["obs_positions"], f3s["obs_ids"],
                f3s["ans_positions"], f3s["ans_ids"])
            loss = loss + self.w_swap * l_swap
            self._log_add("l_swap", l_swap.item())
            self._log_add("l_swap_obs", l_obs.item())
            self._log_add("l_swap_ans", l_ans.item())
            del out3

        if is_pool_a and self.w_nec > 0:
            donor = z_prime  # twin: guaranteed different-answer donor (y' != y)
            bank_donor = self.donor_bank.sample_different_answer(inputs["answer_text"])
            if bank_donor is not None and bank_donor.shape == donor.shape:
                donor = bank_donor.to(donor.device, donor.dtype)  # variety when available
            out4 = self._forward_ce(model, f1, labels=None, ce_pos=pos1, ce_vec=donor)
            nll_ablated = nll_on_positions(out4.logits[0], ids1, f1_spans["ans_positions"])
            l_nec = necessity_hinge(nll_with, nll_ablated, self.margin)
            loss = loss + self.w_nec * l_nec
            self._log_add("l_nec", l_nec.item())
            self._log_add("nll_ablated", nll_ablated.item())
            del out4

        # ---- banks + diagnostics --------------------------------------------
        if z_prime is not None:
            self.donor_bank.add(z_prime, inputs["twin"]["y_prime"])
        self.donor_bank.add(z1.detach(), inputs["answer_text"])
        self._z_ring.append(z1.detach().float().mean(dim=0).cpu())
        if len(self._z_ring) > 256:
            self._z_ring = self._z_ring[-256:]
        if self.w_dec > 0 and self.aux_decoder is not None:
            step = int(getattr(self.state, 'global_step', 0) or 0)
            if step % 50 == 0:
                obs_ids = torch.as_tensor(f1_spans["obs_ids"], device=z1.device)[None]
                zpair = torch.stack([z1.detach(),
                                     z_prime if z_prime is not None else z1.detach().roll(1, 0)])
                gap = self.aux_decoder.decode_gap(zpair, obs_ids.repeat(2, 1))
                self._log_add("decode_gap", gap)

        step = int(getattr(self.state, 'global_step', 0) or 0)
        if step > 0 and step % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        return (loss, None) if return_outputs else loss

    # ------------------------------------------------------------------ log
    @staticmethod
    def effective_rank(x: torch.Tensor) -> float:
        xc = x - x.mean(0, keepdim=True)
        s = torch.linalg.svdvals(xc.float())
        lam = s ** 2
        if lam.sum() <= 0:
            return 0.0
        return float((lam.sum() ** 2 / (lam ** 2).sum()).item())

    def log(self, logs: dict, start_time: float | None = None):
        merged = dict(logs)
        for k in self._acc:
            if self._n[k] > 0:
                merged[k] = round(self._acc[k] / self._n[k], 6)
                self._acc[k] = 0.0
                self._n[k] = 0
        if len(self._z_ring) >= 8:
            merged["z_effective_rank"] = round(
                self.effective_rank(torch.stack(self._z_ring)), 2)
        return super().log(merged, start_time)
