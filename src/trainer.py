from trl import SFTTrainer, SFTConfig
from typing import Optional
import logging
import torch
import os, csv, torch, datetime
import gc
import numpy as np
import math
from time import time

def compute_latents_only_loss(latents, loss_for_latents):
    '''
    Compute a loss (`loss_for_latents`) that backpropagates only through the latent embeddings `latents`.
    '''
    def _flatten_tensors(x):
                # Flatten nested [list/tuple of Tensors] into a flat list of Tensors
                if isinstance(x, (list, tuple)):
                    out = []
                    for y in x:
                        out.extend(_flatten_tensors(y))
                    return out
                return [x]

    ce_vec_list = _flatten_tensors(latents)
    grads = torch.autograd.grad(
        outputs=loss_for_latents,
        inputs=ce_vec_list,
        retain_graph=True,   # we won't reuse the 3rd graph
        create_graph=False,   # stop higher-order graph
        allow_unused=True     # in case some ce vectors are not used
    )

    # Replace None with zeros for unused elements
    safe_grads = []
    for v, g in zip(ce_vec_list, grads):
        if g is None:
            # Create a zero tensor on the same device/dtype/shape
            g = torch.zeros_like(v)
        safe_grads.append(g.detach())  # detach to stop any 3rd-forward param pathg

    proxy_loss = torch.stack([(v * g).sum() for v, g in zip(ce_vec_list, safe_grads)]).sum()
    return proxy_loss

def load_offline_tensor(tensor_dir, batch_metadata, alignment_layer="all_layers", rep_type="rep", align_poss="obs"):
    '''
    Load precomputed teacher representations (observation tokens for the alignment in SFT stage 2 or the latent embeddings for SFT stage 3)
    '''
    teacher_reps = None
    latents_list = []
    for metadata in batch_metadata:
        dataset_name = metadata['dataset_name']
        sample_id = metadata['sample_id']
        metadata_info = f"{alignment_layer}_{dataset_name}_{sample_id}"
        if align_poss == 'obs':
            metadata_str = f"{rep_type}_{metadata_info}.pt"
        elif align_poss == 'latent_end':
            metadata_str = f"{rep_type}_latent_end_{metadata_info}.pt"
        path = os.path.join(tensor_dir, metadata_str)
        if not os.path.isfile(path):
            latents_list = []
            raise RuntimeError(f"Missing teacher latent file: {path}")
        data = torch.load(path, map_location='cpu')
        latents_list.append(data['latent'].detach())
    if batch_metadata is not None and len(latents_list) == len(batch_metadata):
        teacher_reps = latents_list
    return teacher_reps


class CustomTrainerSFT_STAGE1(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.exp_name =kwargs.pop('exp_name')
        # accept processing_class (preferred) and fall back to tokenizer for backward compat
        if 'processing_class' not in kwargs and 'tokenizer' in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')
        super().__init__(*args, **kwargs)
        self.observation_token_acc = 0.
        self.observation_token_acc_step = 0
        self.teacher_ce_cum = 0.0        # cumulative student CE loss
        self.teacher_ce_steps = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        inputs['latent_mode'] = False
        inputs['input_ids'] = inputs['teacher_input_ids']
        inputs['attention_mask'] = inputs['teacher_attention_mask']
        inputs['pixel_values'] = inputs['teacher_pixel_values']
        inputs['image_grid_thw'] = inputs['teacher_image_grid_thw']
        inputs['labels'] = inputs['teacher_labels']
        inputs['ce_emphasize_poss'] = inputs['teacher_observation_poss']
        # Dynamic warmup factor passed to model.forward
        inputs['ce_emphasize_factor'] = self.args.ce_emphasize_factor
        inputs['loss_type'] = ['ce']
        inputs['compute_emphasize_acc'] = True
        (teacher_ce_loss, teacher_outputs) = super().compute_loss(
                model, 
                inputs,
                return_outputs=True, num_items_in_batch=num_items_in_batch
            )

        self.teacher_ce_cum += teacher_ce_loss.item()
        self.teacher_ce_steps += 1

        if getattr(teacher_outputs, 'mean_emphasize_acc', None) is not None:
            self.observation_token_acc += getattr(teacher_outputs, 'mean_emphasize_acc')
            self.observation_token_acc_step += 1

        del teacher_outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        return (teacher_ce_loss, None) if return_outputs else teacher_ce_loss

    def on_epoch_end(self):
        return super().on_epoch_end()

    def log(self, logs: dict, start_time: float | None = None):
        # Merge our rolling averages into the standard logs once per logging call
        merged = dict(logs)
        if self.teacher_ce_steps > 0:
            merged["student_ce_loss"] = round(self.teacher_ce_cum / max(1, self.teacher_ce_steps), 6)
            self.teacher_ce_cum = 0.0
            self.teacher_ce_steps = 0
        if self.observation_token_acc_step > 0:
            merged["observation_token_acc"] = round(self.observation_token_acc/ max(1, self.observation_token_acc_step), 6)
            self.observation_token_acc = 0.
            self.observation_token_acc_step = 0

        # Call parent to keep default behavior (console/TB/W&B/etc.)
        return super().log(merged, start_time)



class CustomTrainerSFT_STAGE2(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.exp_name = kwargs.pop('exp_name')
        # accept processing_class (preferred) and fall back to tokenizer for backward compat
        if 'processing_class' not in kwargs and 'tokenizer' in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')
        super().__init__(*args, **kwargs)

        self.ce_emphasize_factor = self.args.ce_emphasize_factor
        self.teacher_ce_loss_cum = 0.0        # cumulative teacher CE loss
        self.teacher_ce_loss_steps = 0
        self.observation_token_acc = 0.
        self.observation_token_acc_step = 0
        self.alignment_loss_cum = 0.
        self.alignment_loss_steps = 0
        # Stage-2 changes: residual objective (needs the no-aux teacher cache) and
        # the latent grounding InfoNCE (module attached to the model in main.py).
        self.teacher_reps_neg_dir = getattr(self.args, 'teacher_reps_neg_dir', None)
        # Margin recentering (anti shared-direction drift). mu = E[h_pos - h_neg]
        # per kept layer (src.compute_residual_mean). Subtracting it from h_pos
        # makes h_pos' - h_neg zero-mean over the dataset, so shifting every obs
        # state along mu earns no margin -- only the sample-specific residual pays.
        # Pilot 14946764: cross_sample_sim 0.5 -> 0.81 in 500 steps without this.
        self.residual_mu = None
        _rp = getattr(self.args, 'residual_recenter_path', None)
        if _rp:
            _d = torch.load(_rp, map_location='cpu')
            self.residual_mu = _d['mean'].float()          # [num_kept_layer, dim]
            logging.info(f"residual recentering ON: mu {tuple(self.residual_mu.shape)} "
                         f"from {_rp} (n_files={_d.get('n_files')})")
        self.grounding_weight = float(getattr(self.args, 'grounding_weight', 0.0))
        self.grounding_loss_cum = 0.
        self.grounding_loss_steps = 0
        # --keep_layers'd caches need the student stack sliced to the same layers
        _ali = getattr(self.args, 'alignment_layer_indices', None)
        self.alignment_layer_indices = ([int(x) for x in _ali.split(',')] if _ali else None)
        # instrumentation accumulators (spec §7)
        self._s2_stats = {k: [0.0, 0] for k in
                          ("residual_gap", "hinge_active_frac", "nce_top1",
                           "within_block_sim", "cross_sample_sim")}
        self._z_ring = []  # recent pooled latents for cross-sample similarity

    def _recenter(self, h_pos):
        """h_pos - mu for a cached [num_kept_layer, T_obs, dim] teacher tensor."""
        mu = self.residual_mu
        assert h_pos.dim() == 3 and h_pos.shape[0] == mu.shape[0], \
            f"recenter layer mismatch: cache {tuple(h_pos.shape)} vs mu {tuple(mu.shape)}"
        return (h_pos.float() - mu[:, None, :]).to(h_pos.dtype)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        # ------------------------------------------------------------------
        # Latent forward to get ce_patch_pos (positions of latent embeddings) and ce_patch_vec (latent embeddings).
        # Multiple forward is needed since we need to autoregressively generate latents.
        # ------------------------------------------------------------------
        inputs['latent_mode'] = True
        inputs['loss_type'] = []
        model.gradient_checkpointing_disable() # since we set use_cache=True in latent forward, we must disable grad checkpointing
        outputs = model(**inputs, return_dict=True, output_hidden_states=False)

        # ------------------------------------------------------------------
        # Insert the collected latent embeddings into the latent positions, and forward once.
        # ------------------------------------------------------------------
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        inputs['latent_mode'] = False
        inputs['ce_patch_pos'] = outputs.ce_patch_pos
        inputs['ce_patch_vec'] = outputs.ce_patch_vec
        inputs['ce_emphasize_poss'] = inputs['observation_poss']
        inputs['ce_emphasize_factor'] = self.ce_emphasize_factor
        inputs['loss_type'] = ['ce']
        if self.args.alignment_weight != 0:
            inputs['loss_type'].append('alignment')

        inputs['compute_emphasize_acc'] = True
        # Ensure training forward does NOT request attentions (prevents checkpoint recompute mismatch)
        inputs.pop('output_attentions', None)
        inputs.pop('attn_analysis', None)

        if self.args.alignment_weight != 0:
            # Load precomputed teacher representations of observation tokens for
            # the alignment loss. ~0.6% of rows failed the teacher collate and
            # have no cache file; those samples run CE-only (alignment skipped)
            # rather than crashing a multi-day run.
            try:
                teacher_reps = load_offline_tensor(self.args.teacher_reps_dir, batch_metadata=inputs['metadata'],
                alignment_layer=self.args.alignment_layer)
                # Change 1 (residual objective): also load the NO-aux-image teacher
                # cache; the modeling alignment block then computes the margin form
                # (closer to with-image teacher than to without-image teacher)
                # instead of the absolute cosine. Same filenames, separate dir.
                teacher_reps_neg = None
                if self.teacher_reps_neg_dir:
                    teacher_reps_neg = load_offline_tensor(self.teacher_reps_neg_dir, batch_metadata=inputs['metadata'],
                    alignment_layer=self.args.alignment_layer)
                    if self.residual_mu is not None:
                        # recentered positive teacher: h_pos - mu (mu broadcast over
                        # the obs-token axis of [num_kept_layer, T_obs, dim]).
                        # residual_gap / hinge stats are then in recentered terms.
                        teacher_reps = [self._recenter(t) for t in teacher_reps]
            except RuntimeError as e:
                self._missing_reps = getattr(self, '_missing_reps', 0) + 1
                if self._missing_reps <= 5 or self._missing_reps % 100 == 0:
                    logging.warning(f"teacher reps missing (#{self._missing_reps}); CE-only step. {e}")
                teacher_reps = None
                # drop the alignment objective for this step or the modeling
                # block would dereference the absent teacher tensors
                inputs['loss_type'] = ['ce']

            if teacher_reps is not None:
                inputs['alignment_poss'] = inputs['observation_poss']
                inputs['teacher_hidden_states_for_alignment'] = teacher_reps
                if teacher_reps_neg is not None:
                    inputs['teacher_hidden_states_for_alignment_neg'] = teacher_reps_neg
                    inputs['obs_residual_margin'] = float(getattr(self.args, 'obs_residual_margin', 0.2))
                if self.alignment_layer_indices is not None:
                    inputs['alignment_layer_indices'] = self.alignment_layer_indices

        teacher_ce_loss, teacher_output = super().compute_loss(
                model,
                inputs,
                return_outputs=True, num_items_in_batch=num_items_in_batch
            )

        alignment_loss = teacher_output.loss_dict.get('alignment', torch.tensor(0.0))

        # Change 2 (latent grounding InfoNCE). Two calls: the writer-routed one
        # feeds the latent-only surrogate (gradient reaches model params ONLY
        # through the generated latents); the detached one trains the projector,
        # which the surrogate cannot update (it re-injects gradients at the
        # latents, upstream of the projector's own parameters).
        grounding_writer = None
        if self.grounding_weight != 0.0:
            m = model
            while hasattr(m, 'module'):
                m = m.module
            grounding_mod = getattr(m, 'latent_grounding', None)
            assert grounding_mod is not None, \
                "grounding_weight != 0 but model has no latent_grounding module (attach it in main.py)"
            try:
                aux_feats = getattr(outputs, 'aux_image_feats', None) or [None] * len(outputs.ce_patch_vec)
                grounding_writer = grounding_mod(outputs.ce_patch_vec, aux_feats, enqueue=True)
                grounding_proj = grounding_mod([z.detach() for z in outputs.ce_patch_vec],
                                               aux_feats, enqueue=False)
                if not grounding_writer.requires_grad:  # all samples skipped inside the module
                    grounding_writer = None
            except Exception as e:
                # A weird sample (empty obs span / no latents) must not kill a
                # multi-day run; skip the term this step, loudly but rate-limited.
                self._grounding_skips = getattr(self, '_grounding_skips', 0) + 1
                if self._grounding_skips <= 5 or self._grounding_skips % 100 == 0:
                    logging.warning(f"grounding skipped (#{self._grounding_skips}): {e!r}")
                grounding_writer = None
            if grounding_writer is not None:
                self.grounding_loss_cum += float(grounding_writer.detach().item())
                self.grounding_loss_steps += 1
            gs = getattr(grounding_mod, 'last_stats', None)
            if gs:
                for k in ("nce_top1", "within_block_sim"):
                    if gs.get(k) is not None:
                        self._s2_stats[k][0] += gs[k]
                        self._s2_stats[k][1] += 1

        # instrumentation: residual stats from the CE forward; cross-sample latent sim
        ld = getattr(teacher_output, 'loss_dict', None) or {}
        for k in ("residual_gap", "hinge_active_frac"):
            if k in ld:
                self._s2_stats[k][0] += float(ld[k])
                self._s2_stats[k][1] += 1
        with torch.no_grad():
            for z_b in outputs.ce_patch_vec:
                if z_b is not None and z_b.numel():
                    self._z_ring.append(torch.nn.functional.normalize(
                        z_b.detach().float().mean(dim=0), dim=-1).cpu())
            self._z_ring = self._z_ring[-64:]
            if len(self._z_ring) >= 8:
                Z = torch.stack(self._z_ring)
                C = Z @ Z.T
                n = C.shape[0]
                self._s2_stats["cross_sample_sim"][0] += float((C.sum() - n) / (n * (n - 1)))
                self._s2_stats["cross_sample_sim"][1] += 1

        # Latent-routed auxiliary total (alignment/residual + grounding), then the
        # latent-only backprop surrogate: stop_grad(dL/d_latent)^T . latent.
        #
        # DeepSpeed compat: newer DeepSpeed arms a post-backward epilogue that
        # fires on ANY backward -- the surrogate's inner torch.autograd.grad
        # (activation-only, no param grads) then crashes ZeRO-2's reducer
        # (IndexError on empty buckets). On first failure we permanently fall
        # back to DIRECT backprop of the aux losses. This is safe in OUR setup
        # (unlike original Monet's 46% ablation): the attention masks make the
        # latents the ONLY information route into the obs positions, so the
        # shortcut the surrogate guarded against is closed architecturally.
        latent_routed = self.args.alignment_weight * alignment_loss
        if grounding_writer is not None:
            latent_routed = latent_routed + self.grounding_weight * grounding_writer
        has_latent_routed = (isinstance(latent_routed, torch.Tensor) and latent_routed.requires_grad
                             and float(latent_routed.detach()) != 0.0)
        if not hasattr(self, '_latent_only_mode'):
            forced = os.environ.get('MONET_LATENT_ONLY_MODE')
            if forced in ('surrogate', 'direct'):
                self._latent_only_mode = forced
            else:
                # deterministic choice: under DeepSpeed the surrogate's inner
                # autograd.grad crashes the ZeRO reducer, so default to direct.
                self._latent_only_mode = ('direct' if getattr(self, 'is_deepspeed_enabled', False)
                                          else 'surrogate')
            logging.warning(f"latent-only BP mode: {self._latent_only_mode} "
                            f"(masks preserve latent-only information flow in direct mode)")
        if (self.args.emphasize_latent_weight != 0.0 and has_latent_routed
                and self._latent_only_mode == 'surrogate'):
            try:
                latent_only_loss = compute_latents_only_loss(outputs.ce_patch_vec, latent_routed)
                loss = self.args.emphasize_latent_weight * latent_only_loss + teacher_ce_loss
                if grounding_writer is not None:
                    loss = loss + self.grounding_weight * grounding_proj  # projector path (latents detached)
            except (IndexError, RuntimeError) as e:
                logging.warning(
                    f"latent-only surrogate incompatible with this DeepSpeed ({e!r}); "
                    f"switching PERMANENTLY to direct backprop of the aux losses. "
                    f"Latent-only information flow is preserved by the attention masks.")
                self._latent_only_mode = 'direct'
        if self._latent_only_mode == 'direct' or not (self.args.emphasize_latent_weight != 0.0 and has_latent_routed):
            # direct mode: one backprop through everything; grounding_writer's
            # graph already includes the projector, so grounding_proj is NOT
            # added (it would double-count the projector gradient).
            loss = teacher_ce_loss + latent_routed

        if getattr(teacher_output, 'mean_emphasize_acc', None) is not None:
            self.observation_token_acc += getattr(teacher_output, 'mean_emphasize_acc')
            self.observation_token_acc_step += 1

        self.teacher_ce_loss_cum += teacher_ce_loss.item()
        self.teacher_ce_loss_steps += 1
        self.alignment_loss_cum += alignment_loss.item()
        self.alignment_loss_steps += 1

        # Light-touch cleanup without forcing GPU sync every step
        #del teacher_outputs
        step = int(getattr(self.state, 'global_step', 0) or 0)
        if step % 50 == 0:
            try:
                gc.collect()
                # Avoid calling empty_cache() each step
                torch.cuda.empty_cache()
            except Exception:
                pass

        return (loss, None) if return_outputs else loss


    def on_epoch_end(self):
        return super().on_epoch_end()

    def log(self, logs: dict, start_time: float | None = None):
        # Merge our rolling averages into the standard logs once per logging call
        merged = dict(logs)
        if self.teacher_ce_loss_cum > 0:
            merged["teacher_ce_loss"] = round(self.teacher_ce_loss_cum / max(1, self.teacher_ce_loss_steps), 6)
            self.teacher_ce_loss_cum = 0.0
            self.teacher_ce_loss_steps = 0
        if self.alignment_loss_cum > 0:
            merged[f'alignment_loss'] = round(self.alignment_loss_cum / max(1, self.alignment_loss_steps), 6)
            self.alignment_loss_cum = 0.0
            self.alignment_loss_steps = 0
        if self.grounding_loss_steps > 0:
            merged["grounding_loss"] = round(self.grounding_loss_cum / max(1, self.grounding_loss_steps), 6)
            self.grounding_loss_cum = 0.
            self.grounding_loss_steps = 0
        for k, (s, c) in self._s2_stats.items():
            if c > 0:
                merged[k] = round(s / c, 4)
                self._s2_stats[k] = [0.0, 0]
        if self.observation_token_acc_step > 0:
            merged["observation_token_acc"] = round(self.observation_token_acc/ max(1, self.observation_token_acc_step), 6)
            self.observation_token_acc = 0.
            self.observation_token_acc_step = 0


        # Call parent to keep default behavior (console/TB/W&B/etc.)
        return super().log(merged, start_time)

class CustomTrainerSFT_STAGE3(SFTTrainer):
    def __init__(self, *args, **kwargs): 
        self.exp_name =kwargs.pop('exp_name')
        super().__init__(*args, **kwargs)
        self.alignment_weight = self.args.alignment_weight
        self.ce_emphasize_factor: float = float(getattr(self.args, 'ce_emphasize_factor', 1.0))
        # Where to read precomputed teacher latents
        self.teacher_latent_dir = getattr(self.args, 'teacher_latent_dir', None)
        if not self.teacher_latent_dir:
            raise ValueError("teacher_latent_dir must be specified for SFT Stage 3")

        self.observation_token_acc = 0.
        self.observation_token_acc_step = 0
        self.al_loss_cum = 0.0       # cumulative alignment loss since last log
        self.al_steps = 0            # number of micro-steps accumulated
        self.student_ce_loss_cum = 0.0        # cumulative student CE loss
        self.student_ce_loss_steps = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss for SFT stage3 with optional cached teacher latents.
        """
        # Load precomputed teacher latents. Rows whose harvest failed (collate
        # edge cases; ~0.6% in stage 2) have no target file -- run those CE-only
        # instead of crashing a multi-hour run (same fallback as stage 2).
        try:
            teacher_latents = load_offline_tensor(self.teacher_latent_dir, batch_metadata=inputs['metadata'], alignment_layer=self.args.alignment_layer, rep_type="latent")
        except RuntimeError as e:
            self._missing_targets = getattr(self, '_missing_targets', 0) + 1
            if self._missing_targets <= 5 or self._missing_targets % 100 == 0:
                logging.warning(f"stage-3 target latents missing (#{self._missing_targets}); CE-only step. {e}")
            teacher_latents = None

        # Recentered alignment (mod A): subtract the dataset-mean target latent so
        # the cosine budget goes to the content subspace, not the shared mean.
        # _align_mean ([L,d] or [d]) is set by the decode subclass; None => stock.
        _am = getattr(self, '_align_mean', None)
        if _am is not None and teacher_latents is not None:
            rc = []
            for t in teacher_latents:
                m = _am.to(t.device, t.dtype)
                if t.dim() == 3 and m.dim() == 2:      # [L,K,d] - [L,1,d]
                    m = m.unsqueeze(1)
                elif t.dim() == 2 and m.dim() == 2:    # [K,d] - [d]
                    m = m.mean(0)
                rc.append(t - m)
            teacher_latents = rc

        # ------------------------------------------------------------------
        # Latent forward to get ce_patch_pos (positions of latent embeddings) and ce_patch_vec (latent embeddings)
        # ------------------------------------------------------------------
        inputs['latent_mode'] = True
        inputs['input_ids'] = inputs['student_input_ids']
        inputs['attention_mask'] = inputs['student_attention_mask']
        inputs['pixel_values'] = inputs['student_pixel_values']
        inputs['image_grid_thw'] = inputs['student_image_grid_thw']
        if 'labels' in inputs:
            inputs.pop('labels')
        inputs['alignment_poss'] = inputs['student_alignment_poss']
        inputs['teacher_hidden_states_for_alignment'] = teacher_latents
        model.gradient_checkpointing_disable() # since we set use_cache=True in latent forward, we must disable grad checkpointing
        inputs['loss_type'] = []
        inputs['output_hidden_states'] = False
        student_outputs_latent = model(**inputs)
        # Stash the generated latents (in-graph, grad flows to the params that
        # produced them) so a subclass can add a writer loss (L_dec) on them
        # before backward. bsz=1 -> single [K, d] tensor.
        try:
            cpv = student_outputs_latent.ce_patch_vec
            self._last_latents = cpv[0] if isinstance(cpv, (list, tuple)) else cpv
        except Exception:
            self._last_latents = None


        # Student CE forward
        inputs['latent_mode'] = False
        inputs['labels'] = inputs['student_labels']
        inputs['ce_patch_pos'] = student_outputs_latent.ce_patch_pos
        inputs['ce_patch_vec'] = student_outputs_latent.ce_patch_vec
        inputs['ce_emphasize_factor'] = self.ce_emphasize_factor
        inputs['ce_emphasize_poss'] = inputs['observation_poss']
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        inputs['loss_type'] = ['ce', 'alignment'] if teacher_latents is not None else ['ce']
        inputs['compute_emphasize_acc'] = True
        if 'student_attention_mask_4d' in inputs:
            inputs['attention_mask_4d'] = inputs.pop('student_attention_mask_4d')
        (student_ce_loss, student_outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if getattr(student_outputs, 'mean_emphasize_acc', None) is not None:
            self.observation_token_acc += getattr(student_outputs, 'mean_emphasize_acc')
            self.observation_token_acc_step += 1
        alignment_loss = student_outputs.loss_dict['alignment']
        loss = student_ce_loss + self.alignment_weight * alignment_loss
        outputs_student_loss = student_ce_loss.item()

        del student_outputs
        step = int(getattr(self.state, 'global_step', 0) or 0)
        if step > 0 and (step % 20 == 0):
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass

        # Logging
        self.al_loss_cum += float(alignment_loss.detach().item())
        self.al_steps += 1
        self.student_ce_loss_cum += outputs_student_loss
        self.student_ce_loss_steps += 1

        return (loss, None) if return_outputs else loss
    
    def log(self, logs: dict, start_time: float | None = None):
        # Merge our rolling averages into the standard logs once per logging call
        merged = dict(logs)
        if self.al_steps > 0:
            merged["alignment_loss"] = round(self.al_loss_cum / max(1, self.al_steps), 6)
            self.al_loss_cum = 0.0
            self.al_steps = 0
        if self.student_ce_loss_steps > 0:
            merged["student_ce_loss"] = round(self.student_ce_loss_cum / max(1, self.student_ce_loss_steps), 6)
            self.student_ce_loss_cum = 0.0
            self.student_ce_loss_steps = 0
        if self.observation_token_acc_step > 0:
            merged["observation_token_acc"] = round(self.observation_token_acc/ max(1, self.observation_token_acc_step), 6)
            self.observation_token_acc = 0.
            self.observation_token_acc_step = 0

        # Call parent to keep default behavior (console/TB/W&B/etc.)
        return super().log(merged, start_time)

