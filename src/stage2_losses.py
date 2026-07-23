"""Stage-2 Change 2: direct contrastive grounding of the latents.

Nothing in original Stage 2 says what a latent should BE, only what it should
cause -- a single shared vector satisfies a handful of obs-token constraints,
and at batch size 1 no term anywhere penalizes latent(A) ~= latent(B). This
InfoNCE ties each latent block to its OWN aux image's post-merger visual tokens
against a memory queue of other samples' aux features. The queue is not
optional: with bsz=1 + grad accumulation there are no in-batch negatives, and
the queue is what makes cross-sample collapse costly.

The projector sits on the student side only: latents are last-layer OUTPUT
states, post-merger visual tokens live in the LLM's INPUT-embedding space; the
small MLP bridges the two spaces (don't try to match raw vectors).

Note the residual objective (Change 1) lives in the modeling file
(obs_residual_loss in monet_qwen_model/modeling_qwen2_5_vl_monet.py), next to
the alignment block that consumes it.
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentGroundingLoss(nn.Module):
    """InfoNCE: latent block must match its own aux image against a queue of others."""

    def __init__(self, d_model: int, queue_size: int = 4096, temp: float = 0.07):
        super().__init__()
        self.proj = nn.Sequential(          # student side only
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.temp = temp
        self.register_buffer("queue", F.normalize(torch.randn(queue_size, d_model), dim=-1))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("filled", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _enqueue(self, v: torch.Tensor):
        n = v.shape[0]
        p = int(self.ptr)
        q = self.queue.shape[0]
        for i in range(n):  # ring write; n is tiny (bsz=1)
            self.queue[(p + i) % q] = v[i]
        self.ptr[0] = (p + n) % q
        self.filled[0] = min(int(self.filled) + n, q)

    def forward(self, latents: List[torch.Tensor], aux_feats: List[Optional[torch.Tensor]],
                enqueue: bool = True) -> torch.Tensor:
        """
        latents:   list over batch of [K_b, D] student latent blocks (last layer)
        aux_feats: list over batch of [M_b, D] post-merger aux-image tokens
                   (detached upstream; entries may be None -> sample skipped)
        enqueue:   write this step's targets into the queue (disable on the
                   detached projector-training call so each target enters once)
        """
        zs, vs = [], []
        for z_b, v_b in zip(latents, aux_feats):
            if v_b is None or z_b is None or z_b.numel() == 0:
                continue
            zs.append(z_b.mean(dim=0))
            vs.append(v_b.float().mean(dim=0))
        if not zs:
            dev = self.queue.device
            return torch.zeros((), device=dev)

        z = F.normalize(self.proj(torch.stack(zs).float()), dim=-1)        # (B, D)
        v = F.normalize(torch.stack(vs).to(z.device), dim=-1).detach()     # (B, D)

        pos = (z * v).sum(-1, keepdim=True)                                # (B, 1)
        neg = z @ self.queue[: max(int(self.filled), 1)].to(z.device).T    # (B, Q_filled)
        logits = torch.cat([pos, neg], dim=1) / self.temp
        loss = F.cross_entropy(
            logits, torch.zeros(z.shape[0], dtype=torch.long, device=z.device))

        # Instrumentation (spec §7): InfoNCE top-1 (should climb well above
        # 1/(Q+1)) and within-block latent similarity (K slots should decorrelate).
        with torch.no_grad():
            top1 = (logits.argmax(dim=1) == 0).float().mean().item()
            wb = []
            for z_b in latents:
                if z_b is not None and z_b.shape[0] > 1:
                    zn = F.normalize(z_b.float(), dim=-1)
                    c = zn @ zn.T
                    k = c.shape[0]
                    wb.append(((c.sum() - k) / (k * (k - 1))).item())
            self.last_stats = {"nce_top1": top1,
                               "within_block_sim": sum(wb) / len(wb) if wb else None}

        if enqueue:
            self._enqueue(v)
        return loss
