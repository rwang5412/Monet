"""L_nce: InfoNCE helper (writer-side, small weight).

Pulls each sample's pooled Z toward its own observation-span embedding; pushes it
from negatives. It makes latents *different*; L_dec makes them *about something*.
Alone, InfoNCE is satisfiable by content-free per-example fingerprints — hence the
small weight (0.1–0.3 × λ_dec).

Space: the model's OWN (frozen) input-embedding table over the observation token
ids, mean-pooled — NOT teacher hidden states (that would reintroduce the
space-mismatch problem Stage 4 deletes).

Negatives: the counterfactual twin's Z' (same image/question/prefix, differs only
by the edit — the hardest available negative, and free) plus a detached ring
buffer of recent-step pooled targets (Monet trains at bsz=1, so classic in-batch
negatives don't exist).
"""
from collections import deque
from typing import List, Optional

import torch
import torch.nn.functional as F


class NegativeBank:
    """Detached FIFO of pooled observation embeddings from recent steps."""

    def __init__(self, capacity: int = 64):
        self.buf: deque = deque(maxlen=capacity)

    def add(self, pooled_obs_emb: torch.Tensor):
        self.buf.append(pooled_obs_emb.detach().float().cpu())

    def tensor(self, device) -> Optional[torch.Tensor]:
        if not self.buf:
            return None
        return torch.stack(list(self.buf)).to(device)


def info_nce(z_pooled: torch.Tensor, obs_emb_pooled: torch.Tensor,
             extra_negatives: Optional[torch.Tensor] = None,
             temperature: float = 0.07) -> torch.Tensor:
    """InfoNCE over a [B, d] batch of pooled latents vs pooled obs embeddings.

    z_pooled:        [B, d]  mean over the K latents, projected/normalized here.
    obs_emb_pooled:  [B, d]  mean input-embedding of the obs-span tokens (frozen).
    extra_negatives: [N, d]  optional bank columns appended for every row.

    With the row+twin pairing, B=2 and each row's in-batch negative IS its twin.
    """
    zq = F.normalize(z_pooled.float(), dim=-1)
    ks = F.normalize(obs_emb_pooled.float(), dim=-1)
    logits = zq @ ks.t()                                   # [B, B]
    if extra_negatives is not None and extra_negatives.numel() > 0:
        kn = F.normalize(extra_negatives.float(), dim=-1)
        logits = torch.cat([logits, zq @ kn.t()], dim=1)   # [B, B+N]
    labels = torch.arange(zq.shape[0], device=zq.device)
    return F.cross_entropy(logits / temperature, labels)


def pool_latents(ce_patch_vec: List[torch.Tensor]) -> torch.Tensor:
    """[B][K_i, H] (list over batch) -> [B, H] mean pooling."""
    return torch.stack([v.mean(dim=0) for v in ce_patch_vec])


def pool_obs_embeddings(embed_weight: torch.Tensor, input_ids: torch.Tensor,
                        observation_poss: List[List[int]]) -> torch.Tensor:
    """Mean input-embedding of the observation-span tokens, per sample. Frozen
    table lookup — carries no gradient into the backbone."""
    out = []
    for b, poss in enumerate(observation_poss):
        assert len(poss) > 0, "sample without observation span reached L_nce"
        ids = input_ids[b, torch.as_tensor(poss, device=input_ids.device)]
        out.append(embed_weight[ids].float().mean(dim=0))
    return torch.stack(out).to(embed_weight.device)
