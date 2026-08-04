"""L_nec: necessity hinge (reader-side helper, ~0.1 × λ_swap).

    L_nec = max(0, M − (nll_ablated − nll_with.detach()))

* nll_with — gold-answer NLL from the clean forward F1. DETACHED: L_answer owns
  the clean forward; the hinge must not close the gap by making the clean case
  worse.
* nll_ablated — gold-answer NLL from F4 (donor latents at the K positions).
  Carries gradient.
* Donors: real latents from a DIFFERENT-ANSWER example. Never zeros (trivially
  detectable -> trains a "break loudly on anything unusual" gate that mimics
  dependence without reading). Never same-answer donors (self-cancelling). At
  bsz=1 the counterfactual twin's Z' is the guaranteed different-answer donor
  (y' != y is asserted at data build time); a detached cross-step donor bank adds
  variety.
* max(0, ·): once an example clears the margin it contributes zero gradient — we
  want *dependent*, not *maximally broken without latents*.
* M is calibrated ONCE from the signed baseline gap, then fixed; sweep λ only.
"""
from collections import deque
from typing import List, Optional, Tuple

import torch


def necessity_hinge(nll_with: torch.Tensor, nll_ablated: torch.Tensor,
                    margin: float) -> torch.Tensor:
    """Hinge for one sample (scalars). Detach discipline enforced by tests."""
    return torch.clamp(margin - (nll_ablated - nll_with.detach()), min=0.0)


class DonorBank:
    """Detached FIFO of (pooled-or-full) latents + their answer strings from
    recent steps. Provides different-answer donors when the twin is unavailable
    or for extra variety. All entries are detached CPU tensors — donors carry no
    gradient into their producer; the gradient path of L_nec is through the
    ablated forward's READER weights only."""

    def __init__(self, capacity: int = 64):
        self.buf: deque = deque(maxlen=capacity)

    def add(self, z: torch.Tensor, answer_text: str):
        self.buf.append((z.detach().cpu(), str(answer_text)))

    def sample_different_answer(self, answer_text: str) -> Optional[torch.Tensor]:
        """Deterministic scan (no RNG — keeps the zero-weight baseline byte-identical
        guarantee simple): most recent donor whose answer differs."""
        for z, ans in reversed(self.buf):
            if ans != str(answer_text):
                return z
        return None
