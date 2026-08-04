"""L_swap: directed swap supervision (reader-side, primary).

Forward F3 runs the RE-TOKENIZED spliced sequence — the row's conversation text
with the observation span replaced by obs' and the boxed answer replaced by y' —
with the twin's Z' (DETACHED) overriding the K latent positions, over the row's
ORIGINAL question image. Supervision covers exactly two spans:

    observation span -> obs' tokens
    answer span      -> y' tokens

and nothing else (no full-sequence labels).

Why it is gate-proof: the prefix, question, and image in F3 all still support the
*original* observation/answer. The only source of the specific obs'/y' content is
Z'. A model that merely detects "something is off" can move away from the original
but cannot produce the twin's specific content.

Span bookkeeping is the known footgun: obs' and obs tokenize to different lengths,
so spans MUST be recomputed on the spliced sequence (never reused from F1) — the
collator does that; this module just computes the CE and asserts shapes.
"""
from typing import List, Sequence

import torch

from .span_nll import span_ce_to_targets


def swap_loss(logits_b: torch.Tensor,
              obs_prime_positions: Sequence[int], obs_prime_ids: Sequence[int],
              ans_prime_positions: Sequence[int], ans_prime_ids: Sequence[int]) -> torch.Tensor:
    """Dual-span CE for ONE spliced sample. logits_b: [L, V].

    Both spans are supervised from step 0 (rather than picking one after a
    Link-A/B diagnostic): whichever link is broken, the chain Z -> obs -> Y is
    trained end-to-end. The obs-only ablation (arm 2b) recovers attribution.
    """
    assert len(obs_prime_positions) == len(obs_prime_ids) and len(obs_prime_ids) > 0, \
        "obs' span missing/mismatched — collator must recompute spans on the spliced sequence"
    assert len(ans_prime_positions) == len(ans_prime_ids) and len(ans_prime_ids) > 0, \
        "y' span missing/mismatched — collator must recompute spans on the spliced sequence"
    l_obs = span_ce_to_targets(logits_b, obs_prime_positions, obs_prime_ids)
    l_ans = span_ce_to_targets(logits_b, ans_prime_positions, ans_prime_ids)
    return l_obs + l_ans, l_obs.detach(), l_ans.detach()
