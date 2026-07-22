"""Span-restricted NLL utilities shared by the Stage-4 losses.

We avoid passing `labels` into the model for the auxiliary forwards (F3/F4): HF
upcasts the full [B, L, V] logits to fp32 for its CE, which is the dominant memory
cost. Instead we run with labels=None, slice the handful of span positions, and
compute CE on the slice only (upcasting just [S, V]).

Convention: `positions` index the LABEL positions (the tokens being predicted), so
the logit row used for position p is p-1 (standard next-token shift). Position 0 is
never a valid label position.
"""
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F


def find_subsequence(row_ids: torch.Tensor, pattern: Sequence[int],
                     from_end: bool = True) -> Optional[int]:
    """Return the start index of `pattern` inside 1-D `row_ids` (last match if
    `from_end`), or None. Pure tensor ops; no RNG."""
    pattern_t = torch.as_tensor(list(pattern), dtype=row_ids.dtype, device=row_ids.device)
    m = pattern_t.numel()
    if m == 0 or row_ids.numel() < m:
        return None
    windows = row_ids.unfold(0, m, 1)                # [L-m+1, m]
    hits = (windows == pattern_t).all(dim=-1).nonzero(as_tuple=False).flatten()
    if hits.numel() == 0:
        return None
    return int(hits[-1].item() if from_end else hits[0].item())


def find_answer_span(row_ids: torch.Tensor, tokenizer, answer_text: str) -> Optional[List[int]]:
    """Token positions of the final ``\\boxed{answer_text}`` occurrence.

    Tokenized in-context variants are tried because BPE merges differ with the
    preceding character. Returns label positions (see module docstring) or None.
    """
    for variant in (f"\\boxed{{{answer_text}}}", f" \\boxed{{{answer_text}}}",
                    f"boxed{{{answer_text}}}"):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        start = find_subsequence(row_ids, ids, from_end=True)
        if start is not None and start > 0:
            return list(range(start, start + len(ids)))
    return None


def nll_on_positions(logits: torch.Tensor, row_ids: torch.Tensor,
                     positions: Sequence[int]) -> torch.Tensor:
    """Mean NLL of `row_ids[positions]` under `logits` for ONE sample.

    logits: [L, V] (pre-softmax), row_ids: [L]. Uses the p-1 shift convention.
    """
    pos = [p for p in positions if p > 0]
    assert len(pos) > 0, "empty span"
    idx = torch.as_tensor(pos, device=logits.device, dtype=torch.long)
    target = row_ids.to(logits.device)[idx]
    rows = logits.index_select(0, idx - 1).float()   # upcast only the slice
    return F.cross_entropy(rows, target, reduction="mean")


def span_ce_to_targets(logits: torch.Tensor, positions: Sequence[int],
                       target_ids: Sequence[int]) -> torch.Tensor:
    """CE pushing the span at `positions` toward `target_ids` (same length).

    Used by the swap loss where the supervised tokens ARE the input tokens of the
    re-tokenized spliced sequence, so this is teacher-forced CE on those rows.
    """
    assert len(positions) == len(target_ids) and len(positions) > 0
    idx = torch.as_tensor([p for p in positions], device=logits.device, dtype=torch.long)
    tgt = torch.as_tensor(list(target_ids), device=logits.device, dtype=torch.long)
    rows = logits.index_select(0, idx - 1).float()
    return F.cross_entropy(rows, tgt, reduction="mean")
