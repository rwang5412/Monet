"""Counterfactual collator for Stage 4.

Pairs each Pool-A row with its counterfactual twin (sidecar `cf/pairs.jsonl`) and
builds the tokenized variants for the four forwards:

    F1 clean(row)    — base stage-3 student collate of the original example
    F2 clean(twin)   — same, with aux'/original' images and obs'/y' text (Z' source)
    F3 spliced       — row's images + text with obs->obs' and boxed y->boxed y',
                       spans RECOMPUTED on the re-tokenized sequence
    F4 ablated       — F1's inputs reused; only the injected latents differ

Design rules enforced here:
  * originals are read-only — the twin/spliced examples are deep-copied edits
  * spans are recomputed per variant (obs' and obs tokenize differently); we
    ASSERT on failure rather than silently skipping supervision
  * no RNG anywhere in this module (byte-identical-baseline guarantee)

The base collator is injected (main.py's collate_fn_sft_stage3 bound via partial)
so all image resizing / latent-pad / label machinery stays single-sourced.
"""
import copy
import json
import os
from typing import Callable, Dict, List, Optional

import torch

from ..train.span_nll import find_answer_span, find_subsequence


class CFStore:
    """row_id -> twin record, loaded once from cf/pairs.jsonl."""

    def __init__(self, pairs_path: str):
        self.by_id: Dict[str, dict] = {}
        if not os.path.exists(pairs_path):
            # No counterfactual data yet: every row runs Pool-B style (F1 only).
            # Legal for arms 0/1 (baseline + writer-only); arm 2 without pairs
            # would silently train no reader loss, so warn loudly.
            print(f"[CFStore] WARNING: {pairs_path} not found -> 0 twins loaded; "
                  f"all rows run clean-forward only (arms 0/1). Do NOT run arm 2 like this.")
            return
        with open(pairs_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("verifier_pass"):
                        self.by_id[r["row_id"]] = r

    def get(self, metadata: dict) -> Optional[dict]:
        return self.by_id.get(f"{metadata['dataset_name']}_{metadata['sample_id']}")


def _edit_example(example: dict, obs_new: str, ans_old: str, ans_new: str,
                  aux_image_path: Optional[str], orig_image_path: Optional[str]) -> dict:
    """Deep-copied edit of a raw dataset row: swap observation text, boxed answer,
    and (for the twin) the image paths. Assistant turns only; user text untouched."""
    ex = copy.deepcopy(example)
    replaced_obs = replaced_ans = False
    for msg in ex["data"]:
        if msg.get("role") == "user" and orig_image_path:
            for item in msg["content"]:
                if item.get("type") == "image":
                    item["image"] = orig_image_path
        if msg.get("role") != "assistant":
            continue
        for item in msg["content"]:
            if item.get("type") == "image" and aux_image_path:
                item["image"] = aux_image_path
            if item.get("type") == "text":
                t = item["text"]
                if "<observation>" in t and not replaced_obs:
                    s = t.index("<observation>") + len("<observation>")
                    e = t.index("</observation>")
                    t = t[:s] + obs_new + t[e:]
                    replaced_obs = True
                old_boxed, new_boxed = f"\\boxed{{{ans_old}}}", f"\\boxed{{{ans_new}}}"
                if old_boxed in t:
                    t = t.replace(old_boxed, new_boxed)
                    replaced_ans = True
                item["text"] = t
    assert replaced_obs, "row has no <observation> span to edit"
    assert replaced_ans, f"row has no \\boxed{{{ans_old}}} to edit"
    return ex


class CFCollator:
    """Builds the four-forward batch for ONE (row, twin) pair (Monet trains bsz=1)."""

    def __init__(self, base_collate: Callable, tokenizer, cf_store: CFStore,
                 pool_b: bool = False):
        self.base_collate = base_collate
        self.tokenizer = tokenizer
        self.cf = cf_store
        self.pool_b = pool_b  # Pool-B rows: clean forward only, no twin required

    def _spans(self, batch: dict, answer_text: str) -> dict:
        ids = batch["student_input_ids"][0]
        ans = find_answer_span(ids, self.tokenizer, answer_text)
        assert ans is not None, f"answer span \\boxed{{{answer_text}}} not found after tokenization"
        obs = batch["observation_poss"][0]
        assert len(obs) > 0, "observation span empty after tokenization"
        return {"obs_positions": obs, "ans_positions": ans,
                "obs_ids": ids[torch.as_tensor(obs)].tolist(),
                "ans_ids": ids[torch.as_tensor(ans)].tolist()}

    def __call__(self, examples: List[dict]) -> dict:
        assert len(examples) == 1, "Stage 4 uses logical bsz=1 (row+twin pairing)"
        row = examples[0]
        out: dict = {"is_pool_a": False}

        f1 = self.base_collate([row])
        out["f1"] = f1
        y = row["cf_answer"] if "cf_answer" in row else row.get("answer_text")
        # answer text is recovered from the raw row at preprocess time and stashed
        # on the example dict; assert rather than guess.
        assert y is not None, "example missing 'answer_text' (set in stage-4 preprocess)"
        out["f1_spans"] = self._spans(f1, y)
        out["answer_text"] = y

        twin_rec = None if self.pool_b else self.cf.get(row["metadata"])
        if twin_rec is None:
            return out  # Pool-B (or unpaired) row: clean forward only

        out["is_pool_a"] = True
        out["twin"] = twin_rec
        assert twin_rec["y_prime"] != y, "twin must be different-answer (y' != y)"

        # F2: twin example — edited images + obs'/y' text
        twin_ex = _edit_example(row, twin_rec["obs_prime"], y, twin_rec["y_prime"],
                                twin_rec["aux_prime_path"], twin_rec["orig_prime_path"])
        f2 = self.base_collate([twin_ex])
        out["f2"] = f2

        # F3: spliced — ROW's images, twin TEXT (obs', y'); spans recomputed here
        splice_ex = _edit_example(row, twin_rec["obs_prime"], y, twin_rec["y_prime"],
                                  aux_image_path=None, orig_image_path=None)
        f3 = self.base_collate([splice_ex])
        out["f3"] = f3
        out["f3_spans"] = self._spans(f3, twin_rec["y_prime"])
        # F4 reuses F1's tensors; only injected latents differ (no extra collate).
        return out
