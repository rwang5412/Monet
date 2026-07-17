"""Answer extraction + scoring for the CapImagine do(Z) harness.

Monet is prompted to "Put your final answer in \\boxed{}", so we extract the
\\boxed{...} content and reduce it to a single option letter. do(Z) measures a
*within-model* delta (clean vs corrupted), so a single consistent extractor
applied identically to both passes is what matters -- we deliberately do NOT use
the paper's supplementary API judge here (that would add noise unrelated to the
intervention and cost API calls).
"""

import re
import string
from typing import Optional

# The Monet latent span is rendered as <abs_vis_token>...</abs_vis_token> (or
# <latent>). The boxed answer always comes after it, so we can ignore the span.
_BOXED_RE = re.compile(r"\\boxed\s*\{")
_LETTER_PAREN_RE = re.compile(r"\(?\s*([A-H])\s*\)?", re.IGNORECASE)
# An isolated option letter, e.g. inside "\text{A}" or "answer: C".
_ISOLATED_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-H])(?![A-Za-z])")


def extract_boxed(text: str) -> Optional[str]:
    """Return the content of the last ``\\boxed{...}`` with brace matching."""
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()  # char right after the '{'
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i - 1].strip() if depth == 0 else text[start:].strip()


def pred_letter(text: str) -> Optional[str]:
    """Best-effort single option letter (A-H) predicted by the model."""
    boxed = extract_boxed(text)
    if boxed:
        m = _LETTER_PAREN_RE.match(boxed.strip())
        if m:
            return m.group(1).upper()
        single = boxed.strip()
        if len(single) == 1 and single.upper() in string.ascii_uppercase:
            return single.upper()
        # e.g. "\text{A}" or "A. foo": grab an isolated option letter
        m = _ISOLATED_LETTER_RE.search(single)
        if m:
            return m.group(1).upper()
        return None
    # no boxed answer: fall back to a trailing "The answer is (B)" style phrase
    tail = text[-200:]
    m = re.search(r"answer\s*(?:is|:)?\s*\(?([A-H])\)?", tail, re.IGNORECASE)
    return m.group(1).upper() if m else None


def is_correct(text: str, gold_letter: str) -> bool:
    p = pred_letter(text)
    return p is not None and p == gold_letter.strip().upper()
