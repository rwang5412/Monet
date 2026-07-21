"""Dataset loaders for the CapImagine do(Z) harness.

Currently: V* (``lmms-lab/vstar-bench``) and HR-Bench 4k (``DreamMr/HR-Bench``,
config ``hrbench_4k``). Both are multiple-choice with the image embedded in the
HF arrow file, so no separate image folder is needed.

Each loader returns a list of ``Sample`` records with a stable ``id`` (so clean
and corrupt passes line up), a PIL image, the fully-rendered question text
(stem + lettered options), and the gold answer letter.

The exact column names of these HF datasets have drifted across versions, so the
loaders resolve fields defensively and raise a clear error listing the available
columns if they cannot. Run ``python -m evaluation.capimagine.datasets --inspect
<name>`` to print the schema before a full run.
"""

import argparse
import re
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

# Some sources (lmms-lab/vstar-bench) bake the option list AND a terminal
# "Answer with the option's letter ... directly." instruction into the question
# text. That instruction tells the model to skip reasoning, which suppresses latent
# emission and contradicts a "put your answer in \boxed{}" instruction. We strip it
# and parse the inline options so the harness can rebuild a clean MCQ prompt.
_ANSWER_HINT_RE = re.compile(r"answer with the option'?s? letter", re.IGNORECASE)
_OPT_PAREN_RE = re.compile(r"^\s*\(([A-H])\)\s*(.+?)\s*$")      # "(A) rubber"
_OPT_DOT_RE = re.compile(r"^\s*([A-H])[.)]\s+(.+?)\s*$")        # "A. rubber" / "A) rubber"


def _parse_inline_mcq(text: str) -> Tuple[str, Dict[str, str]]:
    """Split a question whose options are inline into (stem, {letter: option}).
    Returns ("", {}) if no inline options are found (e.g. options live in columns)."""
    stem_lines: List[str] = []
    options: Dict[str, str] = {}
    started = False
    for ln in text.splitlines():
        if _ANSWER_HINT_RE.search(ln):
            continue  # drop the redundant/contradictory instruction
        m = _OPT_PAREN_RE.match(ln) or _OPT_DOT_RE.match(ln)
        if m:
            options[m.group(1)] = m.group(2).strip()
            started = True
        elif not started and ln.strip():
            stem_lines.append(ln.strip())
    return " ".join(stem_lines).strip(), options

# HF dataset coordinates
VSTAR = dict(repo="lmms-lab/vstar-bench", config=None, split="test")
HRBENCH_4K = dict(repo="DreamMr/HR-Bench", config="hrbench_version_split", split="hrbench_4k")

DATASETS = {"vstar": VSTAR, "hrbench_4k": HRBENCH_4K}

# Candidate column names, tried in order.
_Q_KEYS = ["question", "text", "query", "prompt"]
_ANS_KEYS = ["answer", "label", "gt_answer", "correct_answer", "gt"]
_IMG_KEYS = ["image", "images", "img", "pil_image"]
_OPTION_KEYS = ["options", "choices", "candidates"]
_LETTER_OPTION_KEYS = list(string.ascii_uppercase[:8])  # A..H as separate columns


@dataclass
class Sample:
    id: str
    image: Image.Image
    question: str          # stem + rendered options
    answer_letter: str     # gold letter, e.g. "B"
    meta: Dict[str, Any]


def _first_key(row: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in row and row[k] is not None:
            return k
    return None


def _to_pil(v: Any) -> Image.Image:
    if isinstance(v, Image.Image):
        return v.convert("RGB")
    if isinstance(v, list) and v:  # some sets store a list of images
        return _to_pil(v[0])
    if isinstance(v, dict) and "bytes" in v and v["bytes"] is not None:
        import io
        return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
    if isinstance(v, str):  # a file path OR a base64-encoded image
        import os
        if os.path.exists(v):
            return Image.open(v).convert("RGB")
        # HR-Bench stores images as base64-encoded JPEG strings in a string column.
        import io, base64
        s = v.split(",", 1)[1] if v.startswith("data:") else v  # strip any data-URI prefix
        s = "".join(s.split())                                   # drop whitespace/newlines
        s += "=" * (-len(s) % 4)                                 # fix missing padding
        try:
            return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
        except Exception as e:
            raise TypeError(f"str image is neither an existing path nor base64: {e!r}")
    raise TypeError(f"Cannot coerce image field of type {type(v)} to PIL")


def _render_options(row: Dict[str, Any]) -> Optional[str]:
    """Return a rendered "(A) ...\n(B) ..." block, or None if not applicable."""
    ok = _first_key(row, _OPTION_KEYS)
    if ok is not None:
        opts = row[ok]
        if isinstance(opts, dict):  # {"A": "...", "B": "..."}
            items = [(k, opts[k]) for k in sorted(opts)]
        else:  # list
            items = list(zip(string.ascii_uppercase, opts))
        return "\n".join(f"({k}) {v}" for k, v in items)
    # per-letter columns A/B/C/D
    letters = [k for k in _LETTER_OPTION_KEYS if k in row and row[k] not in (None, "")]
    if letters:
        return "\n".join(f"({k}) {row[k]}" for k in letters)
    return None


def _answer_to_letter(ans: Any, options_block: Optional[str]) -> str:
    """Normalize a gold answer to a single letter A-H."""
    s = str(ans).strip()
    if len(s) == 1 and s.upper() in string.ascii_uppercase:
        return s.upper()
    # answer given as the option text -> match it back to a letter
    if options_block:
        for line in options_block.splitlines():
            # line looks like "(B) some text"
            if line[1:2].isalpha() and line[3:].strip().lower() == s.lower():
                return line[1:2].upper()
    # last resort: strip a leading "(A)"/"A." style marker
    for ch in s:
        if ch.upper() in string.ascii_uppercase:
            return ch.upper()
    return s.upper()


def load(name: str, limit: Optional[int] = None) -> List[Sample]:
    from datasets import load_dataset

    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; options: {list(DATASETS)}")
    coords = DATASETS[name]
    ds = (load_dataset(coords["repo"], coords["config"], split=coords["split"])
          if coords["config"] else
          load_dataset(coords["repo"], split=coords["split"]))

    cols = ds.column_names
    qk = _first_key(ds[0], _Q_KEYS)
    ak = _first_key(ds[0], _ANS_KEYS)
    ik = _first_key(ds[0], _IMG_KEYS)
    if not (qk and ak and ik):
        raise KeyError(
            f"{name}: could not resolve question/answer/image columns from "
            f"{cols}. Resolved q={qk} a={ak} img={ik}. Inspect with "
            f"`python -m evaluation.capimagine.datasets --inspect {name}`.")

    samples: List[Sample] = []
    n = len(ds) if limit is None else min(limit, len(ds))
    for i in range(n):
        row = ds[i]
        raw_q = str(row[qk]).strip()
        options_block = _render_options(row)  # options in separate columns?
        if options_block is not None:
            # Options live in dedicated columns; question text is the bare stem.
            stem = raw_q
            options = _block_to_options(options_block)
        else:
            # Options (and possibly a redundant instruction) are inline in the text.
            stem, options = _parse_inline_mcq(raw_q)
            if options:
                options_block = "\n".join(f"({k}) {options[k]}" for k in sorted(options))
            else:
                # No parseable options: keep the text, just drop the redundant hint.
                stem = "\n".join(l for l in raw_q.splitlines()
                                 if not _ANSWER_HINT_RE.search(l)).strip()
        question = stem if not options_block else f"{stem}\n{options_block}"
        samples.append(Sample(
            id=f"{name}-{i:05d}",
            image=_to_pil(row[ik]),
            question=question,
            answer_letter=_answer_to_letter(row[ak], options_block),
            meta={"category": row.get("category"), "stem": stem, "options": options},
        ))
    return samples


def _block_to_options(block: str) -> Dict[str, str]:
    opts: Dict[str, str] = {}
    for ln in block.splitlines():
        m = _OPT_PAREN_RE.match(ln) or _OPT_DOT_RE.match(ln)
        if m:
            opts[m.group(1)] = m.group(2).strip()
    return opts


def _inspect(name: str):
    from datasets import load_dataset
    coords = DATASETS[name]
    ds = (load_dataset(coords["repo"], coords["config"], split=coords["split"])
          if coords["config"] else
          load_dataset(coords["repo"], split=coords["split"]))
    print(f"{name}: n={len(ds)} columns={ds.column_names}")
    row = ds[0]
    for k, v in row.items():
        preview = (f"<{type(v).__name__}>" if isinstance(v, Image.Image)
                   else repr(v)[:120])
        print(f"  {k}: {preview}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", choices=list(DATASETS), required=True)
    args = ap.parse_args()
    _inspect(args.inspect)
