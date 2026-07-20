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
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image

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
    if isinstance(v, str):  # path
        return Image.open(v).convert("RGB")
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
        options_block = _render_options(row)
        stem = str(row[qk]).strip()
        question = stem if not options_block else f"{stem}\n{options_block}"
        samples.append(Sample(
            id=f"{name}-{i:05d}",
            image=_to_pil(row[ik]),
            question=question,
            answer_letter=_answer_to_letter(row[ak], options_block),
            meta={"category": row.get("category")},
        ))
    return samples


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
