"""Remove the latent-step tokens from do(Z) prediction files before judging.

The Monet vLLM runner appends the token sampled at each of the K latent steps
to the output text. They decode as one short junk token plus whitespace, e.g.

    "A   \\n\\n red"        "词汇      A. rubber"        "B      The question asks..."

Any answer extractor -- exact matching or an LLM judge -- sees a standalone
option letter at position 0 and takes it. The letter is noise, so a checkpoint
that answers tersely (no \\boxed{}) scores at chance: our stage-3 checkpoint
read 0.31 raw and 0.68 stripped on VStarBench (2026-09-07). The released model
escapes this only because it ends with \\boxed{X}, which extractors prefer.

Rewrites <model>_<data>.xlsx in place (original kept as *.raw.xlsx once) and
prints how many rows matched the prefix pattern -- if that is far below the
emission rate, the pattern needs revisiting.

    python -m evaluation.capimagine.vlmevalkit.strip_latent_prefix \\
        /scratch/$USER/VLMEvalKit/outputs/doz_*/Monet-S3-ours/Monet-S3-ours_VStarBench.xlsx
"""
import argparse
import glob
import os
import re

import pandas as pd

# One leading token (a letter, a CJK word, or -- under corruption -- a full word
# such as "rubber"/"color"/"animal") followed by a run of 2+ whitespace chars. A
# real answer's first word is followed by ONE space, so it is left alone; the
# double spaces Monet leaves mid-sentence ("The scarf is  red") are not at the
# start and are also left alone.
PREFIX = re.compile(r"^\s*\S{1,24}\s{2,}")


def strip_prefix(text: str) -> tuple[str, bool]:
    t = str(text)
    m = PREFIX.match(t)
    return (t[m.end():].strip(), True) if m else (t.strip(), False)


def process(path: str, dry_run: bool = False) -> tuple[int, int]:
    raw = path[:-5] + ".raw.xlsx"
    src = raw if os.path.exists(raw) else path        # idempotent: always strip from the raw copy
    d = pd.read_excel(src)
    out = d["prediction"].astype(str).map(strip_prefix)
    d["prediction"] = [o[0] for o in out]
    n_hit = sum(o[1] for o in out)
    if not dry_run:
        if not os.path.exists(raw):
            os.replace(path, raw)
        d.to_excel(path, index=False)
    return n_hit, len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="prediction xlsx files or globs")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--show_unmatched", action="store_true",
                    help="print the start of every row the pattern did not match")
    a = ap.parse_args()
    files = sorted({f for p in a.paths for f in glob.glob(p, recursive=True)
                    if f.endswith(".xlsx") and not f.endswith(".raw.xlsx")
                    and "result" not in os.path.basename(f)})
    if not files:
        raise SystemExit(f"no prediction files match {a.paths}")
    for f in files:
        n_hit, n = process(f, a.dry_run)
        print(f"{'would strip' if a.dry_run else 'stripped':12s} {n_hit}/{n} rows  {f}")
        if a.show_unmatched:
            raw = f[:-5] + ".raw.xlsx"
            d = pd.read_excel(raw if os.path.exists(raw) else f)
            for t in d["prediction"].astype(str):
                if not PREFIX.match(t):
                    print("   unmatched:", repr(t[:70]))


if __name__ == "__main__":
    main()
