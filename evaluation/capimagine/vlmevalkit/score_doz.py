"""Score a do(Z) 3-pass VLMEvalKit run by HAND-EXTRACTION.

VLMEvalKit's can_infer mis-scores high-emission models (RL): it trips on the latent
garbage in the output, so its acc.csv is unusable for the do(Z) Delta. This reads
the prediction xlsx from the capture / corrupt_mean / corrupt_gauss passes, extracts
the trailing answer directly, and reports clean vs corrupt accuracy + Delta over the
CLEAN-emitting samples, plus the guard (fraction of outputs that changed).

    python -m evaluation.capimagine.vlmevalkit.score_doz --model Monet-7B
"""
import argparse
import glob
import os
import re

import pandas as pd


def hand_extract(pred, options):
    pred = str(pred)
    b = re.findall(r"\\boxed\{\s*\(?([A-H])", pred)
    if b:
        return b[-1].upper()
    ms = list(re.finditer(r"([A-H])[.\)]", pred))  # last "A." / "A)"
    if ms:
        return ms[-1].group(1).upper()
    tail = pred[-60:].lower()                       # trailing option-text match
    for k, v in options.items():
        if isinstance(v, str) and v.strip() and v.lower().strip() in tail:
            return k
    return None


def _load_pass(vlme, mode, model, data):
    fs = sorted(glob.glob(f"{vlme}/outputs/doz_{mode}/{model}/**/*_{data}.xlsx",
                          recursive=True))
    return pd.read_excel(fs[-1]) if fs else None


def _score(df):
    out = {}
    for _, r in df.iterrows():
        opts = {c: r[c] for c in ["A", "B", "C", "D"] if c in r and pd.notna(r[c])}
        pr = hand_extract(r["prediction"], opts)
        ok = pr is not None and pr == str(r["answer"]).strip().upper()
        out[r["index"]] = (ok, str(r["prediction"]))
    return out


def _emit_ids(emit_log, df):
    if not os.path.exists(emit_log):
        return None
    flags = [l.strip() for l in open(emit_log) if l.strip() in ("0", "1")]
    if len(flags) != len(df):
        return None
    return {idx for idx, f in zip(df["index"], flags) if f == "1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlme", default=f"/scratch/{os.environ.get('USER','')}/VLMEvalKit")
    ap.add_argument("--model", default="Monet-7B")
    ap.add_argument("--data", default="VStarBench")
    a = ap.parse_args()

    clean = _load_pass(a.vlme, "capture", a.model, a.data)
    if clean is None:
        raise SystemExit("no capture (clean) pass found under outputs/doz_capture/")
    cs = _score(clean)
    emit = _emit_ids(
        f"/scratch/{os.environ.get('USER','')}/results/emit/"
        f"doz_capture_{a.model}_{a.data}_emit.log", clean)
    universe = emit if emit else set(cs)

    clean_acc = sum(cs[i][0] for i in universe) / max(len(universe), 1)
    print(f"model={a.model} data={a.data}")
    print(f"do(Z) scored over CLEAN-EMITTING samples: N={len(universe)}"
          + ("" if emit else "  (ALL samples -- no emit log found)"))
    print(f"clean_acc = {clean_acc:.4f}")

    for mode in ["corrupt_mean", "corrupt_gauss"]:
        df = _load_pass(a.vlme, mode, a.model, a.data)
        if df is None:
            print(f"\n--- {mode}: MISSING ---")
            continue
        rs = _score(df)
        ids = [i for i in universe if i in rs]
        corr_acc = sum(rs[i][0] for i in ids) / max(len(ids), 1)
        changed = sum(rs[i][1] != cs[i][1] for i in ids) / max(len(ids), 1)
        print(f"\n--- {mode} ---")
        print(f"corrupt_acc = {corr_acc:.4f}   DELTA = {corr_acc - clean_acc:+.4f}"
              "   (<0 => latents load-bearing; ~0 => cosmetic)")
        print(f"frac_text_changed = {changed:.3f}   "
              "(GUARD: must be > 0, else the intervention didn't fire)")


if __name__ == "__main__":
    main()
