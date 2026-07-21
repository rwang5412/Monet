"""Score a do(Z) 3-pass VLMEvalKit run using the DeepSeek-JUDGED per-sample results.

Under the README prompt the model REASONS, so hand-extraction is too noisy to score
a do(Z) Delta. So: run the 3 passes (run_vlmeval_doz.sbatch), THEN judge each pass
with DeepSeek on the login node, THEN run this.

  # 1. passes (GPU sbatch) -> predictions in outputs/doz_{capture,corrupt_mean,corrupt_gauss}/
  # 2. judge each pass (login node, DeepSeek key in $VLME/.env):
  for M in capture corrupt_mean corrupt_gauss; do
    python run.py --data VStarBench --model $MODEL --judge deepseek-chat --reuse \
      --work-dir outputs/doz_$M
  done
  # 3. score:
  python -m evaluation.capimagine.vlmevalkit.score_doz --model $MODEL

Reports clean vs corrupt accuracy + Delta over the CLEAN-EMITTING samples (the do(Z)
metric), plus the text-change guard (corrupt output MUST differ from clean, else the
intervention didn't fire).
"""
import argparse
import glob
import os

import pandas as pd

MODES = ["capture", "corrupt_mean", "corrupt_gauss"]


def _judged(vlme, mode, model, data):
    fs = sorted(glob.glob(
        f"{vlme}/outputs/doz_{mode}/{model}/**/*{data}*result.xlsx", recursive=True))
    return pd.read_excel(fs[-1]) if fs else None


def _pred(vlme, mode, model, data):
    fs = sorted(glob.glob(
        f"{vlme}/outputs/doz_{mode}/{model}/**/{model}_{data}.xlsx", recursive=True))
    return pd.read_excel(fs[-1]) if fs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlme", default=f"/scratch/{os.environ.get('USER','')}/VLMEvalKit")
    ap.add_argument("--model", default="Monet-7B-readme")
    ap.add_argument("--data", default="VStarBench")
    a = ap.parse_args()

    hit, pred = {}, {}
    for m in MODES:
        j = _judged(a.vlme, m, a.model, a.data)
        if j is None:
            continue
        if "hit" not in j.columns:
            raise SystemExit(f"{m}: no 'hit' column in judged result; got {list(j.columns)}. "
                             "Did the judge step run? (see this file's header)")
        hit[m] = {r["index"]: int(r["hit"]) for _, r in j.iterrows()}
        p = _pred(a.vlme, m, a.model, a.data)
        if p is not None:
            pred[m] = {r["index"]: str(r["prediction"]) for _, r in p.iterrows()}

    if "capture" not in hit:
        raise SystemExit("no JUDGED capture (clean) result -- run the judge step first.")

    order = list(_judged(a.vlme, "capture", a.model, a.data)["index"])
    emitlog = (f"/scratch/{os.environ.get('USER','')}/results/emit/"
               f"doz_capture_{a.model}_{a.data}_emit.log")
    flags = [l.strip() for l in open(emitlog)] if os.path.exists(emitlog) else []
    if len(flags) == len(order):
        emit = {i for i, f in zip(order, flags) if f == "1"}
        note = ""
    else:
        emit = set(order)
        note = "  (ALL samples -- emit log missing/mismatched)"

    ch = hit["capture"]
    clean_acc = sum(ch[i] for i in emit) / max(len(emit), 1)
    print(f"model={a.model}  data={a.data}")
    print(f"do(Z) over CLEAN-EMITTING samples: N={len(emit)}{note}")
    print(f"clean_acc = {clean_acc:.4f}")
    for m in ["corrupt_mean", "corrupt_gauss"]:
        if m not in hit:
            print(f"\n--- {m}: MISSING (judge it) ---")
            continue
        mh = hit[m]
        ids = [i for i in emit if i in mh]
        corr = sum(mh[i] for i in ids) / max(len(ids), 1)
        if m in pred and "capture" in pred:
            chg = sum(pred[m].get(i, "") != pred["capture"].get(i, "") for i in ids) / max(len(ids), 1)
            chg_s = f"{chg:.3f}"
        else:
            chg_s = "n/a"
        print(f"\n--- {m} ---")
        print(f"corrupt_acc = {corr:.4f}   DELTA = {corr - clean_acc:+.4f}   "
              "(<0 => latents load-bearing; ~0 => cosmetic)")
        print(f"frac_text_changed = {chg_s}   (GUARD: must be > 0, else hook didn't fire)")


if __name__ == "__main__":
    main()
