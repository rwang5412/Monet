"""Score a do(Z) 3-pass VLMEvalKit run using the DeepSeek-JUDGED per-sample results,
and print a clean CapImagine report.

Pipeline (see run_vlmeval_doz.sbatch + doz_report.sh):
  1. GPU: 3 passes (capture / corrupt_mean / corrupt_gauss) -> predictions
  2. login: judge each pass with DeepSeek (--judge deepseek-chat --reuse --work-dir ...)
  3. this: read judged 'hit' + emit log -> Delta over clean-emitting samples + guard

    python -m evaluation.capimagine.vlmevalkit.score_doz --model Monet-7B-readme
"""
import argparse
import glob
import os

import pandas as pd

MODES = ["capture", "corrupt_mean", "corrupt_gauss"]
LABEL = {"capture": "clean (capture)", "corrupt_mean": "corrupt_mean",
         "corrupt_gauss": "corrupt_gauss"}


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
            raise SystemExit(f"{m}: no 'hit' column (got {list(j.columns)}); judge it first.")
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
        emit_note = f"{len(emit)} of {len(order)}"
    else:
        emit = set(order)
        emit_note = f"{len(emit)} (ALL -- emit log missing)"

    clean_acc = sum(hit["capture"][i] for i in emit) / max(len(emit), 1)

    W = 62
    print("=" * W)
    print(f"  CapImagine do(Z)  |  {a.model}  |  {a.data}")
    print("=" * W)
    print(f"  clean-emitting samples: N = {emit_note}")
    print()
    print(f"  {'pass':<16}{'accuracy':>10}{'Δ vs clean':>13}{'text-changed':>15}")
    print("  " + "-" * (W - 4))
    print(f"  {LABEL['capture']:<16}{clean_acc:>10.4f}{'—':>13}{'—':>15}")

    guard_ok, deltas = True, []
    for m in ["corrupt_mean", "corrupt_gauss"]:
        if m not in hit:
            print(f"  {LABEL[m]:<16}{'MISSING (judge it)':>38}")
            continue
        ids = [i for i in emit if i in hit[m]]
        acc = sum(hit[m][i] for i in ids) / max(len(ids), 1)
        d = acc - clean_acc
        deltas.append(d)
        if m in pred and "capture" in pred:
            chg = sum(pred[m].get(i, "") != pred["capture"].get(i, "") for i in ids) / max(len(ids), 1)
        else:
            chg = float("nan")
        if not (chg > 0.05):
            guard_ok = False
        print(f"  {LABEL[m]:<16}{acc:>10.4f}{d:>+13.4f}{chg:>15.3f}")

    print("  " + "-" * (W - 4))
    worst = min(deltas) if deltas else 0.0
    if not guard_ok:
        verdict = "GUARD FAILED — corrupt output ~= clean; hook may not have fired"
    elif worst <= -0.05:
        verdict = f"LOAD-BEARING — destroying Z costs up to {worst:+.3f} accuracy"
    elif worst >= -0.02:
        verdict = "COSMETIC — Z can be destroyed with ~no accuracy change (disconnect)"
    else:
        verdict = f"WEAK — small effect ({worst:+.3f}); inconclusive"
    print(f"  guard : {'OK (interventions changed the output)' if guard_ok else 'FAILED'}")
    print(f"  verdict: {verdict}")
    print("=" * W)


if __name__ == "__main__":
    main()
