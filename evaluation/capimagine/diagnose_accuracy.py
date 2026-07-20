"""Bucket a clean-pass JSON's wrong answers to trace an accuracy gap (no GPU).

Given a `<dataset>_clean.json` written by the harness, split every record into:
  correct
  wrong: no_latent      -- model never emitted <abs_vis_token> (151666)
  wrong: no_boxed       -- text has NO \\boxed{...} at all (truncation / no answer;
                           an API judge canNOT rescue this -- there's no answer)
  wrong: unparseable    -- has \\boxed{...} but no option letter extracted
                           (API-judge-rescuable: the answer may be there as text)
  wrong: wrong_letter   -- a clean option letter was predicted, just the wrong one
                           (genuinely wrong; no judge/prompt rescues this)

If `no_boxed` is a big chunk, raise --max-tokens (one-line fix). If `wrong_letter`
dominates, the model is genuinely wrong -> look at decoding/latent-size/resolution,
not scoring. If `unparseable` is large, the exact-match-vs-API-judge gap is real.

    python -m evaluation.capimagine.diagnose_accuracy /path/to/vstar_clean.json
"""

import argparse
import json

from evaluation.capimagine import scoring


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clean_json", help="path to a <dataset>_<mode>.json from the harness")
    ap.add_argument("--show", type=int, default=6,
                    help="print this many example texts per wrong bucket")
    args = ap.parse_args()

    recs = json.load(open(args.clean_json))["records"]
    n = len(recs)
    buckets = {"correct": [], "no_latent": [], "no_boxed": [],
               "unparseable": [], "wrong_letter": []}

    for r in recs:
        if r["correct"]:
            buckets["correct"].append(r); continue
        if not r.get("emitted_latent", False):
            buckets["no_latent"].append(r); continue
        boxed = scoring.extract_boxed(r["text"])
        if boxed is None:
            buckets["no_boxed"].append(r)
        elif scoring.pred_letter(r["text"]) is None:
            buckets["unparseable"].append(r)
        else:
            buckets["wrong_letter"].append(r)

    n_emit = sum(r.get("emitted_latent", False) for r in recs)
    print(f"file: {args.clean_json}")
    print(f"N={n}  latent-emitting={n_emit} ({100*n_emit/max(n,1):.1f}%)  "
          f"accuracy(all)={len(buckets['correct'])/max(n,1):.4f}")
    print("\n--- wrong-answer breakdown ---")
    for k in ("no_latent", "no_boxed", "unparseable", "wrong_letter"):
        b = buckets[k]
        print(f"  {k:13s} {len(b):4d}  ({100*len(b)/max(n,1):.1f}% of all)")
    print(f"  {'correct':13s} {len(buckets['correct']):4d}")

    # Ceiling: accuracy if an API judge rescued every unparseable answer AND every
    # no-boxed answer were instead a (truncated) correct one -- the optimistic bound.
    rescuable = len(buckets["unparseable"]) + len(buckets["no_boxed"])
    ceil = (len(buckets["correct"]) + rescuable) / max(n, 1)
    print(f"\noptimistic ceiling if judge+truncation fully rescued = {ceil:.4f}")
    print("(if this ceiling is still << 0.82, scoring/truncation is NOT the gap)")

    for k in ("no_boxed", "unparseable", "wrong_letter"):
        b = buckets[k]
        if not b:
            continue
        print(f"\n===== examples: {k} =====")
        for r in b[:args.show]:
            tail = r["text"][-300:].replace("\n", " ")
            print(f"  id={r['id']} gold={r['gold']} pred={r['pred']}  ...{tail}")


if __name__ == "__main__":
    main()
