"""CPU preflight for the modified Stage-2 run (no GPU, no model weights).

Queue time is expensive; this checks the data-side failure modes on the login
node first. For N samples it:
  1. builds the stage-2 STUDENT tokenization (latent pads, obs spans) exactly as
     the training collator does,
  2. loads that sample's h_pos and h_neg cache files,
  3. verifies: files exist, dtype fp16, layer count == --alignment_layer_indices,
     pos/neg obs-token counts match each other AND the student's span length.

Any mismatch here would crash (or silently skip) the real run.

    python -m src.preflight_stage2 \
      --model_dir /scratch/$USER/monet_weights/Monet-SFT-7B/stage1 \
      --data_path .../Visual_CoT/train.json --dataset_root ... \
      --reps_pos .../teacher_reps_pos --reps_neg .../teacher_reps_neg \
      --alignment_layer_indices 20,21,22,23,24,25,26,27,28 --n 500
"""
import argparse
import json
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="for the processor/tokenizer only")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--reps_pos", required=True)
    ap.add_argument("--reps_neg", required=True)
    ap.add_argument("--alignment_layer_indices", default="20,21,22,23,24,25,26,27,28")
    ap.add_argument("--latent_size", type=int, default=8)
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_dir, use_fast=True,
                                              trust_remote_code=True)
    for t in ("<abs_vis_token_pad>", "<abs_vis_token>", "</abs_vis_token>",
              "<observation>", "</observation>"):
        processor.tokenizer.add_tokens(t, special_tokens=True)
    from src.gate_stage2_collate import _ids, _obs_poss, build_student_stage2_batch

    n_layers = len(args.alignment_layer_indices.split(","))
    rows = json.load(open(args.data_path))[: args.n]
    ids = _ids(processor)

    stats = {"ok": 0, "no_cache": 0, "collate_fail": 0, "layer_bad": 0,
             "dtype_bad": 0, "posneg_mismatch": 0, "student_mismatch": 0,
             "empty_obs": 0}
    for row in rows:
        md = row["metadata"]
        key = f"all_layers_{md['dataset_name']}_{md['sample_id']}"
        fp = os.path.join(args.reps_pos, f"rep_{key}.pt")
        fn = os.path.join(args.reps_neg, f"rep_{key}.pt")
        if not (os.path.exists(fp) and os.path.exists(fn)):
            stats["no_cache"] += 1
            continue
        hp = torch.load(fp, map_location="cpu")["latent"]
        hn = torch.load(fn, map_location="cpu")["latent"]
        if hp.dtype != torch.float16 or hn.dtype != torch.float16:
            stats["dtype_bad"] += 1
            continue
        if hp.shape[0] != n_layers or hn.shape[0] != n_layers:
            stats["layer_bad"] += 1
            continue
        if hp.shape != hn.shape:
            stats["posneg_mismatch"] += 1
            continue
        try:
            batch = build_student_stage2_batch(row, processor, args.dataset_root,
                                               args.latent_size, device="cpu")
            obs = _obs_poss(batch["input_ids"].cpu(), ids)
        except Exception:
            stats["collate_fail"] += 1
            continue
        if not obs:
            stats["empty_obs"] += 1
            continue
        if len(obs) != hp.shape[1]:
            stats["student_mismatch"] += 1
            if stats["student_mismatch"] <= 3:
                print(f"  e.g. {key}: student obs={len(obs)} vs cache T={hp.shape[1]}")
            continue
        stats["ok"] += 1

    total = sum(stats.values())
    print("=" * 56)
    print(f"  Stage-2 preflight  (n={total})")
    for k, v in stats.items():
        print(f"    {k:18s} {v:5d}  ({100*v/max(total,1):.1f}%)")
    bad = total - stats["ok"] - stats["no_cache"]
    print("-" * 56)
    if stats["ok"] >= 0.9 * total:
        print("  VERDICT: healthy -- mismatches will be skipped by the in-run guards")
    elif stats["student_mismatch"] > 0.1 * total:
        print("  VERDICT: STUDENT/CACHE TOKENIZATION DIVERGES -- do NOT launch;")
        print("           the alignment loss would skip most samples")
    else:
        print("  VERDICT: investigate the dominant failure row above before launch")
    print("=" * 56)


if __name__ == "__main__":
    main()
