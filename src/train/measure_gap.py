"""Signed-gap measurement: calibrate the necessity margin M (design doc §4.4).

For N filtered rows, run the base checkpoint (NO training, NO twins needed):

    F1  clean:   latent forward -> Z, CE forward with Z      -> nll_with
    F4  ablated: same CE forward with a DONOR Z              -> nll_ablated
                 (donor = previous row's Z, deterministic ring; different-answer
                  enforced by skipping donors with the same boxed answer)

Reports the distribution of gap = nll_ablated - nll_with over the gold answer
span. Interpretation:
  * gap ~ 0 across rows  -> the answer does not depend on Z at the NLL level —
    the disconnect quantified teacher-forced; M must be set aspirationally
    (e.g. 0.25-1.0 nats) rather than from the base gap.
  * gap >> 0 on a subset -> latents already carry answer-relevant signal there;
    set M near a high percentile of that subset.

Usage (single GPU is fine; bsz=1):
    python -m src.train.measure_gap \
      --load_model_path /scratch/$USER/monet_weights/Monet-SFT-7B/stage3 \
      --data_path .../Visual_CoT/train.json --dataset_root ... \
      --latent_size 8 --n 200 --out gap_report.json
"""
import argparse
import json
import re

import torch


def main():
    # Reuse main.py's argument surface via a slim parser (main.py is a script and
    # cannot be imported without side effects).
    ap = argparse.ArgumentParser()
    ap.add_argument("--load_model_path", required=True)
    ap.add_argument("--data_path", nargs="+", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--latent_size", type=int, default=8)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="gap_report.json")
    args = ap.parse_args()

    # main.py is a script (it trains at import), so we load the model directly:
    # the monet patch swaps the modeling module in sys.modules, then the standard
    # transformers import picks up the patched class.
    import monet_qwen_model.apply_qwen2_5_monet  # noqa: F401  (sys.modules patch)
    from transformers import (AutoProcessor,  # picks up the patched class
                              Qwen2_5_VLForConditionalGeneration)

    processor = AutoProcessor.from_pretrained(args.load_model_path, use_fast=True,
                                              trust_remote_code=True)
    processor.tokenizer.add_tokens("<abs_vis_token_pad>", special_tokens=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.load_model_path, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    from src.train.span_nll import find_answer_span, nll_on_positions

    # -- data: raw rows -> stage-3-style processing is heavy to replicate here, so
    # we lean on the repo collator by spawning it through src.main is NOT possible
    # (script). Minimal equivalent: build the student text directly.
    rows = []
    for p in args.data_path:
        rows.extend(json.load(open(p)))
    latent_pad_id = processor.tokenizer.convert_tokens_to_ids("<abs_vis_token_pad>")

    gaps, skipped = [], 0
    prev_z, prev_ans = None, None
    from src.data.gap_prep import build_student_batch  # shared with tests

    for row in rows:
        if len(gaps) >= args.n:
            break
        try:
            batch, answer_text = build_student_batch(
                row, processor, args.dataset_root, args.latent_size)
        except Exception:
            skipped += 1
            continue
        ids = batch["student_input_ids"][0]
        span = find_answer_span(ids, processor.tokenizer, answer_text)
        if span is None:
            skipped += 1
            continue
        dev = model.device
        common = dict(
            input_ids=batch["student_input_ids"].to(dev),
            attention_mask=batch["student_attention_mask"].to(dev),
            pixel_values=batch["student_pixel_values"].to(dev),
            image_grid_thw=batch["student_image_grid_thw"].to(dev),
            loss_type=[],
        )
        with torch.no_grad():
            lat = model(**common, latent_mode=True, return_dict=True,
                        output_hidden_states=False)
            pos, z = lat.ce_patch_pos[0], lat.ce_patch_vec[0]
            out_with = model(**common, latent_mode=False, return_dict=True,
                             ce_patch_pos=[pos], ce_patch_vec=[z])
            nll_with = nll_on_positions(out_with.logits[0], ids.to(dev), span).item()

            donor = prev_z if (prev_z is not None and prev_ans != answer_text
                               and prev_z.shape == z.shape) else None
            if donor is None:
                prev_z, prev_ans = z, answer_text
                continue
            out_abl = model(**common, latent_mode=False, return_dict=True,
                            ce_patch_pos=[pos], ce_patch_vec=[donor.to(z.device, z.dtype)])
            nll_abl = nll_on_positions(out_abl.logits[0], ids.to(dev), span).item()
        gaps.append({"nll_with": nll_with, "nll_ablated": nll_abl,
                     "gap": nll_abl - nll_with})
        prev_z, prev_ans = z, answer_text
        if len(gaps) % 20 == 0:
            g = sorted(x["gap"] for x in gaps)
            print(f"n={len(gaps)}  median_gap={g[len(g)//2]:+.4f}")

    g = sorted(x["gap"] for x in gaps)
    n = len(g)
    report = {
        "n": n, "skipped": skipped,
        "gap_mean": sum(g) / max(n, 1),
        "gap_p10": g[int(0.10 * n)] if n else None,
        "gap_p50": g[n // 2] if n else None,
        "gap_p90": g[int(0.90 * n)] if n else None,
        "frac_gap_above_0.25": sum(x > 0.25 for x in g) / max(n, 1),
        "samples": gaps,
    }
    json.dump(report, open(args.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
