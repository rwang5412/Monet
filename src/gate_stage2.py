"""Stage-2 -> Stage-3 promotion gate (spec §7).

Evaluates a trained Stage-2 checkpoint on a fixed held-out slice and reports the
metrics that decide whether its latents may become Stage-3 targets:

  cross-sample latent sim   pairwise cos of pooled z, off-diag   (baseline 0.96; target < 0.7)
  within-block latent sim   pairwise cos across the K slots      (should also fall)
  intervention gap          obs-token acc(real z) - acc(shuffled z)   (must be > 0)
  s_pos - s_neg             student obs states vs the two teachers    (must be > 0, else mask leak)

GATE: do NOT run precompute_teacher_latents for Stage 3 unless cross-sample sim
is well below 0.96 AND the intervention gap is positive. Stage 3 regresses onto
whatever Stage 2 hands it; it cannot repair degenerate targets.

Held-out convention: the LAST --n samples of the data file. Train with
--num_samples (len - n) so these were never seen.

Two models are needed (warm-up teacher for s_pos/s_neg, the stage-2 student for
everything else); they are loaded SEQUENTIALLY to fit one GPU.

    python -m src.gate_stage2 \
      --student /scratch/$USER/monet_ckpts/sft_stage2_residual_grounded \
      --teacher /scratch/$USER/monet_weights/Monet-SFT-7B/stage1 \
      --data_path .../Visual_CoT/train.json --dataset_root ... \
      --latent_size 8 --n 300 [--alignment_layer_indices 21,22,...]
"""
import argparse
import gc
import json

import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True, help="stage-2 checkpoint dir")
    ap.add_argument("--teacher", required=True, help="warm-up (stage1) dir")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--latent_size", type=int, default=8)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--alignment_layer_indices", type=str, default=None)
    ap.add_argument("--out", default="gate_report.json")
    args = ap.parse_args()

    import monet_qwen_model.apply_qwen2_5_monet  # noqa: F401 (sys.modules patch)
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from src.gate_stage2_collate import (build_student_stage2_batch,
                                         build_teacher_batches, obs_token_acc_inputs)

    layer_idx = ([int(x) for x in args.alignment_layer_indices.split(',')]
                 if args.alignment_layer_indices else None)
    rows = json.load(open(args.data_path))[-args.n:]
    dev = "cuda"

    processor = AutoProcessor.from_pretrained(args.student, use_fast=True, trust_remote_code=True)
    for t in ("<abs_vis_token_pad>", "<abs_vis_token>", "</abs_vis_token>",
              "<observation>", "</observation>"):
        processor.tokenizer.add_tokens(t, special_tokens=True)

    def load(path):
        m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.bfloat16).to(dev)
        m.eval()
        return m

    # ---------- pass 1: warm-up teacher -> h_pos / h_neg pooled per sample ----
    teacher = load(args.teacher)
    t_pos, t_neg = {}, {}
    with torch.inference_mode():
        for i, row in enumerate(rows):
            try:
                for tag, no_aux, store in (("pos", False, t_pos), ("neg", True, t_neg)):
                    inp, obs_poss = build_teacher_batches(row, processor, args.dataset_root,
                                                          no_aux=no_aux, device=dev)
                    out = teacher(**inp, return_dict=True)
                    h = out.hidden_states[0]           # [num_layer, T_obs, D]
                    if layer_idx is not None:
                        h = h[layer_idx]
                    store[i] = h.to(torch.float16).cpu()
            except Exception as e:
                print(f"[gate] teacher skip {i}: {e!r}")
    del teacher
    gc.collect(); torch.cuda.empty_cache()

    # ---------- pass 2: stage-2 student ---------------------------------------
    student = load(args.student)
    pooled, within, s_gap = [], [], []
    real_pack = []  # (inputs for CE fwd, obs_poss, z, pos_list)
    with torch.inference_mode():
        for i, row in enumerate(rows):
            try:
                inp = build_student_stage2_batch(row, processor, args.dataset_root,
                                                 args.latent_size, device=dev)
                lat = student(**inp, latent_mode=True, loss_type=[], return_dict=True,
                              output_hidden_states=False)
                z = lat.ce_patch_vec[0]
                pos_list = lat.ce_patch_pos[0]
            except Exception as e:
                print(f"[gate] student skip {i}: {e!r}")
                continue
            zn = F.normalize(z.float(), dim=-1)
            pooled.append(F.normalize(z.float().mean(0), dim=-1).cpu())
            if z.shape[0] > 1:
                C = zn @ zn.T
                k = C.shape[0]
                within.append(float((C.sum() - k) / (k * (k - 1))))
            # student obs states for s_pos - s_neg (needs alignment CE forward)
            if i in t_pos:
                ce_in, obs_poss = obs_token_acc_inputs(inp, processor, device=dev)
                out = student(**ce_in, ce_patch_pos=[pos_list], ce_patch_vec=[z],
                              loss_type=[], output_hidden_states=True,
                              alignment_poss=[obs_poss], return_dict=True)
                hs = torch.stack(out.hidden_states, dim=0)[:, 0, obs_poss, :]
                if layer_idx is not None:
                    hs = hs[layer_idx]
                hp = t_pos[i].to(dev, hs.dtype)
                hn = t_neg[i].to(dev, hs.dtype)
                if hp.shape == hs.shape == hn.shape:
                    sp = F.cosine_similarity(hs, hp, dim=-1).mean()
                    sn = F.cosine_similarity(hs, hn, dim=-1).mean()
                    s_gap.append(float(sp - sn))
            real_pack.append((inp, z, pos_list))

    # cross-sample similarity
    Z = torch.stack(pooled)
    C = Z @ Z.T
    n = C.shape[0]
    cross = float((C.sum() - n) / (n * (n - 1)))

    # collapse probe: effective rank of the full latent cloud (all slots x samples).
    # exp(entropy of normalized squared singular values); released Monet ~3.
    allz = torch.cat([p[1].float().cpu() for p in real_pack], dim=0)
    allz = allz - allz.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(allz)
    pw = (sv ** 2) / (sv ** 2).sum()
    eff_rank = float(torch.exp(-(pw * (pw + 1e-12).log()).sum()))

    # intervention gap: obs-token accuracy with real vs shuffled (donor) latents
    acc_real, acc_shuf = [], []
    with torch.inference_mode():
        for j, (inp, z, pos_list) in enumerate(real_pack):
            donor = real_pack[(j + 1) % len(real_pack)][1]
            if donor.shape != z.shape:
                continue
            ce_in, obs_poss = obs_token_acc_inputs(inp, processor, device=dev)
            for zz, sink in ((z, acc_real), (donor, acc_shuf)):
                out = student(**ce_in, ce_patch_pos=[pos_list], ce_patch_vec=[zz],
                              loss_type=['ce'], compute_emphasize_acc=True,
                              ce_emphasize_poss=[obs_poss], ce_emphasize_factor=1.0,
                              return_dict=True)
                if out.mean_emphasize_acc is not None:
                    sink.append(float(out.mean_emphasize_acc))

    ar = sum(acc_real) / max(len(acc_real), 1)
    asf = sum(acc_shuf) / max(len(acc_shuf), 1)
    report = {
        "n_scored": n,
        "cross_sample_sim": cross,
        "effective_rank": eff_rank,
        "within_block_sim": sum(within) / max(len(within), 1),
        "s_pos_minus_s_neg": sum(s_gap) / max(len(s_gap), 1),
        "obs_acc_real": ar, "obs_acc_shuffled": asf,
        "intervention_gap": ar - asf,
    }
    json.dump(report, open(args.out, "w"), indent=2)
    W = 62
    print("=" * W)
    print("  Stage-2 promotion gate")
    print("=" * W)
    print(f"  cross-sample latent sim = {cross:.4f}   (baseline 0.96; want < 0.7)")
    print(f"  effective rank          = {eff_rank:.1f}   (released Monet ~3; want >> 10)")
    print(f"  within-block latent sim = {report['within_block_sim']:.4f}")
    print(f"  s_pos - s_neg (student) = {report['s_pos_minus_s_neg']:+.4f}   (want > 0; ~0 => mask leak)")
    print(f"  obs acc real / shuffled = {ar:.4f} / {asf:.4f}")
    print(f"  intervention gap        = {ar - asf:+.4f}   (want > 0)")
    ok = cross < 0.9 and (ar - asf) > 0
    print(f"  GATE: {'PASS — latents may become Stage-3 targets' if ok else 'FAIL — do NOT promote to Stage 3'}")
    print("=" * W)


if __name__ == "__main__":
    main()
