"""Dataset-mean visual residual mu = E[h_pos - h_neg] for Stage-2 margin recentering.

Why: the residual margin rewards cos(h, h_pos) - cos(h, h_neg). The teacher
residual h_pos - h_neg has a component shared by EVERY sample (the "I looked at a
crop" direction mu) plus a sample-specific part. A student can earn margin by
shifting all its obs states along mu without encoding anything about its own
crop -- and the pilot (14946764) shows exactly that drift: cross_sample_sim
0.5 -> 0.81 in 500 steps while retrieval stays fine. Subtracting mu from h_pos
makes h_pos' - h_neg zero-mean across the dataset, so the shared shift earns
nothing and only the sample-specific residual pays.

Reads paired files from the pos/neg caches (same filenames, [num_kept_layer,
T_obs, dim] fp16), averages (h_pos - h_neg) over obs tokens and files, writes
residual_mean.pt = {"mean": [num_kept_layer, dim], ...}. Also reports how much of
the residual is shared and the recentered teacher ceiling (for --obs_residual_margin).
CPU only; 10K files is plenty (dim 3584, noise averages out) and reads ~20GB.

    python -m src.compute_residual_mean \
      --pos_dir /scratch/$USER/monet_ckpts/teacher_reps_pos \
      --neg_dir /scratch/$USER/monet_ckpts/teacher_reps_neg \
      --out /scratch/$USER/monet_ckpts/residual_mean.pt --max_files 10000
"""
import argparse
import os
import random

import torch
import torch.nn.functional as F


def _load(path):
    return torch.load(path, map_location="cpu")["latent"].float()   # [L, T, D]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_dir", required=True)
    ap.add_argument("--neg_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_files", type=int, default=10000, help="0 = all (slow: ~236GB read)")
    ap.add_argument("--stats_files", type=int, default=1000,
                    help="second pass on this many files for the shared-fraction / ceiling report")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = sorted(f for f in os.listdir(args.pos_dir) if f.startswith("rep_") and f.endswith(".pt"))
    names = [f for f in names if os.path.isfile(os.path.join(args.neg_dir, f))]
    assert names, "no paired rep_*.pt files"
    random.Random(args.seed).shuffle(names)
    if args.max_files > 0:
        names = names[:args.max_files]
    print(f"{len(names)} paired files")

    # ---- pass 1: mu = mean over files of (mean over obs tokens of h_pos - h_neg) ----
    acc, n, skipped = None, 0, 0
    for i, f in enumerate(names):
        try:
            p, q = _load(os.path.join(args.pos_dir, f)), _load(os.path.join(args.neg_dir, f))
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"skip {f}: {e!r}")
            continue
        if p.shape != q.shape or p.dim() != 3:
            skipped += 1
            continue
        d = (p - q).mean(dim=1)                       # [L, D]
        acc = d if acc is None else acc + d
        n += 1
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(names)}")
    assert n > 0, "no usable pairs"
    mu = acc / n                                      # [L, D]
    torch.save({"mean": mu, "n_files": n, "shape": list(mu.shape),
                "pos_dir": args.pos_dir, "neg_dir": args.neg_dir}, args.out)
    print(f"wrote mu {tuple(mu.shape)} over {n} files ({skipped} skipped) -> {args.out}")

    # ---- pass 2: diagnostics on a subset --------------------------------------
    #  shared_frac : ||mu|| / E||h_pos - h_neg||   (how much of the residual is shared)
    #  cos(r, mu)  : E cos(h_pos - h_neg, mu)      (per-sample residual vs the mean)
    #  ceiling_raw : E[1 - cos(h_pos, h_neg)]      -> bound for the plain margin
    #  ceiling_rec : E[cos(h_pos, h_pos - mu) - cos(h_pos, h_neg)] -> bound for the
    #                recentered margin (teacher's own obs state as the "student")
    L = mu.shape[0]
    res_norm = torch.zeros(L); cos_mu = torch.zeros(L)
    ceil_raw = torch.zeros(L); ceil_rec = torch.zeros(L)
    m = 0
    for f in names[:args.stats_files]:
        try:
            p, q = _load(os.path.join(args.pos_dir, f)), _load(os.path.join(args.neg_dir, f))
        except Exception:
            continue
        if p.shape != q.shape or p.dim() != 3:
            continue
        r = p - q                                                     # [L, T, D]
        res_norm += r.norm(dim=-1).mean(dim=1)
        cos_mu += F.cosine_similarity(r, mu[:, None, :].expand_as(r), dim=-1).mean(dim=1)
        ceil_raw += (1 - F.cosine_similarity(p, q, dim=-1)).mean(dim=1)
        pr = p - mu[:, None, :]
        ceil_rec += (F.cosine_similarity(p, pr, dim=-1) - F.cosine_similarity(p, q, dim=-1)).mean(dim=1)
        m += 1
    if m:
        res_norm /= m; cos_mu /= m; ceil_raw /= m; ceil_rec /= m
        mu_norm = mu.norm(dim=-1)
        print(f"\nper-kept-layer diagnostics over {m} files:")
        print(f"  {'layer':>5} {'||mu||':>8} {'E||r||':>8} {'shared':>7} {'cos(r,mu)':>10} "
              f"{'ceil_raw':>9} {'ceil_rec':>9}")
        for l in range(L):
            print(f"  {l:>5} {mu_norm[l]:8.3f} {res_norm[l]:8.3f} {mu_norm[l] / res_norm[l]:7.3f} "
                  f"{cos_mu[l]:10.3f} {ceil_raw[l]:9.4f} {ceil_rec[l]:9.4f}")
        print(f"\n  mean ceiling  plain={ceil_raw.mean():.4f}  recentered={ceil_rec.mean():.4f}")
        print("  set --obs_residual_margin at or a bit below the recentered ceiling.")


if __name__ == "__main__":
    main()
