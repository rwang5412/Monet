"""Latent-collapse probe: measure how much Monet's captured latents actually vary
across examples, to decide whether a swap/interchange loss has any raw material.

Feed it the [N, H] dump written by MonetLatentHook (MONET_LATENT_DUMP set during a
capture pass). It reports, over the N captured latent vectors:

  effective rank (participation ratio)  -- (Σλ)^2 / Σλ^2 of the covariance eigs.
      ~1-3 of H => heavily collapsed (latents live in a tiny subspace);
      hundreds  => diverse, high-dimensional.
  top-1 PC variance fraction            -- 1 PC explaining ~everything => collapsed.
  pairwise cosine similarity            -- mean ~1.0 => all latents point the same
      way (collapsed); spread out / ~0 => diverse directions.
  swap vs mean distance                 -- ||Z_A - Z_B|| (what a SWAP loss injects)
      vs ||Z - mu|| (what corrupt_mean injects). If swap distance is comparable to
      the mean-deviation, swapping is a real perturbation and the loss has signal.
      If swap distance << mean-deviation, examples are clustered -> swap is weak.

    python -m evaluation.capimagine.vlmevalkit.probe_collapse \
      --latents /scratch/$USER/results/vlmeval_doz/Monet-7B-readme_VStarBench_latents.pt
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True, help="[N, H] .pt from MONET_LATENT_DUMP")
    ap.add_argument("--pairs", type=int, default=20000, help="random pairs for cosine/swap")
    args = ap.parse_args()

    X = torch.load(args.latents, map_location="cpu").float()  # [N, H]
    N, H = X.shape
    mu = X.mean(0)
    Xc = X - mu

    # covariance spectrum -> effective rank
    cov = (Xc.T @ Xc) / max(N - 1, 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    total = eig.sum().item()
    eff_rank = ((eig.sum() ** 2) / (eig ** 2).sum()).item() if (eig ** 2).sum() > 0 else 0.0
    top1 = (eig.max() / eig.sum()).item() if total > 0 else float("nan")
    # variance captured by top-k PCs
    ev = torch.sort(eig, descending=True).values
    cum = torch.cumsum(ev, 0) / max(total, 1e-9)
    k90 = int((cum < 0.90).sum().item()) + 1

    # pairwise cosine + swap distance vs mean-deviation
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, N, (args.pairs, 2), generator=g)
    a, b = X[idx[:, 0]], X[idx[:, 1]]
    Xn = X / X.norm(dim=1, keepdim=True).clamp_min(1e-8)
    cos = (Xn[idx[:, 0]] * Xn[idx[:, 1]]).sum(1)
    swap_dist = (a - b).norm(dim=1).mean().item()          # SWAP magnitude
    mean_dev = Xc.norm(dim=1).mean().item()                # corrupt_mean magnitude
    sigma_scale = eig.sqrt().mean().item()

    W = 64
    print("=" * W)
    print(f"  Latent-collapse probe  |  {args.latents.split('/')[-1]}")
    print("=" * W)
    print(f"  N latents = {N}   H = {H}")
    print(f"  effective rank (participation ratio) = {eff_rank:.1f} / {H}")
    print(f"  top-1 PC variance fraction           = {top1:.3f}")
    print(f"  PCs to reach 90% variance            = {k90} / {H}")
    print(f"  pairwise cosine sim: mean={cos.mean():.3f} "
          f"sd={cos.std():.3f} p50={cos.median():.3f} p95={cos.quantile(0.95):.3f}")
    print(f"  swap distance ||Z_A - Z_B||          = {swap_dist:.3f}")
    print(f"  mean-deviation ||Z - mu|| (corrupt_mean) = {mean_dev:.3f}")
    print(f"  ratio swap/mean-dev                  = {swap_dist / max(mean_dev, 1e-9):.3f}")
    print("  " + "-" * (W - 4))
    # crude verdict
    if eff_rank < 5 or top1 > 0.8 or cos.mean() > 0.95:
        v = "COLLAPSED — latents ~indistinguishable; swap loss has little/no signal"
    elif eff_rank > 0.1 * H and cos.mean() < 0.6:
        v = "DIVERSE — latents vary a lot; swap has raw material (problem is routing)"
    else:
        v = "PARTIAL collapse — anisotropic; swap has some but limited signal"
    print(f"  verdict: {v}")
    print("=" * W)


if __name__ == "__main__":
    main()
