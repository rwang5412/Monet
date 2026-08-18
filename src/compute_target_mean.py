"""Compute the dataset-mean target latent for recentered Stage-3 alignment (mod A).

Plain cosine alignment to the Stage-2 targets is satisfiable by the shared mean
direction (~84% of each vector), so it copies the mean and garbles the thin
content subspace. Recentering subtracts this mean from the target before the
cosine, so the alignment budget goes to the content subspace.

Reads every harvested target latent file, averages per layer, writes mean.pt
([num_layer, dim] or [dim]). Cheap CPU job over the target dir.

    python -m src.compute_target_mean \
      --target_dir /scratch/$USER/monet_ckpts/teacher_latents_modified \
      --out /scratch/$USER/monet_ckpts/teacher_latents_modified/align_mean.pt
"""
import argparse
import glob
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_files", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.target_dir, "*.pt")))
    files = [f for f in files if not f.endswith("align_mean.pt")]
    if args.max_files > 0:
        files = files[:args.max_files]
    assert files, f"no target .pt files in {args.target_dir}"

    acc = None      # [num_layer, dim] running sum of per-(file,slot) means
    n = 0
    for i, f in enumerate(files):
        try:
            lat = torch.load(f, map_location="cpu")["latent"].float()   # [L,K,d] or [K,d]
        except Exception as e:
            print(f"skip {f}: {e!r}")
            continue
        if lat.dim() == 2:              # [K, d] -> treat as single "layer"
            lat = lat.unsqueeze(0)
        m = lat.mean(dim=1)             # [L, d] mean over the K slots
        acc = m if acc is None else acc + m
        n += 1
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(files)} files")
    mean = acc / max(n, 1)             # [L, d]
    out = args.out or os.path.join(args.target_dir, "align_mean.pt")
    torch.save({"mean": mean, "n_files": n, "shape": list(mean.shape)}, out)
    print(f"wrote mean {tuple(mean.shape)} over {n} files -> {out}")
    # sanity: how much of the target is the shared mean (per-layer norm ratio)
    print(f"mean per-layer L2 norm: {mean.norm(dim=-1).mean().item():.3f}")


if __name__ == "__main__":
    main()
