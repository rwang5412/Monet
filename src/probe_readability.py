"""Pre-run diagnostic: is the latents' content LINEARLY (LM-)readable, or only
recoverable by a nonlinear projector?

The Stage-2 gate showed content-inert: nce_top1 = 0.77 proves a TRAINED MLP
projector can retrieve the aux image from a pooled latent block, but it does NOT
prove the content sits in a direction the LM's own (roughly linear, per-head)
attention can reach. This partitions that:

  fit a LINEAR map and a small MLP from pooled latent -> aux-image features on a
  held-out fit split; measure retrieval top-1 on a disjoint eval split.

    linear ~= MLP, both >> chance  -> content is LINEARLY accessible; decode-CE
                                      has an easy target, the reader just doesn't
                                      attend. (Reader problem.)
    linear << MLP                  -> content lives in a nonlinear code only the
                                      projector inverts; the LM has no machinery
                                      to decode it. Representational mismatch --
                                      decode-CE (text target) is doing the heavy
                                      lifting of re-coding, not just re-attending.

Inference only + closed-form ridge + a tiny MLP fit. One GPU, ~1-2h.

    python -m src.probe_readability \
      --student /scratch/$USER/monet_ckpts/sft_stage2_residual0.05_ground1.0_latent8 \
      --data_path .../Visual_CoT/train.json --dataset_root ... \
      --latent_size 8 --n 300 --n_fit 200
"""
import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F


def _retrieval_top1(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of rows whose predicted vector's nearest (cosine) target is its own."""
    p = F.normalize(pred.float(), dim=-1)
    t = F.normalize(target.float(), dim=-1)
    sim = p @ t.T                                  # [N, N]
    return float((sim.argmax(dim=1) == torch.arange(sim.shape[0])).float().mean())


def _ridge_fit(X: torch.Tensor, Y: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """Closed-form linear map W minimizing ||XW - Y||^2 + lam||W||^2."""
    d = X.shape[1]
    A = X.T @ X + lam * torch.eye(d, dtype=X.dtype)
    return torch.linalg.solve(A, X.T @ Y)          # [d_in, d_out]


def _mlp_fit(X, Y, Xe, Ye, hidden=2048, epochs=400, lr=1e-3):
    """Small MLP z->aux; returns eval-set predictions."""
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.GELU(),
                        nn.Linear(hidden, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(net(X), Y)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return net(Xe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--latent_size", type=int, default=8)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n_fit", type=int, default=200)
    ap.add_argument("--out", default="probe_readability.json")
    args = ap.parse_args()

    import monet_qwen_model.apply_qwen2_5_monet  # noqa: F401
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from src.gate_stage2_collate import build_student_stage2_batch

    rows = json.load(open(args.data_path))[-args.n:]
    dev = "cuda"
    processor = AutoProcessor.from_pretrained(args.student, use_fast=True, trust_remote_code=True)
    for t in ("<abs_vis_token_pad>", "<abs_vis_token>", "</abs_vis_token>",
              "<observation>", "</observation>"):
        processor.tokenizer.add_tokens(t, special_tokens=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.student, torch_dtype=torch.bfloat16).to(dev).eval()

    z_pool, aux_pool, within = [], [], []
    with torch.inference_mode():
        for i, row in enumerate(rows):
            try:
                inp = build_student_stage2_batch(row, processor, args.dataset_root,
                                                 args.latent_size, device=dev)
                lat = model(**inp, latent_mode=True, loss_type=[], return_dict=True,
                            output_hidden_states=False)
                z = lat.ce_patch_vec[0].float()               # [K, d]
                aux = getattr(lat, 'aux_image_feats', None)
                if aux is None or aux[0] is None:
                    continue
                a = aux[0].float()                            # [M, d]
                z_pool.append(z.mean(0).cpu())
                aux_pool.append(a.mean(0).cpu())
                zn = F.normalize(z, dim=-1)
                C = zn @ zn.T
                k = C.shape[0]
                within.append(float((C.sum() - k) / (k * (k - 1))))
            except Exception as e:
                if i < 5:
                    print(f"[probe] skip {i}: {e!r}")

    Z = torch.stack(z_pool)                                   # [N, d]
    A = torch.stack(aux_pool)                                 # [N, d]
    N = Z.shape[0]
    nf = min(args.n_fit, N - 20)
    Zf, Ze = Z[:nf], Z[nf:]
    Af, Ae = A[:nf], A[nf:]
    # standardize inputs (helps both fits)
    mu, sd = Zf.mean(0, keepdim=True), Zf.std(0, keepdim=True) + 1e-6
    Zf, Ze = (Zf - mu) / sd, (Ze - mu) / sd

    W = _ridge_fit(Zf, Af, lam=10.0)
    lin_pred = Ze @ W
    lin_top1 = _retrieval_top1(lin_pred, Ae)
    mlp_pred = _mlp_fit(Zf, Af, Ze, Ae)
    mlp_top1 = _retrieval_top1(mlp_pred, Ae)
    chance = 1.0 / Ae.shape[0]

    report = {
        "n_scored": N, "n_eval": Ae.shape[0],
        "linear_top1": lin_top1, "mlp_top1": mlp_top1, "chance": chance,
        "within_block_sim": sum(within) / max(len(within), 1),
        "linear_over_mlp": (lin_top1 / mlp_top1) if mlp_top1 > 0 else 0.0,
    }
    json.dump(report, open(args.out, "w"), indent=2)
    W_ = 64
    print("=" * W_)
    print("  Stage-2 latent readability probe  (aux-image retrieval from latents)")
    print("=" * W_)
    print(f"  n eval / chance         = {Ae.shape[0]} / {chance:.4f}")
    print(f"  LINEAR ridge  top-1     = {lin_top1:.4f}")
    print(f"  MLP (nonlinear) top-1   = {mlp_top1:.4f}")
    print(f"  linear / MLP ratio      = {report['linear_over_mlp']:.3f}")
    print(f"  within-block sim        = {report['within_block_sim']:.4f}")
    if lin_top1 >= 0.8 * mlp_top1 and lin_top1 > 5 * chance:
        print("  READ: content is LINEARLY accessible -> reader problem;")
        print("        decode-CE has an easy target (re-attend, don't re-code).")
    elif mlp_top1 > 5 * chance and lin_top1 < 0.5 * mlp_top1:
        print("  READ: content is NONLINEAR-only -> representational mismatch;")
        print("        decode-CE's text target must RE-CODE, not just re-attend.")
    else:
        print("  READ: inconclusive / weak retrieval both ways -- inspect.")
    print("=" * W_)


if __name__ == "__main__":
    main()
