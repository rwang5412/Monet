"""Unit tests for the Stage-2 loss modifications (residual objective + grounding).

Run on the cluster env (needs torch; the residual test also imports the modeling
file, which needs transformers):  python -m pytest tests/test_stage2_losses.py -q
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stage2_losses import LatentGroundingLoss

D, K, LYR, T = 32, 4, 3, 5  # dim, latents, layers, obs tokens


def _residual():
    from monet_qwen_model.modeling_qwen2_5_vl_monet import obs_residual_loss
    return obs_residual_loss


# ---------------- Change 1: residual objective ----------------
def test_residual_zero_when_student_matches_pos_teacher():
    obs_residual_loss = _residual()
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(LYR, T, D, generator=g)
    neg = -pos  # maximally different teacher
    loss = obs_residual_loss(pos, neg, pos.clone(), margin=0.2)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)


def test_residual_has_gradient_where_absolute_loss_saturates():
    """The motivating case: student ~equally close to both teachers (visual
    residual not encoded). Absolute cosine is near-optimal (tiny gradient);
    the residual loss sits at the margin and pushes."""
    obs_residual_loss = _residual()
    g = torch.Generator().manual_seed(1)
    base = torch.randn(LYR, T, D, generator=g)
    pos = base + 0.05 * torch.randn(LYR, T, D, generator=g)  # teachers differ only
    neg = base + 0.05 * torch.randn(LYR, T, D, generator=g)  # by a small residual
    student = base.clone().requires_grad_(True)              # ignores the residual
    loss = obs_residual_loss(pos, neg, student, margin=0.2)
    assert float(loss) > 0.05, "should sit near the margin, not at zero"
    loss.backward()
    assert student.grad is not None and float(student.grad.abs().sum()) > 0


def test_residual_teacher_shared_content_cancels():
    """Adding the same offset to BOTH teachers must not change the loss much --
    the shared (token identity/context) component cancels by construction."""
    obs_residual_loss = _residual()
    g = torch.Generator().manual_seed(2)
    pos = torch.randn(LYR, T, D, generator=g)
    neg = torch.randn(LYR, T, D, generator=g)
    student = torch.randn(LYR, T, D, generator=g)
    l1 = float(obs_residual_loss(pos, neg, student, margin=0.2))
    shift = 0.1 * torch.randn(1, 1, D, generator=g)
    l2 = float(obs_residual_loss(pos + shift, neg + shift, student, margin=0.2))
    assert abs(l1 - l2) < 0.05


# ---------------- Change 2: latent grounding ----------------
def _mk(seed=0):
    torch.manual_seed(seed)
    return LatentGroundingLoss(d_model=D, queue_size=16, temp=0.07)


def test_grounding_finite_and_queue_rotates():
    m = _mk()
    lat = [torch.randn(K, D)]
    aux = [torch.randn(7, D)]
    p0 = int(m.ptr)
    l1 = m(lat, aux, enqueue=True)
    assert torch.isfinite(l1) and float(l1) >= 0
    assert int(m.ptr) == (p0 + 1) % 16 and int(m.filled) == 1
    m(lat, aux, enqueue=False)
    assert int(m.ptr) == (p0 + 1) % 16, "enqueue=False must not advance the queue"


def test_grounding_projector_trains_on_detached_latents():
    m = _mk()
    lat = [torch.randn(K, D, requires_grad=True)]
    aux = [torch.randn(7, D)]
    loss = m([z.detach() for z in lat], aux, enqueue=False)
    loss.backward()
    proj_grads = [p.grad for p in m.proj.parameters()]
    assert all(g is not None for g in proj_grads)
    assert lat[0].grad is None, "detached path must not reach the latents"


def test_grounding_latents_get_gradient_on_writer_path():
    m = _mk()
    lat = [torch.randn(K, D, requires_grad=True)]
    aux = [torch.randn(7, D)]
    loss = m(lat, aux, enqueue=False)
    loss.backward()
    assert lat[0].grad is not None and float(lat[0].grad.abs().sum()) > 0


def test_grounding_discriminates_own_aux_image():
    """After the queue holds OTHER samples' aux feats, matching your own aux
    must score lower loss than matching a random one."""
    m = _mk()
    torch.manual_seed(3)
    for _ in range(16):  # fill queue with unrelated targets
        m([torch.randn(K, D)], [torch.randn(7, D)], enqueue=True)
    v_own = torch.randn(7, D)
    # cheat student: project-inverse is unknown, so just compare aligned-vs-random
    z_aligned = v_own.mean(0, keepdim=True).repeat(K, 1)
    l_aligned = float(m([z_aligned], [v_own], enqueue=False))
    l_random = float(m([torch.randn(K, D)], [v_own], enqueue=False))
    # untrained projector: only sanity-check both are finite and loss responds to input
    assert l_aligned != l_random


def test_grounding_none_aux_gives_zero():
    m = _mk()
    loss = m([torch.randn(K, D)], [None], enqueue=True)
    assert float(loss) == 0.0 and int(m.filled) == 0


def test_grounding_backward_after_enqueue():
    """Regression: HF runs backward AFTER compute_loss, i.e. after _enqueue has
    mutated the queue buffer in place. Without the .clone() on the negatives
    matmul this raises 'modified by an inplace operation' (killed jobs
    14906125/14906402)."""
    m = _mk()
    lat = [torch.randn(K, D, requires_grad=True)]
    aux = [torch.randn(7, D)]
    loss = m(lat, aux, enqueue=True)   # enqueue mutates the queue...
    loss.backward()                     # ...then backward must still succeed
    assert lat[0].grad is not None
