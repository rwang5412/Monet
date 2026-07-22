"""Stage-4 unit tests (design doc §6.3). Pure-torch; no GPU or checkpoint needed.

Run:  python -m pytest tests/test_stage4.py -q
"""
import inspect
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train.decode_loss import LatentObsDecoder
from src.train.nce_loss import NegativeBank, info_nce, pool_latents
from src.train.necessity_loss import DonorBank, necessity_hinge
from src.train.span_nll import (find_answer_span, find_subsequence,
                                nll_on_positions, span_ce_to_targets)
from src.train.swap_loss import swap_loss
from src.train.trainer_stage4 import CustomTrainerSFT_STAGE4

V, H, K, L = 500, 64, 8, 40


def _logits(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(L, V, generator=g)


# 1. all losses finite and >= 0 -------------------------------------------------
def test_losses_finite_nonnegative():
    dec = LatentObsDecoder(vocab_size=V, d_latent=H, d_dec=32, n_layers=1, n_heads=4)
    z = torch.randn(2, K, H)
    tgt = torch.randint(1, V, (2, 10))
    l_dec = dec(z, tgt)
    l_nce = info_nce(torch.randn(2, H), torch.randn(2, H))
    ls, lo, la = swap_loss(_logits(), [5, 6], [11, 12], [20, 21], [31, 32])
    l_nec = necessity_hinge(torch.tensor(1.0), torch.tensor(1.2), margin=1.0)
    for l in (l_dec, l_nce, ls, l_nec):
        assert torch.isfinite(torch.as_tensor(l)).all() and float(l) >= 0.0


# 2. hinge: nll_with detached, ablated branch carries gradient ------------------
def test_hinge_gradient_flow():
    nll_with = torch.tensor(1.0, requires_grad=True)
    nll_ablated = torch.tensor(1.2, requires_grad=True)
    loss = necessity_hinge(nll_with, nll_ablated, margin=1.0)
    loss.backward()
    assert nll_ablated.grad is not None and float(nll_ablated.grad) != 0.0
    assert nll_with.grad is None or float(nll_with.grad) == 0.0


# 3. hinge zero at/beyond margin -----------------------------------------------
def test_hinge_zero_when_gap_clears_margin():
    assert float(necessity_hinge(torch.tensor(1.0), torch.tensor(2.5), 1.0)) == 0.0
    assert float(necessity_hinge(torch.tensor(1.0), torch.tensor(2.0), 1.0)) == 0.0
    assert float(necessity_hinge(torch.tensor(1.0), torch.tensor(1.5), 1.0)) == pytest.approx(0.5)


# 4. spliced supervision touches ONLY the given span rows -----------------------
def test_span_ce_uses_only_span_rows():
    logits = _logits().requires_grad_(True)
    loss = span_ce_to_targets(logits, positions=[10, 11], target_ids=[3, 4])
    loss.backward()
    g = logits.grad.abs().sum(dim=-1)
    touched = set(g.nonzero(as_tuple=False).flatten().tolist())
    assert touched == {9, 10}  # rows p-1 for label positions p


# 5. spans recomputed: mismatch asserts, spliced spans found on re-tokenization -
def test_span_mismatch_asserts():
    with pytest.raises(AssertionError):
        swap_loss(_logits(), [5, 6], [11], [20], [31])       # obs len mismatch
    with pytest.raises(AssertionError):
        swap_loss(_logits(), [], [], [20], [31])             # empty obs span


def test_find_answer_span_roundtrip():
    class TinyTok:
        def encode(self, s, add_special_tokens=False):
            return [ord(c) % V for c in s]
    row = torch.tensor([ord(c) % V for c in "prefix \\boxed{42} tail"])
    span = find_answer_span(row, TinyTok(), "42")
    assert span is not None and len(span) == len("\\boxed{42}")


# 6. decoder can only ever see Z ------------------------------------------------
def test_decoder_input_isolation():
    sig = inspect.signature(LatentObsDecoder.forward)
    assert list(sig.parameters) == ["self", "z", "target_ids"], \
        "decoder must take latents and targets ONLY (no image/question/prefix)"
    # and its output must actually depend on z
    dec = LatentObsDecoder(vocab_size=V, d_latent=H, d_dec=32, n_layers=1, n_heads=4)
    tgt = torch.randint(1, V, (1, 10))
    torch.manual_seed(0)
    a = dec(torch.zeros(1, K, H), tgt).item()
    b = dec(torch.ones(1, K, H) * 3.0, tgt).item()
    assert a != b


# 7. all weights 0 => early exit before ANY RNG or bank access ------------------
def test_zero_weight_early_exit_no_rng_no_banks():
    t = CustomTrainerSFT_STAGE4.__new__(CustomTrainerSFT_STAGE4)  # skip HF init
    t.w_dec = t.w_nce = t.w_swap = t.w_nec = 0.0
    t.margin = 1.0
    t.donor_bank, t.neg_bank = DonorBank(), NegativeBank()
    t._acc = {k: 0.0 for k in ("l_answer",)}
    t._n = {k: 0 for k in ("l_answer",)}
    t._z_ring = []
    t._log_add = lambda k, v: None
    calls = {"latent": 0, "ce": 0}

    def fake_latent(model, batch, no_grad):
        calls["latent"] += 1
        return [1, 2], torch.zeros(2, 4)

    def fake_ce(model, batch, labels, ce_pos=None, ce_vec=None, want_logits=False):
        calls["ce"] += 1
        class Out:  # minimal stand-in
            loss = torch.tensor(1.234, requires_grad=True)
            logits = torch.zeros(1, 8, V)
        return Out()

    t._forward_latent = fake_latent
    t._forward_ce = fake_ce
    rng_before = torch.random.get_rng_state()
    inputs = {"f1": {"student_input_ids": torch.zeros(1, 8, dtype=torch.long),
                     "student_labels": torch.zeros(1, 8, dtype=torch.long)},
              "f1_spans": {"ans_positions": [5], "ans_ids": [7],
                           "obs_positions": [3], "obs_ids": [4]}}
    loss = t.compute_loss(model=None, inputs=inputs)
    assert float(loss) == pytest.approx(1.234)
    assert torch.equal(rng_before, torch.random.get_rng_state()), "arm-0 consumed RNG"
    assert len(t.donor_bank.buf) == 0 and len(t.neg_bank.buf) == 0, "arm-0 touched banks"
    assert calls == {"latent": 1, "ce": 1}, "arm-0 must run exactly F1"


# 8. donor filter excludes same-answer rows -------------------------------------
def test_donor_bank_excludes_same_answer():
    bank = DonorBank()
    bank.add(torch.randn(K, H), "A")
    bank.add(torch.randn(K, H), "B")
    d = bank.sample_different_answer("A")
    assert d is not None
    bank2 = DonorBank()
    bank2.add(torch.randn(K, H), "A")
    assert bank2.sample_different_answer("A") is None, "same-answer donor must be refused"


# bonus: decode_gap is positive when the decoder actually reads Z ---------------
def test_decode_gap_sign_after_overfit():
    torch.manual_seed(0)
    dec = LatentObsDecoder(vocab_size=50, d_latent=8, d_dec=16, n_layers=1, n_heads=2)
    z = torch.randn(2, 4, 8)
    tgt = torch.stack([torch.arange(1, 9), torch.arange(9, 17)])
    opt = torch.optim.Adam(dec.parameters(), lr=3e-3)
    for _ in range(300):
        opt.zero_grad()
        loss = dec(z, tgt)
        loss.backward()
        opt.step()
    assert dec.decode_gap(z, tgt) > 0.5, \
        "after overfitting two rows, shuffling Z must hurt (decoder reads Z)"
