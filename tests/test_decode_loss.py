"""Correctness tests for L_dec (decode-CE writer loss).

The one subtle risk in wiring L_dec into Stage 3 is the gradient path: the loss
must flow back INTO the latents (so it shapes what the writer produces), and the
decode_gap tripwire must actually distinguish real vs shuffled latents. These
tests pin both without a GPU or the full model.
"""
import torch

from src.train.decode_loss import LatentObsDecoder


def _decoder(vocab=64, d=32, K=8):
    torch.manual_seed(0)
    return LatentObsDecoder(vocab_size=vocab, d_latent=d, d_dec=32,
                            n_layers=2, n_heads=4, max_len=32, pad_id=0)


def test_gradient_flows_into_latents():
    """L_dec must produce a non-zero gradient on the latent block itself."""
    dec = _decoder()
    z = torch.randn(1, 8, 32, requires_grad=True)
    tgt = torch.randint(1, 64, (1, 12))
    loss = dec(z, tgt)
    loss.backward()
    assert z.grad is not None and z.grad.abs().sum() > 0, \
        "L_dec did not backprop into the latents -- writer loss would be a no-op"


def test_loss_is_finite_and_positive():
    dec = _decoder()
    z = torch.randn(1, 8, 32)
    tgt = torch.randint(1, 64, (1, 10))
    loss = dec(z, tgt)
    assert torch.isfinite(loss) and loss.item() > 0


def test_decode_gap_rewards_content():
    """A decoder trained to map distinct latents -> distinct targets must score
    real latents strictly better than shuffled ones (positive decode_gap)."""
    torch.manual_seed(1)
    dec = _decoder()
    # two samples with clearly different latents and different targets
    z = torch.randn(2, 8, 32)
    tgt = torch.stack([torch.randint(1, 64, (10,)), torch.randint(1, 64, (10,))])
    opt = torch.optim.Adam(dec.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        loss = dec(z, tgt)
        loss.backward()
        opt.step()
    gap = dec.decode_gap(z, tgt)   # L(shuffled) - L(real)
    assert gap > 0.1, f"decode_gap={gap:.3f}; decoder should use Z after fitting"


def test_shuffle_is_deterministic_at_b2():
    """decode_gap uses a batch roll (no RNG) so it is stable at B=2."""
    dec = _decoder()
    z = torch.randn(2, 8, 32)
    tgt = torch.randint(1, 64, (2, 8))
    assert dec.decode_gap(z, tgt) == dec.decode_gap(z, tgt)


def test_pooled_redundant_block_cannot_decode_two_sentences():
    """The design claim: a block whose K slots are identical (within-block
    collapse, sim=1.0) has less capacity than a differentiated one. After equal
    fitting, the redundant block should reach a HIGHER (worse) decode loss when
    the two targets are long/distinct -- i.e. L_dec penalises the 0.92 redundancy
    that pooled InfoNCE permits."""
    torch.manual_seed(2)
    tgt = torch.stack([torch.randint(1, 64, (16,)), torch.randint(1, 64, (16,))])

    def fit(z):
        d = _decoder()
        opt = torch.optim.Adam(d.parameters(), lr=1e-2)
        for _ in range(400):
            opt.zero_grad(); l = d(z, tgt); l.backward(); opt.step()
        return d(z, tgt).item()

    base = torch.randn(2, 1, 32)
    redundant = base.repeat(1, 8, 1)                       # 8 identical slots
    differentiated = base + 0.5 * torch.randn(2, 8, 32)    # slots differ
    assert fit(redundant) > fit(differentiated), \
        "redundant block should decode worse -- else L_dec doesn't reward slot diversity"
