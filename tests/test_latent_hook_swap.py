"""do(Z) `swap` mode: each latent is replaced by a REAL latent from a different
sample, slot-aligned, walking the pool -- the intervention stage-3 L_swap trains
against, so do(Z) can measure what was trained (mean/gauss are off-manifold)."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference", "vllm"))
import monet_latent_hook as mlh  # noqa: E402


def _pool(n_samples, k, h):
    # sample s, slot j  ->  vector filled with s + j/10, so identity is readable
    return torch.stack([torch.full((h,), s + j / 10) for s in range(n_samples) for j in range(k)])


def test_donor_shift_is_multiple_of_k_and_nonzero():
    assert mlh.MonetLatentHook._donor_shift(80, 8) == 40
    assert mlh.MonetLatentHook._donor_shift(81, 8) == 40
    assert mlh.MonetLatentHook._donor_shift(10, 8) == 8      # n//2//k == 0 -> fall back to k
    assert mlh.MonetLatentHook._donor_shift(3, 8) == 2       # tiny pool: never 0, never >= n


def test_swap_returns_other_samples_latent_at_same_slot(tmp_path, monkeypatch):
    k, h, n_s = 4, 6, 10
    pool_path = str(tmp_path / "pool.pt")
    torch.save(_pool(n_s, k, h), pool_path)
    monkeypatch.setenv("MONET_LATENT_POOL", pool_path)
    monkeypatch.setenv("LATENT_SIZE", str(k))
    hook = mlh.MonetLatentHook(mode="swap")
    assert hook.active

    # generation of sample 0, slots 0..3 (hidden = the sample's own latent)
    for j in range(k):
        own = torch.full((h,), 0 + j / 10, dtype=torch.bfloat16)
        out = hook.process(own)
        assert out.dtype == own.dtype and out.shape == own.shape
        s, slot = divmod(float(out[0]) * 10, 10)          # decode (sample, slot)
        assert round(slot) == j                            # slot-aligned
        assert int(s) != 0                                 # different sample
    # the walk continues into a different donor for the next sample
    nxt = hook.process(torch.full((h,), 1.0))
    assert int(float(nxt[0])) not in (0, 1) and abs(float(nxt[0]) % 1) < 1e-6


def test_swap_requires_pool(monkeypatch):
    monkeypatch.delenv("MONET_LATENT_POOL", raising=False)
    monkeypatch.delenv("MONET_LATENT_DUMP", raising=False)
    with pytest.raises(FileNotFoundError):
        mlh.MonetLatentHook(mode="swap")


def test_other_modes_unchanged():
    assert mlh.MonetLatentHook(mode="off").process(torch.ones(3)).sum() == 3
    assert mlh._VALID_MODES == {"off", "capture", "corrupt_mean", "corrupt_gauss", "swap"}
