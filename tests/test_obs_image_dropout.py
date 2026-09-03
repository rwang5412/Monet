"""--obs_image_dropout: L_swap gap bucketed by image visibility.

The stage-3 collator blinds observation tokens to the question image with
probability p per step and records the outcome in batch['obs_image_masked'].
The trainer must (a) pop that key so it never reaches the model forward,
(b) bucket the swap gap into visible/masked, and (c) emit swap_gap_visible --
the only number that predicts a do(Z) effect at an inference where the image
is always present. These tests pin the trainer side without a model.
"""
import torch

from src.train import trainer_stage3_decode as T
from src.train.trainer_stage3_decode import CustomTrainerSFT_STAGE3_Decode


class _Stub(CustomTrainerSFT_STAGE3_Decode):
    """Skip the heavy Trainer __init__; set only what log() reads."""
    def __init__(self, decode_weight=0.0, swap_weight=1.0):
        self.decode_weight = decode_weight
        self.swap_weight = swap_weight
        self._dec_loss_cum = self._dec_gap_cum = 0.0
        self._dec_steps = self._dec_gap_donor = 0
        self._swap_loss_cum = self._swap_gap_cum = 0.0
        self._swap_steps = 0
        self._swap_gap_vis_cum = 0.0; self._swap_vis_steps = 0
        self._swap_gap_msk_cum = 0.0; self._swap_msk_steps = 0
        self._obs_masked_steps = 0
        self._log_window = 0


def _capture_parent_log(monkeypatch):
    """super().log(...) resolves to the parent class; make it return `merged`."""
    monkeypatch.setattr(T.CustomTrainerSFT_STAGE3, "log",
                        lambda self, logs, start_time=None: logs, raising=False)


def test_bucketed_keys_absent_when_dropout_not_in_use(monkeypatch):
    _capture_parent_log(monkeypatch)
    s = _Stub()
    s._swap_gap_cum, s._swap_steps, s._log_window = 0.3, 3, 10
    out = s.log({})
    assert "swap_gap" in out
    assert "swap_gap_visible" not in out and "obs_masked_frac" not in out


def test_bucketed_gaps_and_masked_frac_are_emitted(monkeypatch):
    _capture_parent_log(monkeypatch)
    s = _Stub()
    s._log_window = 10
    s._obs_masked_steps = 4                               # 4 of 10 steps blinded
    s._swap_gap_vis_cum, s._swap_vis_steps = 0.06, 6      # image visible: small gap
    s._swap_gap_msk_cum, s._swap_msk_steps = 0.80, 4      # image masked: large gap
    s._swap_gap_cum, s._swap_steps = 0.86, 10
    out = s.log({})
    assert out["swap_gap_visible"] == 0.01
    assert out["swap_gap_masked"] == 0.2
    assert out["obs_masked_frac"] == 0.4
    # window counters reset after emission
    assert s._swap_vis_steps == 0 and s._swap_msk_steps == 0 and s._obs_masked_steps == 0


def test_visible_bucket_reads_zero_not_nan_when_every_step_was_masked(monkeypatch):
    """p=1.0 (== the always-on flag): no visible steps. Must not divide by zero."""
    _capture_parent_log(monkeypatch)
    s = _Stub()
    s._log_window = 5
    s._obs_masked_steps = 5
    s._swap_gap_msk_cum, s._swap_msk_steps = 1.0, 5
    out = s.log({})
    assert out["swap_gap_visible"] == 0.0 and out["swap_gap_masked"] == 0.2
    assert out["obs_masked_frac"] == 1.0
