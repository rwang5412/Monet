"""Span selection for L_swap (--swap_span).

The original design supervised L_swap on OBSERVATION tokens only, on the grounds
that teacher-forced answer NLL (~0.11) is too small to yield gradient. That
reasoning is wrong: nll_real is detached and the gradient flows through
nll_donor, which is free to rise however confident the model is. Since do(Z) is
an ANSWER-level measurement, being able to supervise the answer span directly is
the escalation lever. These tests pin the selection logic without a model.
"""
import torch

from src.train.trainer_stage3_decode import CustomTrainerSFT_STAGE3_Decode


class _Stub(CustomTrainerSFT_STAGE3_Decode):
    """Bypass the heavy Trainer __init__; only the span helpers are exercised."""
    def __init__(self, swap_span="obs"):
        self.swap_span = swap_span


def _inputs(seq_len=20, obs=(5, 6, 7), ans=(12, 13, 14)):
    labels = torch.full((1, seq_len), -100)
    for p in list(obs) + list(ans):
        labels[0, p] = 1
    return {
        'student_input_ids': torch.arange(seq_len).unsqueeze(0),
        'student_labels': labels,
        'observation_poss': [list(obs)],
    }


def test_answer_span_excludes_observation_tokens():
    inp = _inputs()
    ans = _Stub()._answer_positions(inp)
    assert ans == [12, 13, 14], ans
    assert not (set(ans) & {5, 6, 7}), "answer span must not include observation tokens"


def test_swap_span_selects_the_right_positions():
    inp = _inputs()
    obs_pos = _Stub()._obs_positions(inp)
    assert _Stub("obs")._swap_positions(inp, obs_pos) == [5, 6, 7]
    assert _Stub("answer")._swap_positions(inp, obs_pos) == [12, 13, 14]
    assert _Stub("both")._swap_positions(inp, obs_pos) == [5, 6, 7, 12, 13, 14]


def test_default_is_unchanged_behavior():
    """Default must stay 'obs' so existing runs are bit-identical."""
    inp = _inputs()
    obs_pos = _Stub()._obs_positions(inp)
    assert _Stub()._swap_positions(inp, obs_pos) == obs_pos


def test_no_labels_gives_none_not_crash():
    inp = _inputs()
    inp.pop('student_labels')
    assert _Stub("answer")._answer_positions(inp) is None
