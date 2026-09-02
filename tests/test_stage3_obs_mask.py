"""Stage-3 observation masking.

Gate 15480729 diagnosis: the Stage-3 checkpoint produced the best latents this
project has made (effective_rank 98.1, cross_sample_sim 0.501) and the LM read
NOTHING from them (content gap +0.0002, obs NLL 0.090, presence gap only +0.171).
Cause: Stage 3 builds no attention mask, so observation tokens attend the QUESTION
IMAGE directly and never need the latents. Masked Stage 2 gets presence +0.722 and
content +0.0175 on the same measurement.

These tests pin the mask semantics without a model: observation rows must lose the
image columns while KEEPING the latent columns (the latents are the replacement
visual route, not another thing to block).
"""
import torch

from src.utils import build_4d_attn_wo_helper_images

# token ids used to synthesise a sequence
V_START, IMG_PAD, V_END = 100, 101, 102
ABS_START, ABS_PAD, ABS_END = 200, 201, 202
OBS_START, OBS_END = 300, 301
ANS_START = 400
TOKENS = {"v_start": V_START, "img_pad": IMG_PAD, "v_end": V_END,
          "abs_start": ABS_START, "abs_pad": ABS_PAD, "abs_end": ABS_END,
          "obs_start": OBS_START, "obs_end": OBS_END, "ans_start": ANS_START}


def _seq():
    """[img][question][ans_start][latents][observation][answer] — Stage-3 shape."""
    ids = ([V_START, IMG_PAD, IMG_PAD, V_END]          # 0-3   question image
           + [50, 51]                                   # 4-5   question text
           + [ANS_START]                                # 6     assistant turn
           + [ABS_START, ABS_PAD, ABS_PAD, ABS_END]     # 7-10  latent block
           + [OBS_START, 60, 61, OBS_END]               # 11-14 observation
           + [70, 71])                                  # 15-16 answer
    return torch.tensor([ids]), torch.ones(1, len(ids), dtype=torch.long)


IMG_COLS = [0, 1, 2, 3]
LATENT_COLS = [8, 9]
OBS_ROWS = [12, 13]          # observation content tokens


def test_unmasked_observation_can_see_the_question_image():
    """The pre-fix behaviour, pinned so a regression is visible."""
    ids, pad = _seq()
    a = build_4d_attn_wo_helper_images(ids, pad, TOKENS)[0, 0]
    assert a[OBS_ROWS[0], IMG_COLS].any(), "baseline should allow obs->image"


def test_observation_loses_the_image_but_keeps_the_latents():
    ids, pad = _seq()
    a = build_4d_attn_wo_helper_images(
        ids, pad, TOKENS, observation_tokens_cannot_see_question_image=True)[0, 0]
    for r in OBS_ROWS:
        assert not a[r, IMG_COLS].any(), f"obs row {r} can still see the question image"
        assert a[r, LATENT_COLS].all(), f"obs row {r} lost the latents — the replacement route"


def test_answer_rows_are_untouched():
    """Scope the rule to observations, as Stage 2 does; the answer path is the
    accuracy path and must keep the image."""
    ids, pad = _seq()
    a = build_4d_attn_wo_helper_images(
        ids, pad, TOKENS, observation_tokens_cannot_see_question_image=True)[0, 0]
    assert a[15, IMG_COLS].any(), "answer rows must still see the question image"


def test_causality_and_padding_are_preserved():
    ids, pad = _seq()
    a = build_4d_attn_wo_helper_images(
        ids, pad, TOKENS, observation_tokens_cannot_see_question_image=True)[0, 0]
    L = ids.shape[1]
    upper = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    assert not (a & upper).any(), "mask leaked future positions"


def test_flag_off_is_bit_identical_to_before():
    ids, pad = _seq()
    off = build_4d_attn_wo_helper_images(ids, pad, TOKENS)
    default = build_4d_attn_wo_helper_images(
        ids, pad, TOKENS, observation_tokens_cannot_see_question_image=False)
    assert torch.equal(off, default)
