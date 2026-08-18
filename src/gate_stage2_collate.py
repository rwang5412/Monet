"""Single-row collation helpers for the Stage-2 promotion gate.

main.py's collators are unimportable (the file trains at import), so the gate
rebuilds the teacher and student batches here, mirroring:
  * precompute_teacher_reps.collate_fn_precompute_teacher_rep (pos + no-aux)
  * main.collate_fn_sft_stage2 (student: latent pads + 4D mask)
The student mask is built with observation_tokens_cannot_see_question_image=True
-- the gate evaluates checkpoints trained under the modified regime.
"""
import copy
import os
from typing import Tuple

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

from .utils import (add_latent_pad_after_auxiliary_img, build_4d_attn,
                    find_ids_poss, generate_labels_after_multi_token_start,
                    process_multiple_question_img, remove_auxiliary_images,
                    replace_latent_placeholder_with_img_pad,
                    resize_by_token_budget)


def _ids(processor):
    tok = processor.tokenizer
    one = lambda s: tok(s, return_tensors="pt")["input_ids"][0]
    d = {
        "v_start": one("<|vision_start|>"), "v_end": one("<|vision_end|>"),
        "img_pad": one("<|image_pad|>"), "abs_start": one("<abs_vis_token>"),
        "abs_end": one("</abs_vis_token>"), "abs_pad": one("<abs_vis_token_pad>"),
        "obs_start": one("<observation>"), "obs_end": one("</observation>"),
        "ans_start": one("<|im_start|>assistant"), "end_pad": one("<|endoftext|>"),
    }
    return d


def _absolutize(row: dict, root: str) -> list:
    ex = copy.deepcopy(row["data"])
    for msg in ex:
        for item in msg.get("content", []):
            if item.get("type") == "image" and not os.path.isabs(str(item["image"])):
                item["image"] = os.path.join(root, item["image"])
    return ex


def _obs_poss(input_ids, ids):
    """Match the TRAINING/CACHE convention exactly: range(start, end) -- the
    span INCLUDES the <observation> tag token (main.py collates and
    precompute_teacher_reps both do this); excludes </observation>."""
    starts = find_ids_poss(input_ids, ids["ans_start"], ids["obs_start"])
    ends = find_ids_poss(input_ids, ids["ans_start"], ids["obs_end"])
    poss = []
    for s_, e_ in zip(starts[0], ends[0]):
        poss.extend(range(s_, e_))
    return poss


def _teacher_obs_mask(input_ids, pad_mask, obs_poss, ids):
    """Duplicate of precompute_teacher_reps.build_teacher_obs_mask (that file is
    an unimportable script): obs rows blind to question-image spans."""
    row = input_ids[0].cpu()
    L = row.numel()
    allowed = torch.tril(torch.ones((L, L), dtype=torch.bool)).unsqueeze(0)
    valid = pad_mask[0].cpu().bool()
    allowed[0] &= valid.unsqueeze(0)
    allowed[0] &= valid.unsqueeze(1)
    pat = ids["ans_start"]
    k = int(pat.numel())
    ans_start = -1
    for s in range(0, L - k + 1):
        if torch.equal(row[s:s+k], pat):
            ans_start = s
            break
    if ans_start != -1 and obs_poss:
        v_s = torch.nonzero(row == ids["v_start"].item(), as_tuple=False).flatten()
        v_e = torch.nonzero(row == ids["v_end"].item(), as_tuple=False).flatten()
        q_img = []
        for a, b in zip(v_s.tolist(), v_e.tolist()):
            if b < ans_start:
                q_img.extend(range(a, b + 1))
        if q_img:
            o = torch.tensor(obs_poss, dtype=torch.long)
            allowed[0][o.unsqueeze(1), torch.tensor(q_img, dtype=torch.long)] = False
    return allowed.unsqueeze(1)  # [1,1,L,L]


def qtext_masked_4d(student_inp: dict, obs_poss, ans_poss, processor):
    """Discriminator mask: block the observation+answer rows from attending to the
    question-TEXT columns (everything before <|im_start|>assistant that is NOT a
    question-image span and not padding -- i.e. system + user text).

    Returns a NEW attention_mask_4d dict (the base student mask, already blocking
    the question IMAGE, further blocked from question TEXT). Rows keep access to
    the latents, their own prefix, and themselves, so no row is fully masked.
    Used to test: does obs/answer content come from the question text? If NLL
    barely rises under this mask, text was not the content source.
    """
    ids = _ids(processor)
    base = student_inp["attention_mask_4d"]["full_attention"]   # [1,1,L,L] bool, on device
    m = base.clone()
    dev = m.device
    row = student_inp["input_ids"][0].to(dev)
    L = row.numel()
    pat = ids["ans_start"].to(dev)
    k = int(pat.numel())
    ans_start = -1
    for s in range(0, L - k + 1):
        if torch.equal(row[s:s + k], pat):
            ans_start = s
            break
    if ans_start <= 0:
        return {"full_attention": m}   # can't locate assistant turn; no-op
    # question-image columns to EXCLUDE from the text set
    v_s = torch.nonzero(row == ids["v_start"].item(), as_tuple=False).flatten().tolist()
    v_e = torch.nonzero(row == ids["v_end"].item(), as_tuple=False).flatten().tolist()
    img_cols = set()
    for a, b in zip(v_s, v_e):
        if b < ans_start:
            img_cols.update(range(a, b + 1))
    qtext_cols = [p for p in range(ans_start) if p not in img_cols]
    block_rows = sorted(set(list(obs_poss) + list(ans_poss)))
    if qtext_cols and block_rows:
        r = torch.tensor(block_rows, dtype=torch.long, device=dev)
        c = torch.tensor(qtext_cols, dtype=torch.long, device=dev)
        m[0, 0][r.unsqueeze(1), c] = False
    return {"full_attention": m}


def build_teacher_batches(row, processor, root, no_aux: bool, device) -> Tuple[dict, list]:
    ids = _ids(processor)
    example = _absolutize(row, root)
    if no_aux:
        example = remove_auxiliary_images([example])[0]
        t = processor.apply_chat_template(example, tokenize=False)
        parts = t.split("<|im_start|>assistant")
        text = process_multiple_question_img(parts[0])
        for p in parts[1:]:
            text += "<|im_start|>assistant" + p.replace("<abs_vis_token></abs_vis_token>", "")
    else:
        text = replace_latent_placeholder_with_img_pad(
            processor.apply_chat_template(example, tokenize=False))
    image_inputs, _ = process_vision_info([example])
    image_inputs, _ = resize_by_token_budget(image_inputs)
    batch = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
    obs_poss = _obs_poss(batch["input_ids"], ids)
    m4d = _teacher_obs_mask(batch["input_ids"], batch["attention_mask"], obs_poss, ids)
    inp = dict(latent_mode=False,
               input_ids=batch["input_ids"].to(device),
               attention_mask=batch["attention_mask"].to(device),
               pixel_values=batch["pixel_values"].to(device),
               image_grid_thw=batch["image_grid_thw"].to(device),
               attention_mask_4d={"full_attention": m4d.to(device)},
               alignment_poss=[obs_poss], output_hidden_states=True, loss_type=[])
    return inp, obs_poss


def build_student_stage2_batch(row, processor, root, latent_size: int, device) -> dict:
    ids = _ids(processor)
    example = _absolutize(row, root)
    text = replace_latent_placeholder_with_img_pad(
        processor.apply_chat_template(example, tokenize=False))
    text = add_latent_pad_after_auxiliary_img([text], latent_size, "<abs_vis_token_pad>")[0]
    image_inputs, _ = process_vision_info([example])
    image_inputs, _ = resize_by_token_budget(
        image_inputs, global_max_pixels=1500 * 28 * 28, per_img_max_pixels=1280 * 28 * 28)
    batch = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
    m4d, _ = build_4d_attn(
        input_ids=batch["input_ids"], pad_mask=batch["attention_mask"], token_ids=ids,
        observation_tokens_cannot_see_question_image=True, return_type='bool')
    return dict(input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=batch["pixel_values"].to(device),
                image_grid_thw=batch["image_grid_thw"].to(device),
                attention_mask_4d={"full_attention": m4d.to(device)})


def obs_token_acc_inputs(student_inp: dict, processor, device) -> Tuple[dict, list]:
    """CE-forward inputs (labels + obs positions) derived from a student batch."""
    ids = _ids(processor)
    input_ids = student_inp["input_ids"].cpu()
    obs_poss = _obs_poss(input_ids, ids)
    labels = generate_labels_after_multi_token_start(
        input_ids, ids["ans_start"],
        ignore_ids=[ids["img_pad"], ids["v_start"], ids["v_end"], ids["end_pad"],
                    ids["abs_pad"], ids["abs_end"], ids["obs_start"], ids["obs_end"]])
    ce = {k: v for k, v in student_inp.items()}
    ce["latent_mode"] = False
    ce["labels"] = labels.to(device)
    return ce, obs_poss
