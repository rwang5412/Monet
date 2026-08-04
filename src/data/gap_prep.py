"""Standalone single-row student collation for measurement scripts.

Replicates the stage-3 student path (main.py::collate_fn_sft_stage3) for ONE raw
dataset row without main.py's script globals, so measurement tools
(src/train/measure_gap.py) can run against the base checkpoint with no training
side effects. Kept deliberately minimal: single latent block, student-only
tensors, no labels/observation machinery beyond what the gap needs.
"""
import copy
import os
import re
from typing import Tuple

from PIL import Image
from qwen_vl_utils import process_vision_info

from ..utils import (remove_auxiliary_images, replace_img_pad_with_latent_pad,
                     replace_latent_placeholder_with_img_pad,
                     resize_by_token_budget)


def _absolutize_images(example: dict, dataset_root: str) -> dict:
    ex = copy.deepcopy(example)
    for msg in ex["data"]:
        for item in msg.get("content", []):
            if item.get("type") == "image" and not os.path.isabs(str(item["image"])):
                item["image"] = os.path.join(dataset_root, item["image"])
    return ex


def extract_boxed_answer(example: dict) -> str:
    texts = " ".join(
        it.get("text", "") for m in example["data"] if m.get("role") == "assistant"
        for it in m["content"] if it.get("type") == "text")
    m = re.findall(r"\\boxed\{([^{}]*)\}", texts)
    assert m, "row has no \\boxed{} answer"
    return m[-1]


def build_student_batch(row: dict, processor, dataset_root: str,
                        latent_size: int, img_tokens: int = 2000) -> Tuple[dict, str]:
    """Return (student batch tensors, boxed answer text) for one raw row."""
    ex = _absolutize_images(row, dataset_root)
    answer_text = extract_boxed_answer(ex)
    example = ex["data"]

    text = processor.apply_chat_template(example, tokenize=False)
    text = replace_latent_placeholder_with_img_pad(text)
    image_inputs, _ = process_vision_info(example)
    image_inputs, new_sizes = resize_by_token_budget(
        image_inputs, global_max_pixels=img_tokens * 28 * 28,
        per_img_max_pixels=img_tokens * 28 * 28)

    student_texts = replace_img_pad_with_latent_pad([text], latent_size,
                                                    "<abs_vis_token_pad>")
    user_examples = remove_auxiliary_images([example])
    user_image_inputs, _ = process_vision_info(user_examples)
    if new_sizes is not None:
        n_user = len(user_image_inputs)
        for i in range(n_user):
            user_image_inputs[i] = user_image_inputs[i].resize(new_sizes[i], Image.BICUBIC)

    batch = processor(text=student_texts, images=user_image_inputs,
                      return_tensors="pt", padding=True)
    assert student_texts[0].count("<|image_pad|>") == len(user_image_inputs)
    return {
        "student_input_ids": batch["input_ids"],
        "student_attention_mask": batch["attention_mask"],
        "student_pixel_values": batch["pixel_values"],
        "student_image_grid_thw": batch["image_grid_thw"],
    }, answer_text
