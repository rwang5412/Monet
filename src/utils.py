import torch
import logging
import os
import numpy as np
import json
import random
import argparse
from datasets import Dataset
from typing import List, Union
import math
from PIL import Image
import math

def get_args():
    parser = argparse.ArgumentParser()
    # ===== Basic arguments =====
    parser.add_argument("--load_model_path", type=str, default='./checkpoints/model_sft_stage1')
    parser.add_argument("--data_path", type=str, default='PathToJsonlData', nargs='+')
    parser.add_argument("--stage", type=str, default="sft_stage1", choices=['sft_stage1', 'sft_stage2', 'sft_stage3'])
    parser.add_argument("--task", type=str, default="mm-reasoning", choices=["mm-reasoning"])
    parser.add_argument("--save_model_path", type=str, default='./checkpoints/',help="Path to save the model checkpoints.")
    parser.add_argument("--resume_from_checkpoint", default=False, action="store_true")
    parser.add_argument("--dataset_root", type=str, default="", help="Root directory for the dataset.")
    parser.add_argument("--deepspeed", type=str, default="./deepspeed/ds_zero2_gpu.json",
                        help="Path to DeepSpeed config JSON file")
    parser.add_argument("--num_samples", default=-1, help="-1 means all data", type=int)
    parser.add_argument("--max_seq_len", type=int, default=4096, help="Maximum allowed sequence length after processing.")
    parser.add_argument("--image_resize", type=str, choices=["global", "clear_question_img"], default="global")
    parser.add_argument("--save_freq", type=int, default=250)
    parser.add_argument("--log_freq", default=10, type=int)
    parser.add_argument("--allow_no_observation", action='store_true', default=False)
    # ===== Basic training hyperparameters =====
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for training.")
    parser.add_argument("--bsz", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=10)  
    parser.add_argument("--shuffle_train", action='store_true', default=False, help="Whether to shuffle the training dataset.")

    # ===== Monet arguments =====
    parser.add_argument("--alignment", type=str, default="observation_all", choices=["observation_end", "boxed_start", "observation_all"], help="The alignment strategy for Monet.")
    parser.add_argument("--latent_size", type=int, default=4)
    parser.add_argument("--ce_emphasize_factor", default=1.0, type=float)
    parser.add_argument("--only_predict_obs", action='store_true', default=False)
    parser.add_argument("--alignment_weight", default=1.0, help="Weight of the alignment loss in SFT Stage 2 and SFT Stage 3.", type=float)
    parser.add_argument("--alignment_layer", choices=["all_layers", "last_layer"])
    parser.add_argument("--emphasize_latent_weight", default=1.0, type=float, help="Weight of the loss that only flow through latents in SFT Stage 2.")
    parser.add_argument("--sft_stage2_align_poss", default='obs', choices=['obs', 'latent_end'])
    # ---- Stage-2 modifications (defaults preserve original behavior exactly) ----
    parser.add_argument("--teacher_reps_neg_dir", type=str, default=None,
                        help="Directory of NO-aux-image teacher obs reps (h_neg). When set, the "
                             "Stage-2 alignment becomes the residual margin objective "
                             "(closer to h_pos than to h_neg by obs_residual_margin). Generate "
                             "with precompute_teacher_reps.py --no_aux_images.")
    parser.add_argument("--obs_residual_margin", type=float, default=0.2)
    parser.add_argument("--residual_recenter_path", type=str, default=None,
                        help="Stage-2 margin recentering: path to residual_mean.pt (from "
                             "src.compute_residual_mean). Subtracts the dataset-mean residual "
                             "E[h_pos - h_neg] from h_pos so a global shift of the student's obs "
                             "states earns no margin (anti cross-sample-sim drift). None = off.")
    parser.add_argument("--grounding_weight", type=float, default=0.0,
                        help="lambda for the latent-grounding InfoNCE (Stage-2 Change 2); 0 disables.")
    parser.add_argument("--grounding_queue_size", type=int, default=4096)
    parser.add_argument("--grounding_temp", type=float, default=0.07)
    # Stage-3 modification: decode-CE writer loss (L_dec, the causal lever).
    parser.add_argument("--decode_weight", type=float, default=0.0,
                        help="lambda for the decode-CE loss (Stage-3): the K latents must "
                             "reconstruct the observation sentence alone. 0 disables (stock Stage 3).")
    parser.add_argument("--decode_d", type=int, default=1024, help="L_dec decoder width.")
    parser.add_argument("--decode_layers", type=int, default=4, help="L_dec decoder depth.")
    parser.add_argument("--decode_max_len", type=int, default=96, help="L_dec max observation length.")
    parser.add_argument("--slot_dropout", type=int, default=2,
                        help="L_dec: hide this many of K slots from the decoder's cross-attention "
                             "each step so no single slot is the sole carrier (breaks within-block 0.92).")
    parser.add_argument("--align_recenter_path", type=str, default=None,
                        help="Stage-3 recentered alignment (mod A): path to align_mean.pt "
                             "(from src.compute_target_mean). Subtracts the dataset-mean target "
                             "before the cosine so alignment targets the content subspace. None = off.")
    # Stage-3 modification: random-donor margin swap (reader-side, accuracy-preserving).
    parser.add_argument("--swap_weight", type=float, default=0.0,
                        help="lambda for L_swap (Stage-3): a random donor's latents must make "
                             "the answer worse by --swap_margin. 0 disables. Keep LOW (~0.1-0.3).")
    parser.add_argument("--swap_margin", type=float, default=0.15,
                        help="target answer-NLL gap donor-vs-real (nats): the causality dial. "
                             "Small (0.1-0.2) preserves accuracy for a modest do(Z).")
    parser.add_argument("--swap_every", type=int, default=1, help="compute L_swap every N steps (cost knob).")
    parser.add_argument("--swap_span", type=str, default="obs", choices=["obs", "answer", "both"],
                        help="Span L_swap is supervised on. 'obs' = original design; 'answer' = "
                             "the reasoning+answer tokens where do(Z) is actually measured "
                             "(nll_real is detached, so a confident answer still gives gradient "
                             "through nll_donor); 'both' = union.")
    parser.add_argument("--swap_bank", type=int, default=64, help="donor ring-buffer size.")
    parser.add_argument("--no_aux_images", action='store_true', default=False,
                        help="precompute_teacher_reps.py only: run the teacher WITHOUT auxiliary "
                             "images (produces the h_neg cache for the residual objective).")
    parser.add_argument("--teacher_obs_mask", action='store_true', default=False,
                        help="precompute_teacher_reps.py only: block observation tokens from the "
                             "QUESTION image in the teacher forward. REQUIRED for both h_pos and "
                             "h_neg caches of the residual objective (matches the student's "
                             "--observation_tokens_cannot_see_question_image).")
    parser.add_argument("--probe_fig4", type=int, default=0,
                        help="precompute_teacher_reps.py: run the Figure-4 PRECONDITION on N "
                             "samples (obs-token accuracy with vs without aux image, forward "
                             "only, no caching) and exit. Gap ~0 => warm-up didn't take.")
    parser.add_argument("--probe_layer_sel", type=int, default=0,
                        help="precompute_teacher_reps.py: probe per-layer mean(1-cos(h_pos,h_neg)) "
                             "on N samples (both variants in-memory, no caching), write "
                             "layer_sel.json to --save_model_path, and exit.")
    parser.add_argument("--keep_layers", type=str, default=None,
                        help="precompute_teacher_reps.py: comma-separated hidden-state layer "
                             "indices to KEEP in the cached reps (from the layer_sel probe). "
                             "Cuts cache storage; training must pass the same indices via "
                             "--alignment_layer_indices.")
    parser.add_argument("--alignment_layer_indices", type=str, default=None,
                        help="Training: comma-separated layer indices the teacher caches were "
                             "saved with (--keep_layers); the student hidden-state stack is "
                             "sliced to match before the alignment/residual loss.")
    parser.add_argument("--sft_stage2_global_img_tokens", type=int, help="Maximum img pixels in a sequence will be sft_stage2_global_max_img_tokens*28*28", default=1500)
    parser.add_argument("--sft_stage2_per_img_tokens", type=int, help="Maximum pixels per img will be sft_stage2_global_max_img_tokens*28*28", default=1280)
    parser.add_argument("--sft_stage3_img_tokens", type=int, help="Maximum img pixels in a sequence will be sft_stage3_max_img_tokens*28*28", default=2000)

    # ===== Training record arguments =====
    parser.add_argument("--log_file", type=str, default='./log.txt')
    parser.add_argument("--wandb_name", default=None, help="Name for the Weights & Biases run. If None, no W&B logging is done.")
    
    # ==== Custom attention =====
    parser.add_argument("--not_use_4d", action='store_true', default=False)
    parser.add_argument("--not_mask_image", action='store_true', default=False)
    parser.add_argument("--mask_latent", action='store_true', default=False,
                        help="If set, make latent tokens (A_i) invisible to all subsequent tokens in build_additive_bias.")
    parser.add_argument("--observation_tokens_only_see_image_tokens", action='store_true', default=False)
    parser.add_argument("--observation_tokens_only_see_latent_tokens", action='store_true', default=False)
    parser.add_argument("--observation_tokens_cannot_see_question_image", action='store_true', default=False)
    parser.add_argument("--latent_can_see_all_previous", action='store_true', default=True)
    parser.add_argument("--observation_tokens_only_see_question_and_latent", action='store_true', default=False)
    parser.add_argument("--mask_question_image", action='store_true', default=False)
    # ===== Precomputed teacher latent loading =====
    parser.add_argument("--teacher_latent_dir", type=str, default=None,
                        help="Directory that stores precomputed teacher latents (files named latent_{sample_id:08d}.pt). If not set, defaults to {save_model_path or ./checkpoints}/teacher_latents.")
    parser.add_argument("--teacher_reps_dir", type=str, default=None)
    parser.add_argument("--attn_analysis", action='store_true', default=False)
    parser.add_argument("--output_latent_embeds", action='store_true', default=False)
    parser.add_argument("--output_hidden_states", action='store_true', default=False)
    parser.add_argument("--resume", action="store_true", default=False)


    return parser.parse_args()

def seed_everything(seed: int = 42):
    """
    Set seed for reproducibility across random, numpy, torch, and environment.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_jsonl_dataset(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        data = data[:]
    return Dataset.from_list(data)

def load_json_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def replace_latent_placeholder_with_img_pad(text, image_pad="<|vision_start|><|image_pad|><|vision_end|>", latent_placeholder="<abs_vis_token></abs_vis_token>", sep_token="<|im_start|>assistant") -> str:
    text = text.split(sep_token)
    res_text = process_multiple_question_img(text[0])
    assistant_texts = text[1:]
    for text in assistant_texts:
        if latent_placeholder in text:
            text = text.replace(image_pad, "")
            text = text.replace(latent_placeholder, image_pad)
        res_text += sep_token + text
    return res_text

def remove_auxiliary_images(examples):
    new_examples = []
    for example in examples:
        # `example` is a list of turn dicts
        new_example = []
        for turn in example:
            # Create a shallow copy of the turn so we don't modify the original
            new_turn = dict(turn)
            if turn.get("role") == "assistant":
                # Filter out image-type content
                new_turn["content"] = [
                    item for item in turn.get("content", [])
                    if item.get("type") != "image"
                ]
            # Add the updated turn to this new example
            new_example.append(new_turn)
        new_examples.append(new_example)

    return new_examples

def process_multiple_question_img(question_str):
    if "<abs_vis_token></abs_vis_token>" in question_str:
        question_str = question_str.replace("<|vision_start|><|image_pad|><|vision_end|>", "").replace("<abs_vis_token></abs_vis_token>", "<|vision_start|><|image_pad|><|vision_end|>")
    return question_str

def replace_img_pad_with_latent_pad(texts, latent_size, latent_pad_str="<abs_vis_token_pad>"):
    update_texts = []
    latent_pad_strs = latent_pad_str*latent_size
    for i, text in enumerate(texts):
        turns = text.split("<|im_start|>assistant")
        upd_text = process_multiple_question_img(turns[0])
        for turn in turns[1:]:
            upd_text += "<|im_start|>assistant" + turn.replace("<|vision_start|><|image_pad|><|vision_end|>", f"<abs_vis_token>{latent_pad_strs}</abs_vis_token>")
        update_texts.append(upd_text)
    return update_texts

def add_latent_pad_after_auxiliary_img(texts, latent_size, latent_pad_str="<abs_vis_token_pad>"):
    update_texts = []
    latent_pad_strs = latent_pad_str*latent_size
    for i, text in enumerate(texts):
        turns = text.split("<|im_start|>assistant")
        upd_text = process_multiple_question_img(turns[0])
        for turn in turns[1:]:
            upd_text += "<|im_start|>assistant" + turn.replace("<|vision_start|><|image_pad|><|vision_end|>", f"<|vision_start|><|image_pad|><|vision_end|><abs_vis_token>{latent_pad_strs}</abs_vis_token>")
        update_texts.append(upd_text)
    return update_texts
 
def find_subsequence(row: torch.Tensor, pattern: Union[torch.Tensor, List[torch.Tensor]], start: int=0) -> int:

    seq_len = row.size(0)
    # Naive scan over all possible start positions
    if isinstance(pattern, torch.Tensor):
        max_pat_len = pattern.size(0)
    elif isinstance(pattern, list):
        max_pat_len = max(pat.size(0) for pat in pattern)

    for start_idx in range(start, seq_len - max_pat_len + 1):
        # Compare row[start_idx : start_idx + pat_len] to pattern
        if isinstance(pattern, torch.Tensor):
            pat_len = pattern.size(0)
            if torch.all(row[start_idx : start_idx + pat_len] == pattern):
                return start_idx
        elif isinstance(pattern, list):
            for pat in pattern:
                if isinstance(pat, torch.Tensor):
                    pat_len = pat.size(0)
                    if torch.all(row[start_idx : start_idx + pat_len] == pat):
                        return start_idx

    return -1

def find_ids_poss(input_ids: torch.Tensor, answer_start_token_pattern: torch.Tensor, ids_tensor_or_list: Union[torch.Tensor,List[torch.Tensor]]) -> List[List[int]]:
    batch_poss = []
    for i in range(input_ids.shape[0]):
        manipulation_result_poss = []
        start_idx = find_subsequence(input_ids[i], answer_start_token_pattern, 0)
        while start_idx != -1:
            start_idx = find_subsequence(input_ids[i], ids_tensor_or_list, start_idx+1)
            if start_idx != -1:
                manipulation_result_poss.append(start_idx)
        manipulation_result_poss = manipulation_result_poss[:]
        batch_poss.append(manipulation_result_poss)
    return batch_poss
    
def generate_labels_after_multi_token_start(
    input_ids: torch.Tensor,
    start_sequence: torch.Tensor,
    ignore_ids: List[int] = None
) -> torch.Tensor:
    """
    For each row in `input_ids`, find the *first* occurrence of `start_sequence`
    (a 1D tensor of multiple token IDs). Mask all tokens up to and including
    that entire sub-sequence (set them to -100), and also mask any padding tokens
    anywhere in the row. The remainder (tokens *after* the sub-sequence) are kept.

    Args:
      input_ids: 2D tensor [batch_size, seq_len].
      start_sequence: 1D tensor of shape [k], the multi-token "start" pattern.
      pad_token_id: which ID is used as padding (default=0).
    
    Returns:
      labels: a new 2D tensor [batch_size, seq_len], where tokens before (and
              including) the sub-sequence are -100, as well as any pad tokens,
              and tokens after the sub-sequence are kept as in `input_ids`.
    """
    batch_size, seq_len = input_ids.shape
    
    # Clone so we can modify in-place
    labels = input_ids.clone()
    
    for b in range(batch_size):
        row = labels[b]
        # Find first occurrence of the entire sub-sequence
        start_idx = find_subsequence(row, start_sequence)
        
        if start_idx == -1:
            # Sub-sequence not found -> mask everything
            logging.warning(f"Couldn't find the <|im_start|>assistant, all labels are -100")
            row[:] = -100
        else:
            # The sub-sequence length
            sub_len = start_sequence.size(0)
            end_of_subseq = start_idx + sub_len  # the position *after* the sub-sequence
            
            # Mask everything up to (and including) the sub-sequence
            row[:end_of_subseq] = -100
        
        for id in ignore_ids:
            # Mask specified tokens (<|endoftext|>, <|vision_start|>, <|image_pad|>, <|vision_end|>)
            row[row == id] = -100


    
    return labels

def generate_labels_after_multi_token_start_only_allow(
    input_ids: torch.Tensor,
    start_sequence: torch.Tensor,
    allowed_poss: List[List[int]] = None
) -> torch.Tensor:
    """
    For each row in `input_ids`, find the *first* occurrence of `start_sequence`
    (a 1D tensor of multiple token IDs). Mask all tokens up to and including
    that entire sub-sequence (set them to -100), and also mask any padding tokens
    anywhere in the row. The remainder (tokens *after* the sub-sequence) are kept.

    Args:
      input_ids: 2D tensor [batch_size, seq_len].
      start_sequence: 1D tensor of shape [k], the multi-token "start" pattern.
      pad_token_id: which ID is used as padding (default=0).
    
    Returns:
      labels: a new 2D tensor [batch_size, seq_len], where tokens before (and
              including) the sub-sequence are -100, as well as any pad tokens,
              and tokens after the sub-sequence are kept as in `input_ids`.
    """
    batch_size, seq_len = input_ids.shape
    
    # Clone so we can modify in-place
    labels = input_ids.clone()
    
    for b in range(batch_size):
        row = labels[b]
        # Find first occurrence of the entire sub-sequence
        start_idx = find_subsequence(row, start_sequence)
        
        if start_idx == -1:
            # Sub-sequence not found -> mask everything
            logging.warning(f"Couldn't find the <|im_start|>assistant, all labels are -100")
            row[:] = -100
        else:
            # The sub-sequence length
            sub_len = start_sequence.size(0)
            end_of_subseq = start_idx + sub_len  # the position *after* the sub-sequence
            
            # Mask everything up to (and including) the sub-sequence
            row[:end_of_subseq] = -100
        
        mask = torch.ones_like(row, dtype=torch.bool)
        allowed_pos = allowed_poss[b]
        mask[allowed_pos] = False
        row[mask] = -100

    return labels

def resize_by_token_budget(images,
                           global_max_pixels=2000*28*28,
                           per_img_max_pixels=1280*28*28,
                           divisor=28):
    '''Resuze images to fit within a global token budget and per-image token budget.'''
    total = sum(img.width * img.height for img in images)
    if total <= global_max_pixels:
        return images, None

    ratio = math.sqrt(global_max_pixels / total)

    processed = []
    new_sizes = []
    for img in images:
        w, h = int(img.width * ratio), int(img.height * ratio)
        w = max(divisor, (w // divisor) * divisor)
        h = max(divisor, (h // divisor) * divisor)

        if w * h > per_img_max_pixels:
            r = math.sqrt(per_img_max_pixels / (w * h))
            w = max(divisor, int(w * r) // divisor * divisor)
            h = max(divisor, int(h * r) // divisor * divisor)

        processed.append(img.resize((w, h), Image.BICUBIC))
        new_sizes.append((w, h))
    return processed, new_sizes

def resize_diff(images, 
                question_img_max_pixels=1280*28*28,#2000*28*28, 
                remain_global_max_pixels=800*28*28,#800*3*28*28,
                remain_per_img_max_pixels=800*28*28,#1280*28*28,
                divisor=28):
    processed = []
    new_sizes = []
    question_img_processed, question_img_new_sizes = resize_by_token_budget(
        [images[0]], 
        global_max_pixels=question_img_max_pixels, 
        per_img_max_pixels=question_img_max_pixels,
        divisor=divisor
    )
    processed.append(question_img_processed[0])
    new_sizes.append(question_img_new_sizes[0] if question_img_new_sizes is not None else None)
    remain_img_processed, remain_img_new_sizes = resize_by_token_budget(
        images[1:], 
        global_max_pixels=remain_global_max_pixels, 
        per_img_max_pixels=remain_per_img_max_pixels,
        divisor=divisor
    )
    processed.extend(remain_img_processed)
    new_sizes.extend(remain_img_new_sizes if remain_img_new_sizes is not None else [None]*len(remain_img_processed))
    return processed, new_sizes

def find_helper_img_segs(ids, token_ids):
    device = ids.device
    def between(start_pos, end_pos, wanted_id=None):
        s = start_pos + 1
        e = end_pos
        if s >= e:
            return torch.empty(0, dtype=torch.long, device=device)
        if wanted_id is None:
            return torch.arange(s, e, device=device, dtype=torch.long)
        mask = (ids[s:e] == wanted_id)
        return torch.nonzero(mask, as_tuple=False).squeeze(-1) + s
    v_starts = torch.nonzero(ids == token_ids['v_start'], as_tuple=False).squeeze(-1)
    v_ends   = torch.nonzero(ids == token_ids['v_end'],   as_tuple=False).squeeze(-1)
    v_ptr, e_ptr = 0, 0
    Vs, Ve = [], []
    while v_ptr < v_starts.numel() and e_ptr < v_ends.numel():
        if v_starts[v_ptr] < v_ends[e_ptr]:
            Vs.append(v_starts[v_ptr].item()); Ve.append(v_ends[e_ptr].item())
            v_ptr += 1; e_ptr += 1
        else:
            e_ptr += 1
    # drop the question image (first vision pair)
    Q_img_idx = between(Vs[0], Ve[0], wanted_id=token_ids['img_pad'])
    if len(Vs) > 0 and len(Ve) > 0:
        Vs = Vs[1:]
        Ve = Ve[1:]
    helper_img_segs = [[between(vs, ve, wanted_id=token_ids['img_pad'])] for vs, ve in zip(Vs, Ve)]
    return Q_img_idx, helper_img_segs

def find_segments_1d(ids, token_ids):
    """
    ids: 1D LongTensor, shape [L]
    token_ids: dict with keys:
        'v_start', 'v_end', 'img_pad',
        'abs_start', 'abs_end', 'abs_pad',
        'obs_start', 'obs_end'

    Returns a list for each step S_i with:
        (I_idx: LongTensor, A_idx: LongTensor, O_blocks: List[LongTensor])

    Notes:
    - We assume the first <|vision_start|>...</|vision_end|> pair is the question image
      and is excluded from steps (i.e., only subsequent pairs are treated as I_i).
    - O can appear multiple times inside a single T_i; DO NOT merge them.
      O_blocks contains one LongTensor per <observation>...</observation> block.
    - No cross-step O is expected; we only pair O blocks fully inside each T_i.
    """

    L = ids.numel()
    device = ids.device

    # Helper to collect indices between two tags (exclusive) that match 'wanted_id' (or all if None)
    def between(start_pos, end_pos, wanted_id=None):
        s = start_pos + 1
        e = end_pos
        if s >= e:
            return torch.empty(0, dtype=torch.long, device=device)
        if wanted_id is None:
            return torch.arange(s, e, device=device, dtype=torch.long)
        mask = (ids[s:e] == wanted_id)
        return torch.nonzero(mask, as_tuple=False).squeeze(-1) + s

    # 1) Pair all I_i by <|vision_start|> ... <|vision_end|>
    v_starts = torch.nonzero(ids == token_ids['v_start'], as_tuple=False).squeeze(-1)
    v_ends   = torch.nonzero(ids == token_ids['v_end'],   as_tuple=False).squeeze(-1)
    v_ptr, e_ptr = 0, 0
    Vs, Ve = [], []
    while v_ptr < v_starts.numel() and e_ptr < v_ends.numel():
        if v_starts[v_ptr] < v_ends[e_ptr]:
            Vs.append(v_starts[v_ptr].item()); Ve.append(v_ends[e_ptr].item())
            v_ptr += 1; e_ptr += 1
        else:
            e_ptr += 1
    # drop the question image (first vision pair)
    Q_img_idx = between(Vs[0], Ve[0], wanted_id=token_ids['img_pad'])
    if len(Vs) > 0 and len(Ve) > 0:
        Vs = Vs[1:]
        Ve = Ve[1:]

    # 2) Pair all A_i by <abs_vis_token> ... </abs_vis_token>
    a_starts = torch.nonzero(ids == token_ids['abs_start'], as_tuple=False).squeeze(-1)
    a_ends   = torch.nonzero(ids == token_ids['abs_end'],   as_tuple=False).squeeze(-1)
    a_ptr, b_ptr = 0, 0
    As, Ae = [], []
    while a_ptr < a_starts.numel() and b_ptr < a_ends.numel():
        if a_starts[a_ptr] < a_ends[b_ptr]:
            As.append(a_starts[a_ptr].item()); Ae.append(a_ends[b_ptr].item())
            a_ptr += 1; b_ptr += 1
        else:
            b_ptr += 1

    # Precompute all observation tag positions to avoid repeated arange() inside loops
    obs_starts_all = torch.nonzero(ids == token_ids['obs_start'], as_tuple=False).squeeze(-1)
    obs_ends_all   = torch.nonzero(ids == token_ids['obs_end'],   as_tuple=False).squeeze(-1)

    S = []
    n_steps = min(len(Vs), len(As))
    for i in range(n_steps):
        vs, ve = Vs[i], Ve[i]
        as_, ae = As[i], Ae[i]

        # I_i and A_i indices (exclusive between their own tags)
        I_idx = between(vs, ve, wanted_id=token_ids['img_pad'])
        A_idx = between(as_, ae, wanted_id=token_ids['abs_pad'])

        # Text region T_i: from end of A_i to start of next vision, or to sequence end
        t_end = Vs[i + 1] if (i + 1) < len(Vs) else L

        # Restrict observation tag candidates to the current T_i window
        # Start tags: ae <= pos < t_end
        # End tags:   ae <  pos <= t_end
        # (Fully contained O will satisfy start < end and both within this window.)
        in_start_win = (obs_starts_all >= ae) & (obs_starts_all < t_end)
        in_end_win   = (obs_ends_all   >  ae) & (obs_ends_all   <= t_end)
        o_starts = obs_starts_all[in_start_win]
        o_ends   = obs_ends_all[in_end_win]

        # Greedy 1-1 pairing within T_i
        O_blocks = []
        p, q = 0, 0
        while p < o_starts.numel() and q < o_ends.numel():
            s_pos = o_starts[p].item()
            e_pos = o_ends[q].item()
            if s_pos < e_pos:
                O_idx = between(s_pos, e_pos, wanted_id=None)
                if O_idx.numel() > 0:
                    O_blocks.append(O_idx)
                p += 1
                q += 1
            else:
                q += 1

        S.append((I_idx, A_idx, O_blocks))

    return Q_img_idx, S

def build_4d_attn(
    input_ids,
    pad_mask,
    token_ids,
    large_neg: float = 1e-6,
    not_mask_image: bool = False,
    mask_latent: bool = False,
    observation_tokens_only_see_image_tokens: bool = False,
    observation_tokens_only_see_latent_tokens: bool = False,
    observation_tokens_cannot_see_question_image: bool = False,
    observation_tokens_only_see_question_and_latent: bool = False,
    latent_can_see_all_previous: bool = True,
    mask_question_image: bool = False,
    return_type: str = 'bool'
):
    """
    input_ids: LongTensor [B, L]
    pad_mask:  LongTensor/BoolTensor [B, L], 1/True for real tokens
    token_ids: dict of special token ids
    large_neg: float used as "negative infinity" added to logits (not applied here)
    Returns:
      allowed: BoolTensor [B, 1, L, L], True=allowed to attend, False=blocked
      (Includes causal mask and padding mask already.)
    Notes:
      - This version expects find_segments_1d to return O_blocks: List[LongTensor] per step.
      - We do NOT merge O blocks inside a T_i; each O block gets its own lower-tri self-visibility.
    """

    # Keep on CPU as in the original version; model can cast later if needed.
    input_ids = input_ids.cpu()
    pad_mask = pad_mask.cpu()

    B, L = input_ids.shape
    device = input_ids.device

    # Base causal mask (lower triangular, including diagonal)
    causal = torch.tril(torch.ones((L, L), dtype=torch.bool, device=device))

    # Valid tokens (both query and key must be valid)
    valid = pad_mask.bool()
    allowed = causal.unsqueeze(0).clone()   # [1, L, L]
    allowed = allowed.repeat(B, 1, 1)       # [B, L, L]
    for b in range(B):
        allowed[b] &= valid[b].unsqueeze(0)  # mask keys (columns)
        allowed[b] &= valid[b].unsqueeze(1)  # mask queries (rows)

    batch_segs=[]
    for b in range(B):
        Q_img_idx, segs = find_segments_1d(input_ids[b], token_ids)
        batch_segs.append(segs)
        if not segs:
            continue

        Lb = input_ids.shape[1]
        ids = input_ids[b]

        if mask_question_image:
            allowed[b][:, Q_img_idx] = False  # no one can see question image tokens

        for (I_idx, A_idx, O_blocks) in segs:
            # --- Latent segment A_i rules ---
            if A_idx.numel():
                # Clear all visibility for A_i queries first
                if not latent_can_see_all_previous:
                    allowed[b][A_idx, :] = False
                else:
                    if mask_question_image and Q_img_idx is not None and Q_img_idx.numel() > 0:
                        allowed[b][A_idx.unsqueeze(1), Q_img_idx] = True
                

                # (1) A_i can see its left I_i (image pads inside this vision pair)
                if I_idx.numel():
                    allowed[b][A_idx.unsqueeze(1), I_idx] = True

                # (Optional) A_i prefix self-visibility (lower-tri within A_i)
                n = A_idx.numel()
                ar = torch.arange(n, device=A_idx.device)
                tri = ar.unsqueeze(1) >= ar.unsqueeze(0)  # (n, n) bool
                rows = A_idx.unsqueeze(1).expand(n, n)
                cols = A_idx.unsqueeze(0).expand(n, n)
                allowed[b][rows, cols] = tri

                # Ensure only A_i (and not others) can see I_i
                if I_idx.numel() and not not_mask_image:
                    not_A = torch.ones(Lb, dtype=torch.bool, device=device)
                    not_A[A_idx] = False
                    not_A_idx = torch.nonzero(not_A, as_tuple=False).squeeze(-1)
                    if not_A_idx.numel():
                        allowed[b][not_A_idx[:, None], I_idx] = False

                # Optionally hide A_i from all subsequent non-A queries as keys
                if mask_latent:
                    r_idx = torch.arange(Lb, device=device)
                    rows_to_block = (r_idx.unsqueeze(0) > A_idx.unsqueeze(1)).any(dim=0)  # rows after any A
                    if rows_to_block.any():
                        allowed[b][rows_to_block.nonzero(as_tuple=False).squeeze(-1)[:, None], A_idx] = False



            # --- Observation blocks: treat each O block independently ---
            if O_blocks and A_idx.numel():
                # Locate the question image range once if needed
                if observation_tokens_cannot_see_question_image:
                    q_v_starts = torch.nonzero(ids == token_ids['v_start'], as_tuple=False).squeeze(-1)
                    q_v_ends   = torch.nonzero(ids == token_ids['v_end'],   as_tuple=False).squeeze(-1)
                    if q_v_starts.numel() > 0 and q_v_ends.numel() > 0:
                        question_img_start = q_v_starts[0].item()
                        question_img_end   = q_v_ends[0].item()
                        question_img_idx = torch.arange(question_img_start, question_img_end + 1, device=device)
                    else:
                        question_img_idx = None
                else:
                    question_img_idx = None

                # Precompute first answer start position for question segmentation (inline search)
                if 'ans_start' in token_ids:
                    pat = token_ids['ans_start']
                    if isinstance(pat, torch.Tensor):
                        k = int(pat.numel())
                        ans_start_pos = -1
                        if k == 1:
                            eq = torch.nonzero(ids == pat.item(), as_tuple=False).squeeze(-1)
                            ans_start_pos = int(eq[0].item()) if eq.numel() > 0 else -1
                        else:
                            Lb_local = int(ids.numel())
                            ans_start_pos = -1
                            for s in range(0, Lb_local - k + 1):
                                if torch.equal(ids[s:s+k], pat):
                                    ans_start_pos = s
                                    break
                    else:
                        ans_start_pos = -1
                else:
                    ans_start_pos = -1

                for O_idx in O_blocks:
                    if O_idx.numel() == 0:
                        continue

                    # Default: no extra rules for O beyond causal/padding and the I->only-A restriction
                    if observation_tokens_only_see_question_and_latent:
                        # O can ONLY see: (a) question tokens: positions < first ans_start that are NOT image tokens
                        #                 (b) all latent pad tokens that appear BEFORE this O block
                        allowed[b][O_idx, :] = False

                        Lb_local = ids.size(0)
                        ar = torch.arange(Lb_local, device=device)
                        # Question tokens: before answer start and not image tokens
                        if ans_start_pos != -1:
                            before_ans = ar < ans_start_pos
                        else:
                            # No answer pattern found: treat as no question tokens
                            before_ans = torch.zeros(Lb_local, dtype=torch.bool, device=device)
                        non_image = (ids != token_ids['img_pad']) & (ids != token_ids['v_start']) & (ids != token_ids['v_end'])
                        question_idx = torch.nonzero(before_ans & non_image, as_tuple=False).squeeze(-1)

                        # Latent tokens prior to this observation block (all abs_pad positions with index < first O position)
                        o_first = int(O_idx[0].item())
                        latent_before_mask = (ids == token_ids['abs_pad']) & (ar < o_first)
                        latent_before_idx = torch.nonzero(latent_before_mask, as_tuple=False).squeeze(-1)

                        if question_idx.numel():
                            allowed[b][O_idx.unsqueeze(1), question_idx] = True
                        if latent_before_idx.numel():
                            allowed[b][O_idx.unsqueeze(1), latent_before_idx] = True

                        # Each O block has its own lower-tri self-visibility
                        n_o = O_idx.numel()
                        ar_o = torch.arange(n_o, device=O_idx.device)
                        tri_o = ar_o.unsqueeze(1) >= ar_o.unsqueeze(0)
                        rows_o = O_idx.unsqueeze(1).expand(n_o, n_o)
                        cols_o = O_idx.unsqueeze(0).expand(n_o, n_o)
                        allowed[b][rows_o, cols_o] = tri_o
                        continue

                    if observation_tokens_only_see_image_tokens:
                        allowed[b][O_idx, :] = False
                        if I_idx.numel():
                            allowed[b][O_idx.unsqueeze(1), I_idx] = True

                    if observation_tokens_only_see_latent_tokens:
                        allowed[b][O_idx, :] = False
                        if not mask_latent and A_idx.numel():
                            allowed[b][O_idx.unsqueeze(1), A_idx] = True

                    if question_img_idx is not None:
                        allowed[b][O_idx.unsqueeze(1), question_img_idx] = False

                    # Each O block has its own lower-tri self-visibility
                    n_o = O_idx.numel()
                    ar_o = torch.arange(n_o, device=O_idx.device)
                    tri_o = ar_o.unsqueeze(1) >= ar_o.unsqueeze(0)
                    rows_o = O_idx.unsqueeze(1).expand(n_o, n_o)
                    cols_o = O_idx.unsqueeze(0).expand(n_o, n_o)
                    allowed[b][rows_o, cols_o] = tri_o

            # --- Vision tokens I_i as queries: restrict to identity (optional safety) ---
            '''if I_idx.numel():
                allowed[b][I_idx, :] = False
                allowed[b][I_idx, I_idx] = True'''

    # Keep return type consistent with the previous implementation (bool mask).
    # If you need an additive bias, convert with: bias = (~allowed).float() * large_neg.
    if return_type == 'bool':
        return allowed.unsqueeze(1), batch_segs  # [B, 1, L, L], bool
    elif return_type == 'additive':
        return (~allowed.unsqueeze(1)).float() * large_neg, batch_segs

def find_segments_1d_wo_helper_images(ids, token_ids):
    """
    ids: 1D LongTensor, shape [L]
    token_ids: dict with keys:
        'v_start', 'v_end', 'img_pad',
        'abs_start', 'abs_end', 'abs_pad',
        'obs_start', 'obs_end'
    Returns: list of tuples for each S_i:
        (I_idx: LongTensor, A_idx: LongTensor, O_idx: LongTensor)
        O_idx may be empty if no <observation>...</observation> in T_i
    """
    L = ids.numel()
    # Helper to collect indices between two tags (exclusive) that match 'wanted_id' (or all if None)
    def between(start_pos, end_pos, wanted_id=None):
        s = start_pos + 1
        e = end_pos
        if s >= e: 
            return torch.empty(0, dtype=torch.long, device=ids.device)
        if wanted_id is None:
            idx = torch.arange(s, e, device=ids.device)
        else:
            mask = (ids[s:e] == wanted_id)
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1) + s
        return idx


    # 2) Parse all A_i by pairing <abs_vis_token> ... </abs_vis_token>
    a_starts = torch.nonzero(ids == token_ids['abs_start'], as_tuple=False).squeeze(-1)
    a_ends   = torch.nonzero(ids == token_ids['abs_end'],   as_tuple=False).squeeze(-1)
    As, Ae = [], []
    a_ptr, b_ptr = 0, 0
    while a_ptr < a_starts.numel() and b_ptr < a_ends.numel():
        if a_starts[a_ptr] < a_ends[b_ptr]:
            As.append(a_starts[a_ptr].item()); Ae.append(a_ends[b_ptr].item())
            a_ptr += 1; b_ptr += 1
        else:
            b_ptr += 1

    # 3) For each (I_i, A_i) in order, find O_i within T_i
    S = []
    for i in range(len(As)):
        as_, ae = As[i], Ae[i]

        A_idx = between(as_, ae, wanted_id=token_ids['abs_pad'])

        # T_i is from ae to next latent start (or end of sequence)
        t_end = As[i+1] if i + 1 < len(As) else L
        # Find all <observation>...</observation> fully inside T_i
        obs_starts = torch.nonzero((ids == token_ids['obs_start']) & (torch.arange(L, device=ids.device) >= ae) & (torch.arange(L, device=ids.device) < t_end), as_tuple=False).squeeze(-1)
        obs_ends   = torch.nonzero((ids == token_ids['obs_end'])   & (torch.arange(L, device=ids.device) >  ae) & (torch.arange(L, device=ids.device) <= t_end), as_tuple=False).squeeze(-1)

        # Pair obs tags in order
        O_all = []
        p, q = 0, 0
        while p < obs_starts.numel() and q < obs_ends.numel():
            if obs_starts[p] < obs_ends[q]:
                # tokens between the two tags (exclusive) belong to O_i
                O_idx = between(obs_starts[p].item(), obs_ends[q].item(), wanted_id=None)
                if O_idx.numel():
                    O_all.append(O_idx)
                p += 1; q += 1
            else:
                q += 1

        O_idx = torch.cat(O_all, dim=0) if len(O_all) else torch.empty(0, dtype=torch.long, device=ids.device)
        S.append((A_idx, O_idx))

    return S

def build_4d_attn_wo_helper_images(input_ids, pad_mask, token_ids, mask_latent: bool = False,
                                   observation_tokens_cannot_see_question_image: bool = False):
    """
    input_ids: LongTensor [B, L]
    pad_mask:  LongTensor/BoolTensor [B, L], 1/True for real tokens
    token_ids: dict of special token ids (see above)
    large_neg: float used as "negative infinity" added to logits

    Returns:
      attn_bias: FloatTensor [B, 1, L, L] with 0 for allowed and large_neg for blocked
                 This bias ALREADY includes causal mask and padding mask.
    """
    input_ids = input_ids.cpu()
    pad_mask = pad_mask.cpu()
    
    B, L = input_ids.shape
    device = input_ids.device

    # Base: causal visibility (lower-triangular including diagonal)
    causal = torch.tril(torch.ones((L, L), dtype=torch.bool, device=device))

    # Start from causal AND valid tokens (both query & key must be valid)
    valid = pad_mask.bool()
    allowed = causal.unsqueeze(0).clone()  # [1, L, L]
    allowed = allowed.repeat(B, 1, 1)      # [B, L, L]
    for b in range(B):
        allowed[b] &= valid[b].unsqueeze(0)  # mask keys
        allowed[b] &= valid[b].unsqueeze(1)  # mask queries

    # Apply per-segment constraints
    for b in range(B):
        segs = find_segments_1d_wo_helper_images(input_ids[b], token_ids)
        if not segs:
            continue

        Lb = input_ids.shape[1]
        # Stage-3 observation masking (gate 15480729 diagnosis): with no mask the
        # observation tokens attend the QUESTION IMAGE directly, so the LM never
        # needs the latents -- obs NLL 0.090, presence gap only +0.171 and content
        # gap +0.0002, versus +0.722 / +0.0175 for masked Stage 2. Stage 3 was
        # systematically undoing what Stage 2 builds. Blocking the image columns
        # for observation rows makes the latents the only visual route, exactly as
        # --observation_tokens_cannot_see_question_image does in Stage 2.
        img_cols = None
        if observation_tokens_cannot_see_question_image:
            img_ids = [token_ids[k] for k in ('img_pad', 'v_start', 'v_end') if k in token_ids]
            m = torch.zeros(Lb, dtype=torch.bool)
            for t in img_ids:
                m |= (input_ids[b] == (t.item() if torch.is_tensor(t) else t))
            img_cols = m.nonzero(as_tuple=False).squeeze(-1)
        for (A_idx, O_idx) in segs:
            if img_cols is not None and img_cols.numel() and O_idx.numel():
                allowed[b][O_idx[:, None], img_cols] = False
            if A_idx.numel():
                # Optional: make A_i invisible to all subsequent tokens (as keys)
                if mask_latent:
                    # rows r are considered "subsequent" if any a in A_idx satisfies a < r
                    r_idx = torch.arange(Lb, device=device)
                    rows_to_block = (r_idx.unsqueeze(0) >= A_idx.unsqueeze(1)).any(dim=0)  # [L]
                    if rows_to_block.any():
                        allowed[b][rows_to_block.nonzero(as_tuple=False).squeeze(-1)[:, None], A_idx] = False

    return allowed.unsqueeze(1)


from typing import List, Tuple, Dict, Any

def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    # Merge overlapping/adjacent [start, end) spans.
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = merged[-1]
        if s <= pe:  # overlap or adjacent
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged

def strip_observation_and_track_retokenized(
    text: str,
    tokenizer,
    open_tag: str = "<observation>",
    close_tag: str = "</observation>",
) -> Dict[str, Any]:
    # 1) Build clean_text and record observation char spans in clean_text.
    clean_parts: List[str] = []
    obs_spans: List[Tuple[int, int]] = []

    i = 0
    depth = 0
    cur_start = None

    while i < len(text):
        next_open = text.find(open_tag, i)
        next_close = text.find(close_tag, i)

        # Choose the nearest tag (open or close).
        candidates = [(next_open, "open"), (next_close, "close")]
        candidates = [(pos, typ) for pos, typ in candidates if pos != -1]
        if not candidates:
            clean_parts.append(text[i:])
            break

        pos, typ = min(candidates, key=lambda x: x[0])
        # Append normal text before the tag.
        clean_parts.append(text[i:pos])
        i = pos + (len(open_tag) if typ == "open" else len(close_tag))

        # Update depth and span bookkeeping.
        clean_len = sum(len(p) for p in clean_parts)
        if typ == "open":
            depth += 1
            if depth == 1:
                cur_start = clean_len
        else:  # close
            if depth > 0:
                depth -= 1
                if depth == 0 and cur_start is not None:
                    obs_spans.append((cur_start, clean_len))
                    cur_start = None

    clean_text = "".join(clean_parts)

    # If unbalanced (text ends inside observation), close at end.
    if depth > 0 and cur_start is not None:
        obs_spans.append((cur_start, len(clean_text)))

    obs_spans = _merge_spans(obs_spans)

    # 2) Retokenize clean_text with offsets.
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("This method needs a fast tokenizer to get offset_mapping.")

    enc = tokenizer(
        clean_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    clean_ids: List[int] = enc["input_ids"]
    offsets: List[Tuple[int, int]] = enc["offset_mapping"]

    # 3) Map observation char spans -> token positions.
    obs_token_positions: List[int] = []
    j = 0  # pointer over obs_spans

    for tidx, (s, e) in enumerate(offsets):
        # Skip empty offsets if any.
        if e <= s:
            continue
        # Advance span pointer until span end > token start.
        while j < len(obs_spans) and obs_spans[j][1] <= s:
            j += 1
        if j >= len(obs_spans):
            break
        span_s, span_e = obs_spans[j]
        # Overlap check: token intersects observation characters.
        if max(s, span_s) < min(e, span_e):
            obs_token_positions.append(tidx)

    return {
        "clean_text": clean_text,
        "clean_ids": clean_ids,
        "obs_token_positions": obs_token_positions,
    }



if __name__=="__main__":
    pass