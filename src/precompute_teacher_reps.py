import re
from glob import glob
import os as _early_os
# Also import standard os for later usages in this file
import os
import datetime as _dt
# Disable parallelism in HuggingFace tokenizers to avoid fork-related warnings/deadlocks
_early_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Enable faster failure and better logs for NCCL collectives
_early_os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
# Remove deprecated var in this process to avoid warnings if present
if "NCCL_ASYNC_ERROR_HANDLING" in _early_os.environ:
    _early_os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
_early_os.environ.setdefault("NCCL_DEBUG", "WARN")  # change to INFO for deeper debugging
_early_os.environ.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "1048576")  # enable flight recorder
import shutil
from functools import partial
import torch
from monet_qwen_model import apply_qwen2_5_monet
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, AutoTokenizer, AutoProcessor
from PIL import Image
import logging
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig
from qwen_vl_utils import process_vision_info
import torch.distributed as dist
from src.utils import *
from src.task import *
from src.trainer import *
import random
import wandb
args=get_args()
assert args.save_model_path != "./checkpoints/", "You must specify the save path of the latent embeddings"
if not getattr(args, 'teacher_obs_mask', False):
    print("WARNING: --teacher_obs_mask is OFF. For the RESIDUAL objective both "
          "teacher caches (h_pos AND h_neg) must be computed with it ON, or the "
          "obs states read the answer region straight off the question image and "
          "s_pos ~= s_neg (flat margin). Only omit it to reproduce the paper's "
          "original absolute-alignment cache.")
config = Qwen2_5_VLConfig.from_pretrained(args.load_model_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.load_model_path,
    config=config,
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(args.load_model_path, use_fast=False)

preprocess_function = task_preporcess_config[args.task]
all_train_dataset = []
for data_path in args.data_path:
    if data_path.endswith('.jsonl'):
        train_dataset = load_jsonl_dataset(data_path)
    elif data_path.endswith('.json'):
        train_dataset = load_json_dataset(data_path)
    all_train_dataset.extend(train_dataset[:])
if args.shuffle_train:
    random.seed(42)
    random.shuffle(all_train_dataset)

train_dataset = []
cur_max = -1
for i, sample in tqdm(enumerate(all_train_dataset[:]), desc="Collecting training data and length check...", total=len(all_train_dataset)):
    processed = preprocess_function(sample, dataset_root=args.dataset_root)
    if processed is not None:
        train_dataset.append(processed)

# Subset support (pilots + the fig4/layer_sel probes).
if getattr(args, 'num_samples', -1) and args.num_samples > 0:
    train_dataset = train_dataset[:args.num_samples]


# ================= Prepare tokenizer special ids and misc =================
processor.tokenizer.add_tokens("<abs_vis_token_pad>", special_tokens=True)
processor.tokenizer.add_tokens("<abs_vis_token>", special_tokens=True)
processor.tokenizer.add_tokens("</abs_vis_token>", special_tokens=True)
processor.tokenizer.add_tokens("<observation>", special_tokens=True)
processor.tokenizer.add_tokens("</observation>", special_tokens=True)

latent_start_idx = processor.tokenizer("<abs_vis_token>", return_tensors="pt")["input_ids"][0]
latent_end_idx = processor.tokenizer("</abs_vis_token>", return_tensors="pt")["input_ids"][0]
latent_pad_idx = processor.tokenizer("<abs_vis_token_pad>", return_tensors="pt")["input_ids"][0]
observation_start_idx = processor.tokenizer("<observation>", return_tensors="pt")["input_ids"][0]
observation_end_idx = processor.tokenizer("</observation>", return_tensors="pt")["input_ids"][0]
end_pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0]
answer_start_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]
img_start_idx = processor.tokenizer("<|vision_start|>", return_tensors="pt")["input_ids"][0]
img_end_idx = processor.tokenizer("<|vision_end|>", return_tensors="pt")["input_ids"][0]
img_pad_idx = processor.tokenizer("<|image_pad|>", return_tensors="pt")["input_ids"][0]

SPECIAL_id = {
    "v_start": img_start_idx,
    "v_end": img_end_idx,
    "img_pad": img_pad_idx,
    "abs_start": latent_start_idx,
    "abs_end": latent_end_idx,
    "abs_pad": latent_pad_idx,
    "obs_start": observation_start_idx,
    "obs_end": observation_end_idx,
}

# Resize embeddings to include newly added tokens if needed
try:
    new_vocab_size = len(processor.tokenizer)
    model.resize_token_embeddings(new_vocab_size)
    model.config.vocab_size = new_vocab_size
except Exception:
    pass

# Configure latent ids on model for downstream logic
model.config.latent_token_id = int(latent_pad_idx)
model.config.latent_start_id = int(latent_start_idx)
model.config.latent_end_id = int(latent_end_idx)
model.config.answer_start_pattern = answer_start_pattern.tolist()

# Freeze visual to match training behavior and eval-only run
for p in model.visual.parameters():
    p.requires_grad = False

model.eval()
try:
    model.gradient_checkpointing_disable()
except Exception:
    pass

def collate_fn_precompute_teacher_rep(examples, alignment="boxed_start"):
    batch = {}
    batch['metadata'] = [ex['metadata'] for ex in examples]
    examples = [ex['data'] for ex in examples]

    if getattr(args, 'no_aux_images', False):
        # h_neg mode (Stage-2 residual objective): the teacher sees the SAME text
        # CoT but NO auxiliary images and no latent placeholders -- pure text +
        # question image. The observation-token COUNT is unchanged (only image
        # pads / placeholders are removed), so the saved reps align 1:1 with the
        # with-aux h_pos cache. The question (user) part gets the same
        # process_multiple_question_img treatment as the with-aux path so the two
        # caches differ ONLY by the auxiliary image.
        examples = remove_auxiliary_images(examples)
        texts = []
        for ex in examples:
            t = processor.apply_chat_template(ex, tokenize=False)
            parts = t.split("<|im_start|>assistant")
            out = process_multiple_question_img(parts[0])
            for p in parts[1:]:
                out += "<|im_start|>assistant" + p.replace("<abs_vis_token></abs_vis_token>", "")
            texts.append(out)
    else:
        texts = [processor.apply_chat_template(ex, tokenize=False) for ex in examples]
        # replace <abs_vis_token></abs_vis_token> with <|vision_start|><|image_pad|><|vision_end|> for each <|im_start|>assistant content
        texts = [replace_latent_placeholder_with_img_pad(text) for text in texts]

    ################################################
    # teacher
    ################################################
    image_inputs, _ = process_vision_info(examples)
    if args.image_resize == "global":
        image_inputs, new_sizes = resize_by_token_budget(image_inputs)
    elif args.image_resize == "clear_question":
        image_inputs, new_sizes = resize_diff(image_inputs) # resize_by_token_budget(image_inputs)
    teacher_texts = texts
    teacher_batch = processor(text=teacher_texts, images=image_inputs, return_tensors="pt", padding=True)
    total_image_pads = 0
    for txt in texts:
        total_image_pads += txt.count("<|image_pad|>")
    assert total_image_pads == len(image_inputs)
    batch['teacher_pixel_values'] = teacher_batch['pixel_values']
    batch['teacher_image_grid_thw'] = teacher_batch['image_grid_thw']
    batch['teacher_input_ids'] = teacher_batch['input_ids']
    batch['teacher_attention_mask'] = teacher_batch['attention_mask']

    if args.sft_stage2_align_poss == 'obs':
        observation_start_poss = find_ids_poss(batch["teacher_input_ids"], answer_start_pattern, observation_start_idx)
        observation_end_poss = find_ids_poss(batch["teacher_input_ids"], answer_start_pattern, observation_end_idx)
        batch["teacher_observation_poss"] = []
        assert len(observation_start_poss) == len(observation_end_poss)
        for start_poss, end_poss in zip(observation_start_poss, observation_end_poss):
            poss_of_a_sample = []
            if len(start_poss) > 0 and len(end_poss) > 0:
                assert len(start_poss) == len(end_poss), f"start_poss: {start_poss}, end_poss: {end_poss}"
                for start, end in zip(start_poss, end_poss):
                    poss_of_a_sample.extend(list(range(start, end)))
            batch["teacher_observation_poss"].append(poss_of_a_sample)
    elif args.sft_stage2_align_poss == 'latent_end':
        batch["latent_end_poss"] = find_ids_poss(batch["teacher_input_ids"], answer_start_pattern, latent_end_idx)

    return batch


def build_teacher_obs_mask(input_ids: torch.Tensor, pad_mask: torch.Tensor,
                           obs_poss: list) -> torch.Tensor:
    """4D attention mask for the TEACHER passes of the residual objective.

    Rule: observation-token rows may NOT attend to QUESTION-image tokens (any
    vision span occurring before the first assistant marker). Everything else is
    plain causal+padding. Consequences per pass:
      * with-aux (h_pos):  obs see the aux image (assistant turn, causal) but
        not the question image  -> encodes text + VISUAL residual
      * no-aux  (h_neg):   there is no aux image -> obs see no image at all
        -> encodes text only
    The student's build_4d_attn cannot be reused here: all its obs rules are
    gated on the presence of a latent block, which teacher sequences lack.
    Returns allowed: BoolTensor [B, 1, L, L].
    """
    ids_cpu = input_ids.cpu()
    B, L = ids_cpu.shape
    causal = torch.tril(torch.ones((L, L), dtype=torch.bool))
    allowed = causal.unsqueeze(0).repeat(B, 1, 1)
    valid = pad_mask.cpu().bool()
    for b in range(B):
        allowed[b] &= valid[b].unsqueeze(0)
        allowed[b] &= valid[b].unsqueeze(1)
        if not obs_poss[b]:
            continue
        row = ids_cpu[b]
        # first assistant marker = end of the question region
        pat = answer_start_pattern
        k = int(pat.numel())
        ans_start = -1
        for s in range(0, L - k + 1):
            if torch.equal(row[s:s+k], pat):
                ans_start = s
                break
        if ans_start == -1:
            continue
        v_starts = torch.nonzero(row == img_start_idx.item(), as_tuple=False).flatten()
        v_ends = torch.nonzero(row == img_end_idx.item(), as_tuple=False).flatten()
        q_img = []
        for s_, e_ in zip(v_starts.tolist(), v_ends.tolist()):
            if e_ < ans_start:  # a question-image span
                q_img.extend(range(s_, e_ + 1))
        if not q_img:
            continue
        o_idx = torch.tensor(obs_poss[b], dtype=torch.long)
        allowed[b][o_idx.unsqueeze(1), torch.tensor(q_img, dtype=torch.long)] = False
    return allowed.unsqueeze(1)  # [B, 1, L, L]


def _device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
        try:
            torch.cuda.set_device(local_rank)
        except Exception:
            pass
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _scan_existing_reps(save_dir: str, align_poss = "obs") -> set[str]:
    """Scan existing rep_*.pt files and return a set of metadata_info strings.
    Expected filename pattern: rep_{metadata_info}.pt
    """
    if not os.path.isdir(save_dir):
        return set()
    done = set()
    if align_poss == 'obs':
        pattern = r"^rep_(.+)\.pt$"
    elif align_poss == 'latent_end':
        pattern = r"^rep_latent_end_(.+)\.pt$"
    for p in glob(os.path.join(save_dir, "rep_*.pt")):
        fname = os.path.basename(p)
        m = re.match(pattern, fname)
        if m:
            done.add(m.group(1))
    return done


def _expected_metadata_info(metadata: dict, args) -> str:
    """Build the expected metadata_info string for a sample metadata dict."""
    dataset_name = metadata["dataset_name"]
    sample_id = metadata["sample_id"]
    # Keep exactly the same prefix rule as in saving code
    if getattr(args, "output_latent_embeds", False):
        prefix = "last_layer"
    else:
        prefix = "all_layers"
    return f"{prefix}_{dataset_name}_{sample_id}"

def _filter_indices_by_resume(train_dataset: list, args, align_poss = "obs") -> list[int]:
    """When --resume is on, drop indices that are already computed in save dir."""
    save_dir = args.save_model_path
    done = _scan_existing_reps(save_dir, align_poss = align_poss)
    # Build a deterministic list of indices to compute
    keep = []
    for idx, ex in enumerate(train_dataset):
        md = ex["metadata"]
        info = _expected_metadata_info(md, args)
        if info not in done:
            keep.append(idx)
    return keep

def _probe_forward(examples, no_aux: bool, device, want_acc: bool):
    """One teacher forward for the probes: both aux variants share this path.
    Returns (mean_obs_token_acc, hidden_states_list) -- hidden states are the
    per-sample [num_layer, T_obs, D] stacks the normal precompute would cache.
    Masking follows --teacher_obs_mask exactly like the caching path, so probe
    numbers reflect the same regime the caches will be built under."""
    prev = getattr(args, 'no_aux_images', False)
    args.no_aux_images = no_aux
    try:
        batch = collate_fn_precompute_teacher_rep(examples)
    finally:
        args.no_aux_images = prev
    alignment_poss = batch['teacher_observation_poss']
    labels = generate_labels_after_multi_token_start(
        batch["teacher_input_ids"], answer_start_pattern,
        ignore_ids=[end_pad_token_idx, img_pad_idx, img_start_idx, img_end_idx,
                    observation_start_idx, observation_end_idx])
    inputs = {
        'latent_mode': False,
        'input_ids': batch['teacher_input_ids'].to(device),
        'attention_mask': batch['teacher_attention_mask'].to(device),
        'pixel_values': batch['teacher_pixel_values'].to(device),
        'image_grid_thw': batch['teacher_image_grid_thw'].to(device),
        'alignment_poss': alignment_poss,
        'output_hidden_states': True,
        'loss_type': [],
    }
    if want_acc:
        inputs['labels'] = labels.to(device)
        inputs['loss_type'] = ['ce']
        inputs['ce_emphasize_poss'] = alignment_poss
        inputs['ce_emphasize_factor'] = 1.0
        inputs['compute_emphasize_acc'] = True
    if getattr(args, 'teacher_obs_mask', False):
        m4d = build_teacher_obs_mask(batch['teacher_input_ids'],
                                     batch['teacher_attention_mask'], alignment_poss)
        inputs['attention_mask_4d'] = {"full_attention": m4d.to(device)}
    out = model(**inputs, return_dict=True)
    acc = getattr(out, 'mean_emphasize_acc', None)
    return acc, out.hidden_states


def probe_fig4(device, n: int):
    """§0 precondition: obs-token prediction accuracy WITH vs WITHOUT the aux
    image on the frozen warm-up teacher. Gap ~0 => warm-up didn't take; stage-2
    residual training would be flat regardless of implementation."""
    accs_pos, accs_neg = [], []
    with torch.inference_mode():
        for ex in tqdm(train_dataset[:n], desc="fig4 probe"):
            try:
                a_pos, _ = _probe_forward([ex], no_aux=False, device=device, want_acc=True)
                a_neg, _ = _probe_forward([ex], no_aux=True, device=device, want_acc=True)
            except Exception as e:
                logging.warning(f"fig4 probe skip: {e}")
                continue
            if a_pos is not None and a_neg is not None:
                accs_pos.append(a_pos)
                accs_neg.append(a_neg)
    n_ok = len(accs_pos)
    mp = sum(accs_pos) / max(n_ok, 1)
    mn = sum(accs_neg) / max(n_ok, 1)
    print("=" * 60)
    print(f"  Figure-4 precondition  (n={n_ok}, teacher_obs_mask={getattr(args,'teacher_obs_mask',False)})")
    print(f"  obs-token acc WITH aux    = {mp:.4f}")
    print(f"  obs-token acc WITHOUT aux = {mn:.4f}")
    print(f"  GAP = {mp - mn:+.4f}   (substantial => proceed; ~0 => re-run Stage 1)")
    print("=" * 60)


def probe_layer_sel(device, n: int):
    """§2 layer selection: per-layer mean(1 - cos(h_pos, h_neg)) over obs tokens.
    Layers where the two teachers already agree contribute only noise under the
    margin loss. Writes layer_sel.json (upper-third by separation) to save dir."""
    import json as _json
    sep_sum, sep_cnt = None, 0
    with torch.inference_mode():
        for ex in tqdm(train_dataset[:n], desc="layer_sel probe"):
            try:
                _, h_pos = _probe_forward([ex], no_aux=False, device=device, want_acc=False)
                _, h_neg = _probe_forward([ex], no_aux=True, device=device, want_acc=False)
            except Exception as e:
                logging.warning(f"layer_sel probe skip: {e}")
                continue
            hp, hn = h_pos[0].float(), h_neg[0].float()   # [num_layer, T_obs, D]
            if hp.shape != hn.shape or hp.numel() == 0:
                continue
            sep = (1 - torch.nn.functional.cosine_similarity(hp, hn, dim=-1)).mean(dim=1)  # [num_layer]
            sep_sum = sep if sep_sum is None else sep_sum + sep
            sep_cnt += 1
    assert sep_cnt > 0, "layer_sel probe: no usable samples"
    sep_mean = (sep_sum / sep_cnt).cpu()
    order = torch.argsort(sep_mean, descending=True)
    k = max(1, len(sep_mean) // 3)                        # upper third
    selected = sorted(order[:k].tolist())
    os.makedirs(args.save_model_path, exist_ok=True)
    out_path = os.path.join(args.save_model_path, "layer_sel.json")
    _json.dump({"selected_layers": selected, "n_probe": sep_cnt,
                "per_layer_separation": sep_mean.tolist(),
                "teacher_obs_mask": bool(getattr(args, 'teacher_obs_mask', False))},
               open(out_path, "w"), indent=2)
    print("=" * 60)
    print(f"  Layer-selection probe (n={sep_cnt})")
    for i, s in enumerate(sep_mean.tolist()):
        tag = "  <== KEEP" if i in selected else ""
        print(f"    layer {i:2d}  mean(1-cos(h_pos,h_neg)) = {s:.4f}{tag}")
    print(f"  wrote {out_path}")
    print(f"  use in caching:  --keep_layers {','.join(map(str, selected))}")
    print(f"  use in training: --alignment_layer_indices {','.join(map(str, selected))}")
    print("=" * 60)


def main():
    # Initialize distributed if requested
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_dist = world_size > 1
    if is_dist and not (dist.is_available() and dist.is_initialized()):
        # Use a generous timeout to avoid false positives on large models
        dist.init_process_group(backend="nccl", timeout=_dt.timedelta(minutes=30))
    try:
        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))

        device = _device()
        model.to(device)

        # ---- probe modes: forward-only, no caching, single process ----
        if getattr(args, 'probe_fig4', 0) > 0:
            probe_fig4(device, args.probe_fig4)
            return
        if getattr(args, 'probe_layer_sel', 0) > 0:
            probe_layer_sel(device, args.probe_layer_sel)
            return

        # Save dir for latents
        out_dir = args.save_model_path
        os.makedirs(out_dir, exist_ok=True)

        keep_layers = None
        if getattr(args, 'keep_layers', None):
            keep_layers = [int(x) for x in args.keep_layers.split(',') if x.strip() != '']
            if rank == 0:
                logging.info(f"[precompute] caching ONLY hidden-state layers {keep_layers} "
                             f"(training must pass --alignment_layer_indices {args.keep_layers})")

        # Iterate data and precompute
        bs = max(1, int(getattr(args, 'bsz', 1)))
        total = len(train_dataset)

        # ===== Resume support: drop finished samples =====
        if getattr(args, "resume", False):
            # Each rank computes the same filtered index list to avoid desync
            indices_to_process = _filter_indices_by_resume(train_dataset, args, align_poss = args.sft_stage2_align_poss)
            total = len(indices_to_process)
            if rank == 0:
                logging.info(f"[resume] filtered unfinished samples: {total} remain (out of {len(train_dataset)}).")
        else:
            indices_to_process = list(range(total))

        if is_dist:
            # Cross-rank consistency check for dataset length; mismatch is a common source of collective hangs
            try:
                t = torch.tensor([total], device=device)
                gathered = [torch.zeros_like(t) for _ in range(world_size)]
                dist.all_gather(gathered, t)
                totals = [int(x.item()) for x in gathered]
                if len(set(totals)) != 1:
                    if rank == 0:
                        logging.error(f"[precompute] dataset length mismatch across ranks: {totals}. "
                                      f"This can lead to deadlocks. Exiting.")
                    return
            except Exception as e:
                if rank == 0:
                    logging.warning(f"[precompute] total all_gather check failed: {e}")
        if rank == 0:
            logging.info(f"[precompute] total samples={total}, batch_size={bs}, saving to {out_dir}; world_size={world_size}")

        # Build index shards per rank
        indices = list(range(total))
        if is_dist:
            per = (total + world_size - 1) // world_size
            start_idx = rank * per
            end_idx = min(total, (rank + 1) * per)
            shard = indices_to_process[start_idx:end_idx]
        else:
            shard = indices_to_process

        # Avoid early barriers that can deadlock if any rank errors; not required for independent precompute

        with torch.inference_mode():
            rng = range(0, len(shard), bs)
            pbar = tqdm(rng, desc=f"[rank {rank}] precompute", disable=False)
            for i in pbar:
                cur_ids = shard[i:i+bs]
                try:
                    examples = [train_dataset[j] for j in cur_ids]
                    batch = collate_fn_precompute_teacher_rep(examples)
                    if args.sft_stage2_align_poss == 'obs':
                        alignment_poss = batch['teacher_observation_poss']
                    elif args.sft_stage2_align_poss == 'latent_end':
                        alignment_poss = batch['latent_end_poss']
                    inputs = {
                        'latent_mode': False,
                        'input_ids': batch['teacher_input_ids'].to(device),
                        'attention_mask': batch['teacher_attention_mask'].to(device),
                        'pixel_values': batch['teacher_pixel_values'].to(device),
                        'image_grid_thw': batch['teacher_image_grid_thw'].to(device),
                        'labels': None,
                        'alignment_poss': alignment_poss,
                        'loss_type': [],
                    }
                    if getattr(args, 'teacher_obs_mask', False):
                        # Residual-objective teacher mask: obs rows blind to the
                        # question image (see build_teacher_obs_mask docstring).
                        m4d = build_teacher_obs_mask(
                            batch['teacher_input_ids'], batch['teacher_attention_mask'],
                            alignment_poss)
                        inputs['attention_mask_4d'] = {"full_attention": m4d.to(device)}
                    if args.output_latent_embeds:
                        inputs['output_latent_embeds'] = True
                    if args.output_hidden_states:
                        inputs['output_hidden_states'] = True

                    outputs = model(**inputs, return_dict=True)

                    if args.output_latent_embeds: # output latent embeddings only for last layer
                        teacher_reps = outputs.latent_embeds
                    elif args.output_hidden_states: # output hidden states for all layers (can also output hidden states of all layers for latents)
                        teacher_reps = outputs.hidden_states
                    # Save per global sample index to avoid collisions
                    B = len(teacher_reps)
                    for b in range(B):
                        metadata = batch['metadata'][b]
                        dataset_name = metadata['dataset_name']
                        sample_id = metadata['sample_id']
                        if args.output_latent_embeds:
                            metadata_info = f"last_layer_{dataset_name}_{sample_id}"
                        elif args.output_hidden_states:
                            metadata_info = f"all_layers_{dataset_name}_{sample_id}"
                        if args.sft_stage2_align_poss == 'obs':
                            metadata_str = f"rep_{metadata_info}.pt"
                        elif args.sft_stage2_align_poss == 'latent_end':
                            metadata_str = f"rep_latent_end_{metadata_info}.pt"
                        save_path = os.path.join(out_dir, metadata_str)
                        rep = teacher_reps[b].detach()
                        if keep_layers is not None and rep.dim() == 3:  # [num_layer, T, D]
                            rep = rep[keep_layers]
                        # fp16 halves cache size; cosine/margin losses are insensitive at this precision
                        torch.save({'metadata_info': metadata_info,
                                    'latent': rep.to(torch.float16).cpu(),
                                    'keep_layers': keep_layers}, save_path)
                except Exception as e:
                    logging.exception(f"[rank {rank}] Failed at batch start={i}, ids={cur_ids}: {e}")
                    # Continue processing other batches instead of crashing and hanging other ranks
                    continue

        # Avoid a final barrier; log completion per-rank to prevent deadlocks if any rank terminated early
        logging.info(f"[precompute] rank {rank} done. Latents saved under: {out_dir}")
    finally:
        if is_dist and dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as e:
                logging.warning(f"[precompute] destroy_process_group failed: {e}")


if __name__ == "__main__":
    main()