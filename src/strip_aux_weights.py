"""Remove training-only tensors (default: the stage-3 L_dec decoder) from an
already-saved HF checkpoint so vLLM can load it. See src/aux_weights.py.

    python -m src.strip_aux_weights --ckpt /scratch/$USER/monet_ckpts/sft_stage3_decode1.0_latent8_full
    python -m src.strip_aux_weights --ckpt DIR --dry_run          # report only

Edits the shards IN PLACE and writes the removed tensors to DIR/aux_weights.pt,
so the operation is reversible. Handles sharded safetensors (with index),
a single model.safetensors, and a single pytorch_model.bin.
"""
import argparse
import json
import os

import torch

from src.aux_weights import AUX_PREFIXES, split_aux_state_dict


def _strip_safetensors_file(path, prefixes, dry_run):
    from safetensors.torch import load_file, save_file
    sd = load_file(path)
    model_sd, aux_sd = split_aux_state_dict(sd, prefixes)
    if aux_sd and not dry_run:
        save_file(model_sd, path, metadata={"format": "pt"})
    return aux_sd


def strip_checkpoint(ckpt, prefixes=AUX_PREFIXES, dry_run=False):
    removed = {}
    index_path = os.path.join(ckpt, "model.safetensors.index.json")
    single_st = os.path.join(ckpt, "model.safetensors")
    single_bin = os.path.join(ckpt, "pytorch_model.bin")

    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        wmap = index["weight_map"]
        hit_keys = [k for k in wmap if k.startswith(tuple(prefixes))]
        for shard in sorted({wmap[k] for k in hit_keys}):
            removed.update(_strip_safetensors_file(os.path.join(ckpt, shard), prefixes, dry_run))
        if removed and not dry_run:
            for k in removed:
                wmap.pop(k, None)
            meta = index.get("metadata", {})
            if "total_size" in meta:
                meta["total_size"] -= sum(t.numel() * t.element_size() for t in removed.values())
            with open(index_path, "w") as f:
                json.dump(index, f, indent=2)
    elif os.path.exists(single_st):
        removed = _strip_safetensors_file(single_st, prefixes, dry_run)
    elif os.path.exists(single_bin):
        sd = torch.load(single_bin, map_location="cpu", weights_only=True)
        model_sd, removed = split_aux_state_dict(sd, prefixes)
        if removed and not dry_run:
            torch.save(model_sd, single_bin)
    else:
        raise FileNotFoundError(f"no model weights found in {ckpt}")

    if removed and not dry_run:
        torch.save(removed, os.path.join(ckpt, "aux_weights.pt"))
    return removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prefix", action="append", default=None,
                    help="key prefix to strip (repeatable); default: latent_obs_decoder.")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    prefixes = tuple(a.prefix) if a.prefix else AUX_PREFIXES
    removed = strip_checkpoint(a.ckpt, prefixes, a.dry_run)
    verb = "would remove" if a.dry_run else "removed"
    if not removed:
        print(f"nothing to strip in {a.ckpt} (prefixes {prefixes})")
        return
    n = sum(t.numel() for t in removed.values())
    print(f"{verb} {len(removed)} tensors / {n/1e6:.1f}M params matching {prefixes}")
    for k in list(removed)[:8]:
        print(f"  {k} {tuple(removed[k].shape)}")
    if len(removed) > 8:
        print(f"  ... {len(removed)-8} more")
    if not a.dry_run:
        print(f"aux tensors saved -> {os.path.join(a.ckpt, 'aux_weights.pt')}")


if __name__ == "__main__":
    main()
