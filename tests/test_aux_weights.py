"""The L_dec decoder must never land in the model weights (vLLM refuses the
checkpoint; job 15482092). Covers the split helper the trainer's _save uses
and the repair script for already-written checkpoints."""
import json
import os

import pytest

from src.aux_weights import AUX_PREFIXES, split_aux_state_dict


def test_split_moves_only_prefixed_keys_and_keeps_order():
    sd = {"model.embed_tokens.weight": 1, "latent_obs_decoder.tok_emb.weight": 2,
          "lm_head.weight": 3, "latent_obs_decoder.layers.0.q.weight": 4}
    model, aux = split_aux_state_dict(sd)
    assert list(model) == ["model.embed_tokens.weight", "lm_head.weight"]
    assert list(aux) == ["latent_obs_decoder.tok_emb.weight", "latent_obs_decoder.layers.0.q.weight"]
    assert AUX_PREFIXES == ("latent_obs_decoder.",)


def test_split_is_noop_without_aux_keys():
    sd = {"a": 1, "b": 2}
    model, aux = split_aux_state_dict(sd)
    assert model == sd and aux == {}


def test_prefix_must_match_at_key_start():
    sd = {"x.latent_obs_decoder.w": 1}
    model, aux = split_aux_state_dict(sd)
    assert aux == {} and "x.latent_obs_decoder.w" in model


torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors")


def _write_sharded_ckpt(d):
    from safetensors.torch import save_file
    s1 = {"model.a": torch.zeros(4), "latent_obs_decoder.w": torch.ones(3)}
    s2 = {"model.b": torch.zeros(2), "latent_obs_decoder.v": torch.ones(5)}
    save_file(s1, os.path.join(d, "model-00001-of-00002.safetensors"), metadata={"format": "pt"})
    save_file(s2, os.path.join(d, "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
    total = sum(t.numel() * t.element_size() for t in {**s1, **s2}.values())
    index = {"metadata": {"total_size": total},
             "weight_map": {k: "model-00001-of-00002.safetensors" for k in s1}
                           | {k: "model-00002-of-00002.safetensors" for k in s2}}
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)


def test_strip_sharded_checkpoint_in_place_is_reversible(tmp_path):
    from safetensors.torch import load_file
    from src.strip_aux_weights import strip_checkpoint
    d = str(tmp_path)
    _write_sharded_ckpt(d)

    dry = strip_checkpoint(d, dry_run=True)
    assert set(dry) == {"latent_obs_decoder.w", "latent_obs_decoder.v"}
    assert "latent_obs_decoder.w" in load_file(os.path.join(d, "model-00001-of-00002.safetensors"))

    removed = strip_checkpoint(d)
    assert set(removed) == {"latent_obs_decoder.w", "latent_obs_decoder.v"}
    for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        assert not any(k.startswith("latent_obs_decoder.") for k in load_file(os.path.join(d, shard)))
    index = json.load(open(os.path.join(d, "model.safetensors.index.json")))
    assert set(index["weight_map"]) == {"model.a", "model.b"}
    assert index["metadata"]["total_size"] == (4 + 2) * 4
    aux = torch.load(os.path.join(d, "aux_weights.pt"))
    assert torch.equal(aux["latent_obs_decoder.v"], torch.ones(5))

    assert strip_checkpoint(d) == {}   # second pass: nothing left to strip
