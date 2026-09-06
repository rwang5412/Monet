"""Keep training-only modules OUT of the saved model weights.

Stage 3 attaches the L_dec decoder as `model.latent_obs_decoder` (src/main.py).
It is discarded at inference, but Trainer.save_model serialises the whole
state_dict, so every stage-3 checkpoint shipped with `latent_obs_decoder.*`
tensors -- and vLLM's strict loader refused the checkpoint outright:

    ValueError: There is no module or parameter named 'latent_obs_decoder'
                in Qwen2_5_VLForConditionalGeneration      (job 15482092)

That killed the free-generation do(Z) pass -- the project's verdict metric --
on the first checkpoint that ever had L_dec live. HF's from_pretrained only
warns on unexpected keys, which is why the gate and harvest never noticed.

`split_aux_state_dict` is used by the stage-3 trainer at save time (the aux
tensors go to a sidecar file) and by `src.strip_aux_weights` to repair
checkpoints that were already written.
"""
AUX_PREFIXES = ("latent_obs_decoder.",)


def split_aux_state_dict(state_dict, prefixes=AUX_PREFIXES):
    """Return (model_sd, aux_sd): tensors whose key starts with any prefix go to
    aux_sd, everything else to model_sd. Key order is preserved. Pure function
    over the mapping -- tensors are not copied."""
    prefixes = tuple(prefixes)
    aux = {k: v for k, v in state_dict.items() if k.startswith(prefixes)}
    model = {k: v for k, v in state_dict.items() if k not in aux}
    return model, aux
