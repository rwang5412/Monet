"""Smoke test: does base Monet-7B actually emit latents on these datasets?

The do(Z) harness is meaningless if the model never enters latent mode, and
latent emission is gated by the system prompt. RUN THIS FIRST on Palmetto. It
generates on a handful of samples (no intervention) and reports how often the
latent-start token (151666) appears. If the rate is ~0, stop and fix the prompt
/ LATENT_SIZE / checkpoint before building anything downstream.
"""

import argparse
import os

from evaluation.capimagine.harness import (
    SYSTEM_PROMPT, MONET_FORMAT_SUFFIX, LATENT_START_ID, _build_messages)

LATENT_END_ID = 151667


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="vstar", choices=["vstar", "hrbench_4k"])
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--prompt-style", default="monet", choices=["monet", "legacy"],
                    help="monet = RL training prompt (default, in-distribution); "
                         "legacy = old custom system prompt")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy (can suppress the latent token); Monet inference "
                         "uses 0.1, RL rollouts used 0.5. When >0, top_p/top_k match "
                         "the canonical inference defaults.")
    ap.add_argument("--latent-size", type=int, default=10)
    args = ap.parse_args()

    os.environ.setdefault("LATENT_SIZE", str(args.latent_size))
    os.environ["MONET_LATENT_MODE"] = "off"  # pure observation

    import inference.apply_vllm_monet  # noqa: F401
    from vllm import SamplingParams
    from transformers import AutoProcessor
    from inference.load_and_gen_vllm import (
        vllm_mllm_init, vllm_mllm_process_batch_from_messages, vllm_generate)
    from evaluation.capimagine import datasets

    samples = datasets.load(args.dataset, limit=args.limit)
    messages = _build_messages(samples, args.prompt_style)

    mllm, _ = vllm_mllm_init(args.model, tp=args.tp, gpu_memory_utilization=args.gpu_mem,
                             max_model_len=args.max_model_len)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    # Greedy (temp=0) can deterministically starve the "sometimes" latent token.
    # When sampling, mirror the canonical inference top_p/top_k so emission is realistic.
    if args.temperature > 0:
        sp = SamplingParams(temperature=args.temperature, top_p=0.8, top_k=50,
                            max_tokens=2048, n=1, skip_special_tokens=False, seed=0)
    else:
        sp = SamplingParams(temperature=0.0, max_tokens=2048, n=1,
                            skip_special_tokens=False, seed=0)
    print(f"[smoke] prompt_style={args.prompt_style} temperature={args.temperature}")
    inputs = vllm_mllm_process_batch_from_messages(messages, processor)
    outputs = vllm_generate(inputs, sp, mllm)

    n_emit = 0
    for i, out in enumerate(outputs):
        ids = list(out.outputs[0].token_ids)
        starts = ids.count(LATENT_START_ID)
        n_lat = sum(1 for t in ids if t == LATENT_START_ID)
        emit = LATENT_START_ID in ids
        n_emit += emit
        print(f"[{i}] emit_latent={emit} n_start={starts} "
              f"has_end={LATENT_END_ID in ids} tokens={len(ids)}")
        print("     text:", out.outputs[0].text[:200].replace("\n", " "))

    rate = n_emit / max(len(outputs), 1)
    print(f"\nlatent-emission rate = {n_emit}/{len(outputs)} = {rate:.2f}")
    if rate == 0:
        print("!! Base Monet-7B emitted NO latents. do(Z) is undefined. "
              "Check: system prompt, LATENT_SIZE env, and that this is a latent "
              "checkpoint. Do not proceed to the harness.")


if __name__ == "__main__":
    main()
