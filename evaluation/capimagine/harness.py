"""CapImagine do(Z) accuracy harness for base Monet-7B.

Reproduces the paper's actual metric: corrupt the continuous latents Z during
*free generation* and measure the change in task accuracy. If the answer really
depends on Z, destroying Z should drop accuracy (Delta < 0). If Delta ~ 0 the
latents are cosmetic -- the CapImagine finding we expect on the base model.

One mode per process (vLLM reads its latent-hook env once at construction):

    clean          model unperturbed; captures mu/sigma of Z; the do(Z) baseline
    corrupt_gauss  every latent replaced with N(mu, sigma)  (destroy)
    corrupt_mean   every latent replaced with the global mean mu  (collapse)

A corrupt run loads the matching clean run, aligns by sample id, and reports:
  * clean_acc / corrupt_acc / delta   (over clean-latent-emitting samples only)
  * frac_text_changed -- the guard: under a destructive corruption the generated
    text MUST change. If it is ~0 while accuracy is flat, the hook is not firing
    (a bug), NOT the "latents are cosmetic" finding.

Usage (run clean first, then each corruption):
    python -m evaluation.capimagine.harness --model $CKPT --dataset vstar \
        --mode clean --out-dir $OUT
    python -m evaluation.capimagine.harness --model $CKPT --dataset vstar \
        --mode corrupt_gauss --out-dir $OUT
"""

import argparse
import json
import os

SYSTEM_PROMPT = (
    "You are a helpful multimodal assistant. You are required to answer the "
    "question based on the image provided. Put your final answer in \\boxed{}."
)
# Exact RL training prompt: RL/examples/format_prompt/monet_format.jinja, wired in
# via RL/examples/config_monet.yaml. verl appends this suffix inline to the user
# question and leaves the system prompt at the template default ("You are a helpful
# assistant."). Both SFT (src/task.py) and RL trained on the default system, so a
# custom system role is off-distribution and suppresses latent emission on the RL
# model. The literal "\n"/"\boxed" match what verl's Template.render actually emits.
MONET_FORMAT_SUFFIX = (
    " You FIRST think about the reasoning process as an internal monologue and then "
    "provide the final answer.\\nRemember: 1. Solve the problem step by step.\\n2. You "
    "MUST put the final answer in \\boxed{}."
)
LATENT_START_ID = 151666

# The paper's evaluation prompt, verbatim from arXiv:2511.21395v2 Appendix C
# ("Detailed Experimental Setup"): the authors run VLMEvalKit with this exact
# system message, max visual tokens 8192x28x28, and score with a rule-based judge
# (exact match) FIRST, then DeepSeek-V3.1 / Gemini-2.5-Pro as SECONDARY judges.
# This is the authoritative config that produced Monet-SFT V* = 82.20.
# NOTE: the repo README (§Evaluation) gives a DIFFERENT prompt ("You are a helpful
# multimodal assistant...") -- that is SYSTEM_PROMPT above. The two sources conflict;
# Appendix C wins because it is the exact setup behind the reported numbers. The
# boxed instruction is not in the paper's system prompt, so we append it inline in
# the user turn, matching how the Monet-SFT-125K training questions are formatted.
PAPER_EVAL_SYSTEM = (
    "You are an expert multimodal large language model designed to reason with "
    "latent visual embeddings."
)
PAPER_EVAL_BOXED = "\nPut your final answer within \\boxed{}."


# Resolution, matched to the paper (max visual tokens 8192x28x28). We set these
# PER-IMAGE in the message dict -- the path qwen-vl-utils/process_vision_info always
# honors, exactly like VLMEvalKit's create_image_content. This is more reliable than
# engine-level mm_processor_kwargs, which vLLM V1 can silently drop.
MAX_PIXELS = 8192 * 28 * 28
MIN_PIXELS = 256 * 28 * 28


def _image_item(sample):
    return {"type": "image", "image": sample.image,
            "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS}


def _vlmevalkit_user_text(sample):
    """Byte-for-byte VLMEvalKit ImageMCQDataset.build_prompt (V* has no hint):
        Question: {stem}\\nOptions:\\nA. {a}\\nB. {b}\\n...\\nPlease select the
        correct answer from the options above. \\n
    No boxed here -- VLMEvalKit carries the boxed instruction in the SYSTEM prompt.
    The trailing 'above. \\n' (space + newline) matches VLMEvalKit exactly."""
    opts = (sample.meta or {}).get("options") or {}
    if not opts:
        return f"Question: {sample.question}\n"
    stem = (sample.meta or {}).get("stem") or sample.question
    lines = "".join(f"{k}. {opts[k]}\n" for k in sorted(opts))
    return (f"Question: {stem}\nOptions:\n{lines}"
            "Please select the correct answer from the options above. \n")


def _build_messages(samples, prompt_style="monet"):
    if prompt_style == "vlmevalkit":
        # Mirror the paper's VLMEvalKit path byte-for-byte: image FIRST, then the
        # clean MCQ text (no boxed), per-image resolution, and the README system
        # prompt -- which is the one the README pairs with VLMEvalKit and which
        # carries the "Put your final answer in \boxed{}" instruction.
        return [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    _image_item(s),
                    {"type": "text", "text": _vlmevalkit_user_text(s)},
                ]},
            ]
            for s in samples
        ]
    if prompt_style == "paper_eval":
        # Appendix C system prompt + the (cleaned) question with boxed appended inline.
        return [
            [
                {"role": "system", "content": PAPER_EVAL_SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": s.question + PAPER_EVAL_BOXED},
                    _image_item(s),
                ]},
            ]
            for s in samples
        ]
    if prompt_style == "monet":
        # In-distribution: default system prompt, training suffix inline in the user turn.
        return [
            [
                {"role": "user", "content": [
                    {"type": "text", "text": s.question + MONET_FORMAT_SUFFIX},
                    _image_item(s),
                ]},
            ]
            for s in samples
        ]
    # legacy: the README's system prompt (kept for comparison).
    return [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": s.question},
                _image_item(s),
            ]},
        ]
        for s in samples
    ]


def _emitted_latent(output) -> bool:
    try:
        return LATENT_START_ID in list(output.outputs[0].token_ids)
    except Exception:
        return False


def run(args):
    # Env MUST be set before vLLM (and the runner's latent hook) is constructed.
    # User-facing "clean" maps to the hook's "capture" mode.
    os.environ["MONET_LATENT_MODE"] = "capture" if args.mode == "clean" else args.mode
    os.environ["MONET_LATENT_STATS"] = args.stats
    os.environ["MONET_LATENT_SEED"] = str(args.seed)
    os.environ.setdefault("LATENT_SIZE", str(args.latent_size))

    # Patch vLLM (sets LATENT_START_ID/END_ID + swaps the runner) before import.
    import inference.apply_vllm_monet  # noqa: F401
    from vllm import SamplingParams
    from transformers import AutoProcessor
    from inference.load_and_gen_vllm import (
        vllm_mllm_init, vllm_mllm_process_batch_from_messages, vllm_generate)
    from evaluation.capimagine import datasets, scoring

    samples = datasets.load(args.dataset, limit=args.limit)
    print(f"[harness] {args.dataset}: {len(samples)} samples, mode={args.mode}")

    mllm, _ = vllm_mllm_init(
        args.model, tp=args.tp, gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    # Greedy (temp=0) gives the cleanest do(Z) Delta -- clean vs corrupt differences
    # are attributable to the latent, not sampling. But greedy can deterministically
    # starve the "sometimes" latent token (see the smoke test); when emission needs
    # sampling, use temp>0 with a fixed seed and mirror the canonical inference top_p/k.
    if args.temperature > 0:
        sampling_params = SamplingParams(
            temperature=args.temperature, top_p=0.8, top_k=50,
            max_tokens=args.max_tokens, n=1, skip_special_tokens=False, seed=0)
    else:
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=args.max_tokens, n=1,
            skip_special_tokens=False, seed=0)
    print(f"[harness] prompt_style={args.prompt_style} temperature={args.temperature}")

    inputs = vllm_mllm_process_batch_from_messages(
        _build_messages(samples, args.prompt_style), processor)
    outputs = vllm_generate(inputs, sampling_params, mllm)

    # Flush the final capture stats (the runner dumps periodically during the run).
    if args.mode == "clean":
        try:
            from inference.vllm.monet_latent_hook import get_hook
            get_hook().finalize()
        except Exception as e:
            print(f"[harness] stats finalize skipped: {e!r}")

    records = []
    for s, out in zip(samples, outputs):
        text = out.outputs[0].text
        records.append({
            "id": s.id,
            "gold": s.answer_letter,
            "pred": scoring.pred_letter(text),
            "correct": scoring.is_correct(text, s.answer_letter),
            "emitted_latent": _emitted_latent(out),
            "text": text,
        })

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.dataset}_{args.mode}.json")
    with open(out_path, "w") as f:
        json.dump({"dataset": args.dataset, "mode": args.mode,
                   "model": args.model, "records": records}, f, indent=2)
    print(f"[harness] wrote {out_path}")

    _summarize(args, records)


def _summarize(args, records):
    n = len(records)
    n_emit = sum(r["emitted_latent"] for r in records)
    acc_all = sum(r["correct"] for r in records) / max(n, 1)
    emit = [r for r in records if r["emitted_latent"]]
    acc_emit = sum(r["correct"] for r in emit) / max(len(emit), 1)
    print(f"\n=== {args.dataset} / {args.mode} ===")
    print(f"N={n}  latent-emitting N={n_emit} ({100*n_emit/max(n,1):.1f}%)")
    print(f"accuracy (all)={acc_all:.4f}  accuracy (latent-emitting)={acc_emit:.4f}")

    if args.mode == "clean":
        if n_emit == 0:
            print("!! WARNING: base model emitted NO latents (151666). do(Z) is "
                  "undefined -- check the system prompt / LATENT_SIZE / checkpoint "
                  "before running any corruption pass.")
        return

    # Corruption pass: compare against the clean pass.
    clean_path = os.path.join(args.out_dir, f"{args.dataset}_clean.json")
    if not os.path.exists(clean_path):
        print(f"!! no clean run at {clean_path}; run --mode clean first for Delta.")
        return
    clean = {r["id"]: r for r in json.load(open(clean_path))["records"]}

    # do(Z) is defined only where the CLEAN pass emitted latents.
    valid = [r for r in records if clean.get(r["id"], {}).get("emitted_latent")]
    if not valid:
        print("!! no clean-latent-emitting samples to score do(Z) over.")
        return
    clean_acc = sum(clean[r["id"]]["correct"] for r in valid) / len(valid)
    corrupt_acc = sum(r["correct"] for r in valid) / len(valid)
    changed = sum(r["text"] != clean[r["id"]]["text"] for r in valid) / len(valid)
    flip_to_wrong = sum(clean[r["id"]]["correct"] and not r["correct"] for r in valid)
    flip_to_right = sum((not clean[r["id"]]["correct"]) and r["correct"] for r in valid)

    print(f"\n--- do(Z): {args.mode} ---")
    print(f"scored N (clean-latent-emitting) = {len(valid)}")
    print(f"clean_acc   = {clean_acc:.4f}")
    print(f"corrupt_acc = {corrupt_acc:.4f}")
    print(f"DELTA       = {corrupt_acc - clean_acc:+.4f}   "
          f"(<0 => latents load-bearing; ~0 => cosmetic)")
    print(f"flip_to_wrong={flip_to_wrong}  flip_to_right={flip_to_right}")
    print(f"frac_text_changed = {changed:.3f}   "
          f"(GUARD: must be well > 0 under corruption, else the hook is not firing)")
    if changed < 0.05:
        print("!! GUARD FAILED: corrupted text is ~identical to clean. The "
              "intervention hook is almost certainly not firing -- do NOT read "
              "Delta~0 as the CapImagine finding until this is fixed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to base Monet-7B checkpoint")
    ap.add_argument("--dataset", required=True, choices=["vstar", "hrbench_4k"])
    ap.add_argument("--mode", required=True,
                    choices=["clean", "corrupt_gauss", "corrupt_mean"])
    ap.add_argument("--prompt-style", default="monet",
                    choices=["monet", "legacy", "paper_eval", "vlmevalkit"],
                    help="vlmevalkit = mirror the paper's VLMEvalKit path (clean MCQ "
                         "prompt + per-image resolution + expert system prompt); "
                         "monet = RL training prompt; paper_eval = expert system prompt "
                         "with the raw question; legacy = the README's prompt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stats", default=None,
                    help="mu/sigma stats path (default: <out-dir>/<dataset>_stats.pt)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy (cleanest do(Z) Delta); use 0.1 if greedy "
                         "starves latent emission. top_p/top_k mirror inference when >0.")
    ap.add_argument("--latent-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.stats is None:
        args.stats = os.path.join(args.out_dir, f"{args.dataset}_stats.pt")
    os.makedirs(args.out_dir, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
