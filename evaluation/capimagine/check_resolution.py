"""Check the actual image resolution a V* image reaches the model at (no GPU).

The paper sets max visual tokens = 8192*28*28. If our pipeline silently downscales
V* images below that, small-object perception collapses and V* accuracy drops flat
across every model -- exactly our symptom. This prints, per sample, the image grid
(t, h, w) and the resulting visual-token count under three settings:

  default   -- the processor's own preprocessor_config.json default (what vLLM V1
               falls back to if mm_processor_kwargs does NOT take effect)
  capped    -- max_pixels=8192*28*28 explicitly (what the harness INTENDS via
               mm_processor_kwargs in load_and_gen_vllm.vllm_mllm_init)

Token count = t*h*w / (merge_size**2)  (merge_size=2 for Qwen2.5-VL -> /4).
If `capped` ~= 8192 but `default` is much smaller, and vLLM is using the default,
that gap is the bug. Compare `default` to what you want (~8192).

    python -m evaluation.capimagine.check_resolution /path/to/Monet-SFT-7B/stage3
"""

import argparse

MAXP = 8192 * 28 * 28
MINP = 256 * 28 * 28


def _tokens(grid, merge=2):
    t, h, w = grid
    return (t * h * w) // (merge * merge)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    from transformers import AutoProcessor
    from evaluation.capimagine import datasets

    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    ip = proc.image_processor
    print(f"processor default: min_pixels={getattr(ip,'min_pixels',None)} "
          f"max_pixels={getattr(ip,'max_pixels',None)} "
          f"size={getattr(ip,'size',None)} merge_size={getattr(ip,'merge_size',None)}")
    print(f"harness intends (mm_processor_kwargs): min={MINP} max={MAXP} "
          f"(= 8192*28*28 max tokens)\n")

    samples = datasets.load("vstar", limit=args.n)
    for s in samples:
        w, h = s.image.size
        default = ip(images=s.image, return_tensors="pt")["image_grid_thw"][0].tolist()
        capped = ip(images=s.image, return_tensors="pt",
                    min_pixels=MINP, max_pixels=MAXP)["image_grid_thw"][0].tolist()
        print(f"{s.id}: orig {w}x{h}")
        print(f"    default grid={default}  tokens={_tokens(default)}")
        print(f"    capped  grid={capped}  tokens={_tokens(capped)}")


if __name__ == "__main__":
    main()
