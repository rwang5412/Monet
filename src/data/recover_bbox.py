"""Recover aux-image bounding boxes by template matching (Stage-3 Change A).

Monet's repackaged Visual_CoT rows carry no bbox, but the auxiliary image IS the
cropped region of the question image. So the box is recoverable: slide the aux
image over the original and take the best-matching location. Multi-scale search
handles the repackaging resize; the match score separates clean crops from
drawing-ops rows (ReFocus/CogCoM style), which are skipped per the Stage-3 spec.

Pilot first (prints per-sample score + bbox, no writes):
    python -m src.data.recover_bbox --data_path .../Visual_CoT/train.json \
        --dataset_root .../Monet-SFT-125K --n 20

Full sweep (writes JSONL sidecar keyed like the teacher caches):
    python -m src.data.recover_bbox ... --n -1 --out .../bbox_viscot.jsonl

Output record:
    {"key": "Visual_CoT_0", "bbox_norm": [x0, y0, x1, y1], "score": 0.98,
     "scale": 1.0, "orig_size": [W, H]}
bbox_norm is in [0,1] coords of the ORIGINAL image. Records with score below
--min_score are written with "usable": false (Change A skips them).
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

try:
    import cv2
    _HAS_CV2 = True
except ImportError:  # pure-numpy fallback is too slow for the full sweep
    _HAS_CV2 = False

SCALES = [1.0, 0.75, 0.5, 1.25, 1.5, 2.0]  # aux may have been resized in repackaging


def match_one(orig: Image.Image, aux: Image.Image):
    """Best normalized-cross-correlation match of aux inside orig over SCALES.
    Returns (score, bbox_pixels_on_orig, scale)."""
    assert _HAS_CV2, "cv2 required: pip install opencv-python-headless"
    o = cv2.cvtColor(np.array(orig.convert("RGB")), cv2.COLOR_RGB2GRAY)
    a0 = cv2.cvtColor(np.array(aux.convert("RGB")), cv2.COLOR_RGB2GRAY)
    best = (-1.0, None, None)
    for s in SCALES:
        w = max(4, int(round(a0.shape[1] * s)))
        h = max(4, int(round(a0.shape[0] * s)))
        if h >= o.shape[0] or w >= o.shape[1]:
            continue
        a = cv2.resize(a0, (w, h), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(o, a, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx > best[0]:
            best = (float(mx), (loc[0], loc[1], loc[0] + w, loc[1] + h), s)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--n", type=int, default=20, help="-1 = full sweep")
    ap.add_argument("--out", default=None, help="JSONL sidecar (full sweep)")
    ap.add_argument("--min_score", type=float, default=0.85)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    rows = json.load(open(args.data_path))
    if args.n > 0:
        rows = rows[:args.n]

    done = set()
    if args.out and args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(l)["key"] for l in f if l.strip()}

    fout = open(args.out, "a") if args.out else None
    n_ok = n_low = n_skip = 0
    scores = []
    for i, row in enumerate(rows):
        md = row["metadata"]
        key = f"{md['dataset_name']}_{md['sample_id']}"
        if key in done:
            continue
        imgs = [it["image"] for m in row["data"] for it in m.get("content", [])
                if it.get("type") == "image"]
        if len(imgs) != 2:   # exactly one question image + one aux
            n_skip += 1
            continue
        try:
            orig = Image.open(os.path.join(args.dataset_root, imgs[0]))
            aux = Image.open(os.path.join(args.dataset_root, imgs[1]))
            score, box, scale = match_one(orig, aux)
        except Exception as e:
            print(f"{key}: ERROR {e!r}")
            n_skip += 1
            continue
        W, H = orig.size
        rec = {"key": key, "usable": bool(box and score >= args.min_score),
               "score": round(score, 4), "scale": scale,
               "bbox_norm": ([box[0] / W, box[1] / H, box[2] / W, box[3] / H]
                             if box else None),
               "orig_size": [W, H]}
        if rec["usable"]:
            n_ok += 1
        else:
            n_low += 1
        scores.append(score)
        if fout:
            fout.write(json.dumps(rec) + "\n")
            if (n_ok + n_low) % 500 == 0:
                fout.flush()
                print(f"[{i+1}/{len(rows)}] usable={n_ok} low={n_low} skip={n_skip}")
        else:
            print(f"{key}: score={score:.3f} scale={scale} bbox_norm="
                  f"{[round(v,3) for v in rec['bbox_norm']] if rec['bbox_norm'] else None}"
                  f"  {'OK' if rec['usable'] else 'LOW'}")

    if fout:
        fout.close()
    s = sorted(scores)
    print(f"\ndone: usable={n_ok} low={n_low} skipped={n_skip}")
    if s:
        print(f"score p10/p50/p90 = {s[int(.1*len(s))]:.3f} / {s[len(s)//2]:.3f} / {s[int(.9*len(s))]:.3f}")


if __name__ == "__main__":
    main()
