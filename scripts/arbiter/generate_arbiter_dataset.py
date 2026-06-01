"""
Arbiter dataset generator — CalEyeZ router training data.

Runs BOTH production models over the held-out (val + test) images of BOTH domains
and writes one row per image with each model's full Top-5 guess, confidences, derived
features, correctness flags, and the ground-truth answer. This is the clean, inspectable
table the XGBoost arbiter (router) is trained on.

Why val + test only:
    The global model overfits its TRAIN split (near-100% confidence there), so features
    extracted from train images would be miscalibrated vs. deployment. val + test are
    unseen by the global model -> realistic confidence/entropy/margin features.

Both models run on EVERY image (both domains): for a global-domain image we still record
what the Israeli model predicted (and how confident it was on out-of-domain input), and
vice-versa. That cross-domain confidence is exactly the signal the router learns from.

Label spaces are disjoint (132 global classes vs 13 Israeli), so at most one model can be
correct on any image -> a clean routing target ('global' / 'israeli' / 'none').

Output columns (one row per image):
    image_path, domain, true_class,
    g_top1..g_top5, g_conf1..g_conf5, g_entropy, g_margin, g_correct,
    i_top1..i_top5, i_conf1..i_conf5, i_entropy, i_margin, i_correct,
    winner
"""

from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent.parent
GLOBAL_W = ROOT / "runs" / "general_model_flattened" / "weights" / "best.pt"
ISRAELI_W = ROOT / "runs" / "israeli_food_yolo11l" / "weights" / "best.pt"
GLOBAL_DS = ROOT / "datasets" / "general_model_flattened"
ISRAELI_DS = ROOT / "datasets" / "israeli-food-master"
OUT = ROOT / "datasets" / "arbiter_dataset.csv"

GLOBAL_IMGSZ = 320
ISRAELI_IMGSZ = 224
SPLITS = ("val", "test")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BATCH = 256


def collect(ds_root: Path, domain: str) -> list[tuple[str, str, str]]:
    """Return [(image_path, domain, true_class), ...] over val+test."""
    out = []
    for split in SPLITS:
        sdir = ds_root / split
        if not sdir.is_dir():
            continue
        for cls_dir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            for img in cls_dir.iterdir():
                if img.suffix.lower() in IMG_EXTS:
                    out.append((str(img), domain, cls_dir.name))
    return out


def predict_all(model: YOLO, paths: list[str], imgsz: int) -> list[dict]:
    """Run model over all paths; return per-image top-5 names/confs + entropy + margin."""
    names = model.names
    rows: list[dict] = []
    for i in range(0, len(paths), BATCH):
        chunk = paths[i:i + BATCH]
        res = model.predict(chunk, imgsz=imgsz, verbose=False, device=0)
        for r in res:
            probs = r.probs
            top5_idx = probs.top5
            top5_conf = [float(c) for c in probs.top5conf.tolist()]
            full = probs.data.cpu().numpy().astype(np.float64)
            full = np.clip(full, 1e-12, 1.0)
            entropy = float(-(full * np.log(full)).sum())
            margin = float(top5_conf[0] - top5_conf[1]) if len(top5_conf) > 1 else float(top5_conf[0])
            rows.append({
                "top": [names[int(j)] for j in top5_idx],
                "conf": top5_conf,
                "entropy": entropy,
                "margin": margin,
            })
        print(f"    {min(i + BATCH, len(paths))}/{len(paths)}", end="\r", flush=True)
    print()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap images per domain (testing)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    for w in (GLOBAL_W, ISRAELI_W):
        if not w.exists():
            raise SystemExit(f"Missing weights: {w}")

    print("Collecting held-out (val+test) images from both domains...")
    items = collect(GLOBAL_DS, "global") + collect(ISRAELI_DS, "israeli")
    if args.limit:
        g = [x for x in items if x[1] == "global"][:args.limit]
        isr = [x for x in items if x[1] == "israeli"][:args.limit]
        items = g + isr
    paths = [x[0] for x in items]
    n_g = sum(1 for x in items if x[1] == "global")
    n_i = len(items) - n_g
    print(f"  total {len(items)} images  (global {n_g}, israeli {n_i})")

    print("Loading models...")
    gm = YOLO(str(GLOBAL_W))
    im = YOLO(str(ISRAELI_W))

    print(f"Running GLOBAL model (imgsz {GLOBAL_IMGSZ}) over all images...")
    g_rows = predict_all(gm, paths, GLOBAL_IMGSZ)
    print(f"Running ISRAELI model (imgsz {ISRAELI_IMGSZ}) over all images...")
    i_rows = predict_all(im, paths, ISRAELI_IMGSZ)

    header = (["image_path", "domain", "true_class"]
              + [f"g_top{k}" for k in range(1, 6)] + [f"g_conf{k}" for k in range(1, 6)]
              + ["g_entropy", "g_margin", "g_correct"]
              + [f"i_top{k}" for k in range(1, 6)] + [f"i_conf{k}" for k in range(1, 6)]
              + ["i_entropy", "i_margin", "i_correct", "winner"])

    n_gcorr = n_icorr = n_none = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for (path, domain, true_cls), g, ir in zip(items, g_rows, i_rows):
            g_corr = int(g["top"][0] == true_cls)
            i_corr = int(ir["top"][0] == true_cls)
            winner = "global" if g_corr else ("israeli" if i_corr else "none")
            n_gcorr += g_corr; n_icorr += i_corr; n_none += int(winner == "none")
            row = ([path, domain, true_cls]
                   + g["top"] + [f"{c:.6f}" for c in g["conf"]]
                   + [f"{g['entropy']:.6f}", f"{g['margin']:.6f}", g_corr]
                   + ir["top"] + [f"{c:.6f}" for c in ir["conf"]]
                   + [f"{ir['entropy']:.6f}", f"{ir['margin']:.6f}", i_corr, winner])
            w.writerow(row)

    print(f"\nWrote {len(items)} rows -> {args.out}")
    print("=== Routing-label summary ===")
    print(f"  global correct : {n_gcorr:>6}  ({n_gcorr/len(items)*100:.1f}%)")
    print(f"  israeli correct: {n_icorr:>6}  ({n_icorr/len(items)*100:.1f}%)")
    print(f"  neither (none) : {n_none:>6}  ({n_none/len(items)*100:.1f}%)")
    print("  (oracle ceiling = global+israeli correct = "
          f"{(n_gcorr+n_icorr)/len(items)*100:.1f}%)")


if __name__ == "__main__":
    main()
