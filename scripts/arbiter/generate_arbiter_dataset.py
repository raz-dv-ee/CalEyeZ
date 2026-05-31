"""
generate_arbiter_dataset.py
----------------------------
Runs inference with the Global and Israeli YOLO11l-cls models on the val/test
splits of both datasets and writes a feature-rich CSV for XGBoost training.

Output columns
--------------
image_path, split_type, ground_truth_class,
global_top1_class, global_top1_conf, global_top2_class, global_top2_conf,
local_top1_class,  local_top1_conf,  local_top2_class,  local_top2_conf,
arbiter_label

arbiter_label encoding
  0  →  ONLY Global correct
  1  →  ONLY Israeli correct
  2  →  BOTH correct
  3  →  NEITHER correct
"""

import csv
import os
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# HARDCODED PATHS
# ──────────────────────────────────────────────────────────────────────────────
GLOBAL_WEIGHTS  = r"E:\final project - models retrain\runs\general_model_yolo11l5\weights\best.pt"
ISRAELI_WEIGHTS = r"E:\final project - models retrain\runs\israeli_food_yolo11l\weights\best.pt"

DATASETS = {
    "global":  r"E:\final project - models retrain\datasets\general_model_data_set",
    "israeli": r"E:\final project - models retrain\datasets\israeli-food-master",
}

OUTPUT_CSV = r"E:\final project - models retrain\arbiter_training_dataset.csv"

# Classification inference image size (must match training resolution)
IMGSZ = 224

# Supported image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_model(weights_path: str, label: str) -> YOLO:
    path = Path(weights_path)
    if not path.exists():
        print(f"[ERROR] {label} weights not found: {weights_path}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO]  Loading {label} model from {weights_path} …")
    model = YOLO(str(path))
    model.fuse()          # fuse Conv+BN layers for faster inference
    return model


def collect_images(dataset_root: str) -> list[tuple[Path, str, str]]:
    """
    Walk <dataset_root>/{val,test}/<class_name>/ and return
    a list of (image_path, split_type, ground_truth_class) tuples.
    train/ directories are intentionally skipped.
    """
    records = []
    root = Path(dataset_root)
    for split in ("val", "test"):
        split_dir = root / split
        if not split_dir.exists():
            print(f"[WARN]  Split directory not found, skipping: {split_dir}")
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix.lower() in IMG_EXTS:
                    records.append((img_file, split, class_name))
    return records


def run_inference(model: YOLO, image_path: Path) -> tuple[str, float, str, float]:
    """
    Run classification inference on a single image.
    Returns (top1_class, top1_conf, top2_class, top2_conf).

    model()  →  list[Results]  (always length-1 for single-image calls)
    results[0].probs  →  Probs object wrapping a softmax probability tensor
      .top5          →  list of top-5 class indices (int)
      .top5conf      →  corresponding confidence scores (torch.Tensor)
      .top1          →  int  index of argmax class
      .top1conf      →  float scalar tensor of the argmax confidence
    names dict maps int index → string label.
    """
    results = model(
        str(image_path),
        imgsz=IMGSZ,
        verbose=False,   # suppress per-image console output
    )

    probs  = results[0].probs          # ultralytics.engine.results.Probs
    names  = results[0].names          # dict {int: str}

    top5_indices = probs.top5          # list[int], sorted descending by conf
    top5_confs   = probs.top5conf      # torch.Tensor shape (5,)

    top1_idx  = top5_indices[0]
    top1_conf = float(top5_confs[0])
    top1_cls  = names[top1_idx]

    # Guard: if fewer than 2 classes exist in the model (edge case)
    if len(top5_indices) >= 2:
        top2_idx  = top5_indices[1]
        top2_conf = float(top5_confs[1])
        top2_cls  = names[top2_idx]
    else:
        top2_idx  = top1_idx
        top2_conf = top1_conf
        top2_cls  = top1_cls

    return top1_cls, top1_conf, top2_cls, top2_conf


def compute_arbiter_label(gt: str, g_top1: str, l_top1: str) -> int:
    """
    0  →  ONLY Global correct
    1  →  ONLY Israeli correct
    2  →  BOTH correct
    3  →  NEITHER correct

    Comparison is case-insensitive and strips surrounding whitespace to
    tolerate minor naming inconsistencies between datasets and model vocab.
    """
    gt_norm = gt.strip().lower()
    g_correct = (g_top1.strip().lower() == gt_norm)
    l_correct = (l_top1.strip().lower() == gt_norm)

    if g_correct and not l_correct:
        return 0
    if l_correct and not g_correct:
        return 1
    if g_correct and l_correct:
        return 2
    return 3


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO]  Using device: {device}")

    global_model  = load_model(GLOBAL_WEIGHTS,  "Global")
    israeli_model = load_model(ISRAELI_WEIGHTS, "Israeli")

    # Collect all images across both datasets (val + test only)
    all_records: list[tuple[Path, str, str]] = []
    for ds_name, ds_path in DATASETS.items():
        imgs = collect_images(ds_path)
        print(f"[INFO]  {ds_name:>8} dataset  →  {len(imgs)} images (val+test)")
        all_records.extend(imgs)

    total = len(all_records)
    print(f"[INFO]  Total images to process: {total}")
    if total == 0:
        print("[ERROR] No images found. Check dataset paths.", file=sys.stderr)
        sys.exit(1)

    csv_columns = [
        "image_path",
        "split_type",
        "ground_truth_class",
        "global_top1_class",
        "global_top1_conf",
        "global_top2_class",
        "global_top2_conf",
        "local_top1_class",
        "local_top1_conf",
        "local_top2_class",
        "local_top2_conf",
        "arbiter_label",
    ]

    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    errors = 0
    t_start = time.time()

    with open(output_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=csv_columns)
        writer.writeheader()

        for idx, (img_path, split_type, gt_class) in enumerate(all_records, start=1):

            # ── progress indicator ──────────────────────────────────────────
            if idx % 100 == 0 or idx == 1 or idx == total:
                elapsed  = time.time() - t_start
                per_img  = elapsed / idx
                remaining = per_img * (total - idx)
                print(
                    f"\r[INFO]  {idx:>5}/{total}  "
                    f"elapsed {elapsed:6.1f}s  "
                    f"ETA {remaining:6.1f}s",
                    end="", flush=True,
                )

            try:
                g_top1, g_top1c, g_top2, g_top2c = run_inference(global_model,  img_path)
                l_top1, l_top1c, l_top2, l_top2c = run_inference(israeli_model, img_path)
            except Exception as exc:
                print(f"\n[WARN]  Skipping {img_path.name}: {exc}")
                errors += 1
                continue

            label = compute_arbiter_label(gt_class, g_top1, l_top1)

            writer.writerow({
                "image_path":         str(img_path),
                "split_type":         split_type,
                "ground_truth_class": gt_class,
                "global_top1_class":  g_top1,
                "global_top1_conf":   round(g_top1c, 6),
                "global_top2_class":  g_top2,
                "global_top2_conf":   round(g_top2c, 6),
                "local_top1_class":   l_top1,
                "local_top1_conf":    round(l_top1c, 6),
                "local_top2_class":   l_top2,
                "local_top2_conf":    round(l_top2c, 6),
                "arbiter_label":      label,
            })

    print()  # newline after progress
    elapsed_total = time.time() - t_start
    written = total - errors
    print(f"\n[DONE]  Wrote {written} rows ({errors} skipped) in {elapsed_total:.1f}s")
    print(f"[DONE]  CSV saved to: {output_path}")

    # ── quick label-distribution summary ───────────────────────────────────
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    with open(output_path, "r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            lbl = int(row["arbiter_label"])
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

    print("\n[INFO]  Arbiter label distribution:")
    labels_desc = {
        0: "Only Global correct",
        1: "Only Israeli correct",
        2: "Both correct",
        3: "Neither correct",
    }
    for lbl, count in label_counts.items():
        pct = 100.0 * count / written if written else 0
        print(f"        Label {lbl} ({labels_desc[lbl]}): {count:>5}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
