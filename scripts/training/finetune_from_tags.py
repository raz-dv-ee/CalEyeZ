"""
Safe fine-tune from tagged real-world samples (the demo's data-tagging flywheel).

Safety guarantees:
  * NEVER overwrites a production model: the fine-tuned weights go to a NEW run dir.
  * Holds out a fraction of the tagged images for an honest real-world before/after measurement.
  * Adds the tagged TRAIN images into the dataset only TEMPORARILY (removed in a finally block), so
    the dataset is restored even if training crashes. Only train/<cls> is touched; val/test stay clean.
  * Regression check: re-validates on the original clean TEST split and aborts the recommendation if
    overall accuracy drops.

Example:
  python scripts/training/finetune_from_tags.py \
      --weights runs/israeli_food_yolo11l/weights/best.pt \
      --data datasets/israeli-food-master --cls hummus \
      --tagged tagging_data/images/hummus --imgsz 224 --epochs 15 --name israeli_ft
"""
from __future__ import annotations
import argparse
import glob
import os
import random
import shutil
from pathlib import Path

from ultralytics import YOLO


def acc_on(model, files, target, imgsz):
    ok = 0
    for f in files:
        r = model.predict(f, imgsz=imgsz, verbose=False)[0]
        ok += int(r.names[int(r.probs.top1)] == target)
    return ok, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cls", required=True, help="comma-separated class names")
    ap.add_argument("--tagged", required=True, help="base dir holding <cls>/ subfolders of tagged images")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr0", type=float, default=1e-4)
    ap.add_argument("--holdout", type=float, default=0.33)
    ap.add_argument("--oversample", type=int, default=2)
    ap.add_argument("--freeze", type=int, default=10, help="freeze first N layers (10 = backbone, head only)")
    ap.add_argument("--name", default="ft")
    args = ap.parse_args()

    classes = [c.strip() for c in args.cls.split(",") if c.strip()]
    random.seed(0)
    hold_by_cls = {}        # class -> list of held-out image paths
    added = []
    for cls in classes:
        tagged = sorted(glob.glob(os.path.join(args.tagged, cls, "*")))
        if len(tagged) < 4:
            print(f"[ft] WARNING: only {len(tagged)} tagged for {cls}, skipping")
            continue
        random.shuffle(tagged)
        n_hold = max(1, int(round(len(tagged) * args.holdout)))
        hold_by_cls[cls] = tagged[:n_hold]
        train_extra = tagged[n_hold:]
        train_dir = os.path.join(args.data, "train", cls)
        os.makedirs(train_dir, exist_ok=True)
        for k, src in enumerate(train_extra):
            for r in range(args.oversample):
                dst = os.path.join(train_dir, f"__tagft_{k}_{r}.jpg")
                shutil.copy(src, dst)
                added.append(dst)
        print(f"[ft] {cls}: {len(tagged)} tagged -> {len(train_extra)} train (x{args.oversample}), {len(hold_by_cls[cls])} held-out")
    if not added:
        raise SystemExit("No classes had enough tagged images.")
    print(f"[ft] temporarily added {len(added)} images total")

    ft_weights = None
    try:
        model = YOLO(args.weights)
        # optimizer='AdamW' (not 'auto') so our low lr0 is actually honoured; freeze the backbone so the
        # features the other 12 classes rely on cannot drift -> prevents catastrophic forgetting.
        model.train(data=args.data, imgsz=args.imgsz, epochs=args.epochs, lr0=args.lr0,
                    optimizer="AdamW", freeze=args.freeze, warmup_epochs=0.0,
                    pretrained=False, name=args.name, exist_ok=True, plots=False, verbose=False)
        ft_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
    finally:
        for f in added:
            try:
                os.remove(f)
            except OSError:
                pass
        print("[ft] removed temporary tagged copies (dataset restored)")

    print("\n================ RESULTS ================")
    orig = YOLO(args.weights)
    new = YOLO(str(ft_weights))

    improved_all = True
    for cls, hold in hold_by_cls.items():
        bo, nb = acc_on(orig, hold, cls, args.imgsz), acc_on(new, hold, cls, args.imgsz)
        print(f"held-out real-world '{cls}' (never trained on):  ORIGINAL {bo[0]}/{bo[1]}  ->  FINETUNED {nb[0]}/{nb[1]}")
        improved_all = improved_all and (nb[0] / max(nb[1], 1) >= bo[0] / max(bo[1], 1))

    print("\nregression check on the original clean TEST split:")
    o_top1 = float(orig.val(data=args.data, split="test", imgsz=args.imgsz, verbose=False).top1)
    n_top1 = float(new.val(data=args.data, split="test", imgsz=args.imgsz, verbose=False).top1)
    print(f"  ORIGINAL  test top-1: {o_top1*100:.2f}%")
    print(f"  FINETUNED test top-1: {n_top1*100:.2f}%   (delta {(n_top1-o_top1)*100:+.2f} pts)")

    print("\nVERDICT:")
    improved = improved_all
    safe = n_top1 >= o_top1 - 0.005          # allow <=0.5 pt noise
    print(f"  real-world improved: {improved}   |   clean test preserved: {safe}")
    print(f"  fine-tuned weights: {ft_weights}")
    print("  (production best.pt is UNTOUCHED; switch the demo only if both checks pass.)")


if __name__ == "__main__":
    main()
