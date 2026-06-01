"""
Training script — General World Cuisine Model
Architecture : YOLO11l-cls (Ultralytics YOLO11, Large, Classification head)
Hardware     : NVIDIA RTX 3060 Ti (8 GB VRAM)
Target       : 132-class food classification, Top-1 accuracy priority
Domain       : Smartphone photos in classroom environments
               (fluorescent lighting, shadows, blur, odd angles)

Dataset      : datasets/general_model_flattened — the cleaned, leakage-free,
               132-class label space (merged sub-classes, normalized labels,
               pork/paprika/soy_beans removed, exact+perceptual de-duplication,
               re-split 70/20/10). See index.html (section 3).

Accuracy levers applied vs the previous run:
  - imgsz 224 -> 320  (largest single gain for fine-grained food)
  - epochs 120 -> 150 with patience 30 and an explicit cosine LR schedule
  - cleaner 132-class label space (done upstream in the flattening pipeline)
Domain-shift augmentation is intentionally kept heavy (classroom robustness),
which trades some clean-test Top-1 for real-world deployability.
"""

import os
import numpy as np
import torch
import albumentations as A
from ultralytics import YOLO

# ─── Paths ────────────────────────────────────────────────────────────────────
DATASET_DIR = r"E:\final project - models retrain\datasets\general_model_flattened"
PROJECT_DIR = r"E:\final project - models retrain\runs"
RUN_NAME    = "general_model_flattened"   # new run; does NOT overwrite ...yolo11l5

IMGSZ  = 320   # raised from 224: biggest accuracy lever for fine-grained food
BATCH  = 16    # lowered from 32 so 320 px fits the 8 GB VRAM budget under AMP
EPOCHS = 150   # raised from 120; early stopping (patience=30) ends it sooner if converged

# ─── Albumentations blur transform (defined at module level so worker
#     processes can pickle it on Windows without re-importing __main__) ────────
blur_transform = A.Compose([
    A.OneOf([
        A.MotionBlur(blur_limit=(3, 9), p=1.0),
        A.Defocus(radius=(1, 5), alias_blur=0.1, p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    ], p=0.35),
])

def apply_blur_augmentation(trainer):
    if hasattr(trainer, "batch") and trainer.batch is not None:
        imgs    = trainer.batch["img"]
        imgs_np = (imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
        augmented = [blur_transform(image=img)["image"] for img in imgs_np]
        trainer.batch["img"] = torch.from_numpy(
            np.stack(augmented).transpose(0, 3, 1, 2).astype("float32") / 255.0
        ).to(imgs.device)


# ─── Windows multiprocessing requires all spawning to happen inside this guard ─
if __name__ == "__main__":
    assert os.path.isdir(DATASET_DIR), f"Dataset directory not found: {DATASET_DIR}"
    assert torch.cuda.is_available(),  "CUDA not detected."

    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}  |  "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model = YOLO("yolo11l-cls.pt")
    model.add_callback("on_train_batch_start", apply_blur_augmentation)

    results = model.train(
        data       = DATASET_DIR,
        task       = "classify",

        epochs     = EPOCHS,
        imgsz      = IMGSZ,
        batch      = BATCH,
        workers    = 4,
        device     = 0,

        amp        = True,
        cache      = False,

        optimizer      = "AdamW",
        lr0            = 0.001,
        lrf            = 0.01,
        cos_lr         = True,        # explicit cosine decay -> sharper final minimum
        momentum       = 0.937,
        weight_decay   = 0.0005,
        warmup_epochs  = 5,
        warmup_momentum= 0.8,

        dropout         = 0.2,
        label_smoothing = 0.1,

        patience    = 30,
        save        = True,
        save_period = 10,

        hsv_h = 0.02,
        hsv_s = 0.75,
        hsv_v = 0.50,
        degrees     = 20.0,
        translate   = 0.15,
        scale       = 0.60,
        shear       = 8.0,
        perspective = 0.0008,
        flipud  = 0.15,
        fliplr  = 0.50,
        erasing = 0.40,
        mixup   = 0.10,
        mosaic  = 0.0,

        project    = PROJECT_DIR,
        name       = RUN_NAME,
        exist_ok   = False,
        pretrained = True,
        verbose    = True,
    )

    print("\n[DONE] Training complete.")
    print(f"       Best weights : {PROJECT_DIR}\\{RUN_NAME}\\weights\\best.pt")
    print(f"       Results CSV  : {PROJECT_DIR}\\{RUN_NAME}\\results.csv")
