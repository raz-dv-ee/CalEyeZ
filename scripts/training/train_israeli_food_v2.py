"""
Israeli food classifier - TRAINING v2
=====================================
What changed vs v1 (israeli_food_yolo11l):
  * DATA, not hyperparameters. We added real images to the data-starved weak classes
    (sabich, malawach, meorav_yerushalmi, bourekas_cheese) and a NEW 14th class "background"
    (non-Israeli food look-alikes pulled from the global set + a little kitchen/countertop).
  * The "background" class is an open-set-recognition (OSR) device: it lets the model answer
    "this is NOT one of my Israeli dishes" instead of confidently shouting one. That makes the
    Israeli model's confidence trustworthy, which is what the XGBoost arbiter routes on.
  * The whole dataset was byte-for-byte (SHA-256) de-duplicated: 0 cross-split leakage.

DESIGN PRINCIPLE: every training hyperparameter is held IDENTICAL to v1. Only the data and the
run name change. This is deliberate - it makes any metric change attributable to the data / OSR
change, not to hyperparameter tuning. (A clean, defensible A/B.)

After this run: regenerate the arbiter dataset with these weights, retrain the arbiter, re-evaluate.
Run:  python scripts/training/train_israeli_food_v2.py
"""
import os
import numpy as np
import torch
import albumentations as A
from ultralytics import YOLO

DATASET_DIR = r"E:\final project - models retrain\datasets\israeli-food-master"
PROJECT_DIR = r"E:\final project - models retrain\runs"
RUN_NAME    = "israeli_food_yolo11l_v2"   # NEW dir -> v1 production best.pt is never overwritten

# ---- core sizes (WHY) -------------------------------------------------------------------------
IMGSZ  = 224   # WHY: (1) matches how v1 + the arbiter feature generator + the live demo run the
               #      Israeli model, so its confidences stay on the SAME scale the arbiter expects;
               #      (2) YOLO11l-cls is pretrained at 224; (3) 14 classes are coarse enough that
               #      224 captures them while keeping the model fast for the edge demo. Changing it
               #      would silently shift the arbiter's input distribution -> kept fixed.
BATCH  = 32    # WHY: largest power-of-2 that fits an 8 GB RTX 3060 Ti at 224 with AdamW + AMP,
               #      while still giving stable BatchNorm statistics (>=16 is the usual floor).
               #      Bigger batch = smoother gradients here; 64 risks OOM at this imgsz/model.
EPOCHS = 120   # WHY: an upper bound only. Real stopping is governed by `patience` (early stop on
               #      val top-1). best.pt = the best epoch, so over-budgeting epochs is safe.

# ---- blur augmentation (WHY) ------------------------------------------------------------------
# The live demo runs on a webcam; motion/defocus blur is common. Training through random blur makes
# the model robust to it. Applied on the GPU batch each step so it costs almost nothing.
blur_transform = A.Compose([
    A.OneOf([
        A.MotionBlur(blur_limit=(3, 9), p=1.0),
        A.Defocus(radius=(1, 5), alias_blur=0.1, p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    ], p=0.35),   # 35% of batches get one blur variant; 65% stay sharp so the model still sees clean food
])

def apply_blur_augmentation(trainer):
    if hasattr(trainer, "batch") and trainer.batch is not None:
        imgs    = trainer.batch["img"]
        imgs_np = (imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
        augmented = [blur_transform(image=img)["image"] for img in imgs_np]
        trainer.batch["img"] = torch.from_numpy(
            np.stack(augmented).transpose(0, 3, 1, 2).astype("float32") / 255.0
        ).to(imgs.device)

if __name__ == "__main__":
    assert os.path.isdir(DATASET_DIR), f"Dataset directory not found: {DATASET_DIR}"
    assert torch.cuda.is_available(), "CUDA not detected."

    model = YOLO("yolo11l-cls.pt")            # WHY: full retrain from the ImageNet-pretrained backbone,
                                              # NOT a fine-tune. A fresh head over a pretrained backbone
                                              # avoids the catastrophic-forgetting / head-reset failure
                                              # that wrecked the freeze-based fine-tune experiments.
    model.add_callback("on_train_batch_start", apply_blur_augmentation)

    results = model.train(
        data       = DATASET_DIR,
        task       = "classify",
        epochs     = EPOCHS,
        imgsz      = IMGSZ,
        batch      = BATCH,
        workers    = 4,            # WHY: data-loading threads; 4 keeps the 3060 Ti fed without thrashing
        device     = 0,
        amp        = True,         # WHY: mixed precision -> ~2x speed + less VRAM, no accuracy cost at 224
        cache      = False,        # WHY: ~6k images; disk cache off keeps RAM free and avoids stale cache
                                   #      after we changed the data (.cache files are rebuilt from disk)
        optimizer       = "AdamW", # WHY: robust default for fine-grained transfer; decoupled weight decay
        lr0             = 0.001,   # WHY: standard AdamW transfer LR; high enough to adapt the head, low
                                   #      enough not to wreck pretrained features
        lrf             = 0.01,    # WHY: final LR = lr0*0.01 via the cosine schedule -> fine convergence
        momentum        = 0.937,   # WHY: Ultralytics default (AdamW beta1); well tested
        weight_decay    = 0.0005,  # WHY: mild L2 regularization; standard for this backbone
        warmup_epochs   = 5,       # WHY: ramp LR for 5 epochs so the fresh classifier head does not
                                   #      destabilize the pretrained backbone early
        warmup_momentum = 0.8,
        dropout         = 0.2,     # WHY: the Israeli set is small/imbalanced -> dropout in the classifier
                                   #      head fights overfitting
        label_smoothing = 0.1,     # WHY: KEY for this project. It stops the model assigning 100% to one
                                   #      class -> better-CALIBRATED confidences. Calibrated confidence is
                                   #      exactly what the arbiter consumes, and what the new "background"
                                   #      class needs to express honest uncertainty.
        patience    = 25,          # WHY: stop if val top-1 has not improved for 25 epochs; best.pt is kept
        save        = True,
        save_period = 10,
        # ---- geometric / photometric augmentation (WHY: invariance to how food is shot) ----
        hsv_h = 0.02,   hsv_s = 0.75,   hsv_v = 0.50,   # lighting/colour shifts (kitchen lighting varies)
        degrees = 20.0, translate = 0.15, scale = 0.60, # rotation/shift/zoom: food has no canonical pose
        shear = 8.0,    perspective = 0.0008,           # mild perspective: angled camera in the demo
        flipud = 0.15,  fliplr = 0.50,                  # food is largely orientation-agnostic
        erasing = 0.40, mixup = 0.10,   mosaic = 0.0,   # erasing+mixup regularize; mosaic OFF (it is a
                                                        # detection trick, harmful for whole-image classify)
        project    = PROJECT_DIR,
        name       = RUN_NAME,
        exist_ok   = False,        # WHY: refuse to overwrite an existing run -> protects prior results
        pretrained = True,
        verbose    = True,
    )
