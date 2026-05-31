# Food Recognition Project — Model Training Guide

## Project Context

Dual-model hierarchical food recognition system for precise nutritional estimation.
Final product is used by **students taking smartphone pictures of food inside classrooms**.

**Priority:** Top-1 Accuracy over inference speed (must still be usable in real-time).
**Hardware:** NVIDIA RTX 3060 Ti — 8 GB VRAM.

---

## Critical Constraint — Domain Shift

Training images are clean web-sourced food photos. Inference images are casual smartphone
photos taken inside classrooms. Real-world conditions include:

- Harsh fluorescent / overhead lighting
- Deep shadows and overexposure
- Low resolution and motion blur from hand-shake
- Slow autofocus / defocus blur
- Weird angles (tilted, sideways, top-down)
- Partial occlusion (hands, utensils, wrappers, adjacent trays)
- Food not centred in frame

All augmentations must target these conditions specifically.

---

## Dataset Structure

```
E:\final project - models retrain\datasets\<DATASET_NAME>\
    train\
        <class_1>\   ← raw .jpg / .png images
        <class_2>\
        ...
    val\
        <class_1>\
        ...
    test\
        <class_1>\
        ...
```

Dataset is **perfectly balanced and pre-split** — do not re-split.

---

## Lessons Learned (Bugs Fixed)

### 1 — Do NOT pass data.yaml to a classification task
For `task=classify`, Ultralytics' `check_cls_dataset()` expects a **directory path**
(the folder containing train/val/test), not a `.yaml` file.
Passing a `.yaml` causes it to construct a garbage download URL and crash.

```python
# WRONG
data = r"E:\...\data.yaml"

# CORRECT
data = r"E:\...\datasets\general_model_data_set"
```

### 2 — Always use `if __name__ == "__main__":` on Windows
Windows uses `spawn` (not `fork`) for multiprocessing. DataLoader workers re-import
the script as a fresh process. Any code outside this guard (model construction,
`model.train()`) will be re-executed by every worker, causing:

```
RuntimeError: An attempt has been made to start a new process before the
current process has finished its bootstrapping phase.
```

**Rule:** Put ALL model construction and training inside `if __name__ == "__main__":`.
Define transforms and callback functions at module level (outside the guard) so they
are picklable by worker processes.

### 3 — data.yaml is NOT needed for classification training
The `generate_yaml.py` output is useful as a reference document but is not passed
to `model.train()`. Ultralytics reads class names directly from the subfolder names.

---

## Architecture Selection — Decision Framework

| Scenario | Recommended Model | Reason |
|---|---|---|
| Max accuracy, 8 GB VRAM | **YOLO11l-cls** | Best Top-1 in budget |
| Accuracy + faster training | YOLO11m-cls | ~1% less accurate, 2× faster |
| Speed-critical deployment | YOLO11s-cls | Lightweight, lower accuracy |
| >10 GB VRAM available | YOLO11x-cls | Best possible accuracy |

**Why YOLO11l over YOLOv8l:**
- Same VRAM footprint (~6.2 GB at batch=32, imgsz=224, AMP on)
- +0.9% Top-1 on ImageNet benchmarks
- C3k2 blocks improve gradient flow on fine-grained texture features
- Partial Self-Attention (PSA) helps distinguish visually similar classes
- Zero-cost upgrade — identical Ultralytics API

**Why NOT YOLOv8x / YOLO11x on this GPU:**
- ~56M params → ~9.4 GB VRAM at batch=32 → OOM on 8 GB card
- Dropping to batch=16 to compensate adds ~2× training time and noisier gradients
- Accuracy gain over YOLO11l is only ~0.4% — not worth it

---

## VRAM Budget (RTX 3060 Ti, 8 GB)

| Component | Estimated VRAM |
|---|---|
| YOLO11l-cls weights | ~2.1 GB |
| Activations FP16 (AMP) | ~2.7 GB |
| Gradients + optimizer state | ~1.1 GB |
| CUDA overhead | ~0.3 GB |
| **Total** | **~6.2 GB** |
| Headroom | ~1.8 GB |

If OOM occurs: reduce `batch` from 32 → 24 → 16. Do NOT increase `imgsz` first —
memory scales quadratically with image size.

---

## Safe Parameter Defaults (8 GB VRAM)

```
imgsz   = 224     # Matches ImageNet pretraining; 320 triples memory for <0.5% gain
batch   = 32      # Max safe batch for 8 GB + AMP
epochs  = 120     # With patience=25 early stopping
workers = 4       # Windows-safe; 8+ can deadlock on Windows DataLoader
amp     = True    # Non-negotiable on 8 GB — cuts activation memory ~40%
cache   = False   # Dataset too large to cache in 8 GB
```

---

## Augmentation Strategy (Domain Shift)

| Augmentation | Value | Simulates |
|---|---|---|
| `hsv_h` | 0.02 | Colour temperature shift (fluorescent vs warm light) |
| `hsv_s` | 0.75 | Washed-out de-saturation / HDR oversaturation |
| `hsv_v` | 0.50 | Overexposure, deep shadows, flickering lights |
| `degrees` | 20.0 | Phone held tilted or sideways |
| `translate` | 0.15 | Food not centred in frame |
| `scale` | 0.60 | Close-up single dish vs wide full-tray shot |
| `shear` | 8.0 | Off-axis shooting angle |
| `perspective` | 0.0008 | Smartphone lens barrel distortion |
| `flipud` | 0.15 | Phone held upside-down, auto-rotate errors |
| `fliplr` | 0.50 | Standard horizontal invariance |
| `erasing` | 0.40 | Hands, utensils, wrappers occluding food |
| `mixup` | 0.10 | Hardens boundaries between similar classes |
| `mosaic` | 0.0 | **DISABLED** — detection-era augment, hurts classification |
| `MotionBlur` (albumentations) | p=0.35 | Hand-shake during capture |
| `Defocus` (albumentations) | p=0.35 | Slow autofocus hunting |
| `GaussianBlur` (albumentations) | p=0.35 | Low-res upscaling artefacts |

Blur augmentations are injected via an `on_train_batch_start` callback using
albumentations because Ultralytics does not expose these natively.

---

## Optimizer Settings

```
optimizer       = "AdamW"   # Better than SGD for fine-tuning pretrained models
lr0             = 0.001     # Conservative — prevents destroying pretrained features
lrf             = 0.01      # Cosine decay → final LR = 0.001 × 0.01 = 0.00001
momentum        = 0.937     # AdamW beta1
weight_decay    = 0.0005    # L2 regularisation
warmup_epochs   = 5         # Ramp LR from ~0 to lr0 — prevents catastrophic forgetting
warmup_momentum = 0.8
dropout         = 0.2       # In classification head
label_smoothing = 0.1       # Prevents overconfidence on ambiguous images
patience        = 25        # Early stopping
save_period     = 10        # Checkpoint every 10 epochs as crash insurance
```

---

## Training Script Template

Copy this for each new model. Change `DATASET_DIR`, `RUN_NAME`, and `nc` (class count).

```python
import os
import numpy as np
import torch
import albumentations as A
from ultralytics import YOLO

DATASET_DIR = r"E:\final project - models retrain\datasets\<DATASET_NAME>"
PROJECT_DIR = r"E:\final project - models retrain\runs"
RUN_NAME    = "<model_run_name>"

IMGSZ  = 224
BATCH  = 32
EPOCHS = 120

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

if __name__ == "__main__":
    assert os.path.isdir(DATASET_DIR), f"Dataset directory not found: {DATASET_DIR}"
    assert torch.cuda.is_available(), "CUDA not detected."

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
        momentum       = 0.937,
        weight_decay   = 0.0005,
        warmup_epochs  = 5,
        warmup_momentum= 0.8,
        dropout         = 0.2,
        label_smoothing = 0.1,
        patience    = 25,
        save        = True,
        save_period = 10,
        hsv_h = 0.02,   hsv_s = 0.75,   hsv_v = 0.50,
        degrees = 20.0, translate = 0.15, scale = 0.60,
        shear = 8.0,    perspective = 0.0008,
        flipud = 0.15,  fliplr = 0.50,
        erasing = 0.40, mixup = 0.10,   mosaic = 0.0,
        project    = PROJECT_DIR,
        name       = RUN_NAME,
        exist_ok   = False,
        pretrained = True,
        verbose    = True,
    )
```

---

## Class Name Generator (optional reference utility)

Not required for training. Useful for auditing the dataset or building a UI label map.

```python
import os, yaml

DATASET_ROOT = r"E:\final project - models retrain\datasets\<DATASET_NAME>"
train_dir    = os.path.join(DATASET_ROOT, "train")
classes      = sorted([d for d in os.listdir(train_dir)
                       if os.path.isdir(os.path.join(train_dir, d))])

print(f"{len(classes)} classes found:")
for i, name in enumerate(classes):
    print(f"  [{i:>3}] {name}")
```

---

## Output Files

| File | Description |
|---|---|
| `runs/<RUN_NAME>/weights/best.pt` | **Use this for inference** — best val Top-1 epoch |
| `runs/<RUN_NAME>/weights/last.pt` | Final epoch checkpoint |
| `runs/<RUN_NAME>/results.csv` | Per-epoch loss, Top-1, Top-5, LR |
| `runs/<RUN_NAME>/confusion_matrix.png` | Val set confusion matrix |
| `runs/<RUN_NAME>/weights/epoch_N.pt` | Periodic checkpoints (every 10 epochs) |

### Resuming a crashed run
```python
model = YOLO(r"runs\<RUN_NAME>\weights\last.pt")
results = model.train(resume=True)
```

---

## Planned Models

| Model | Dataset | Classes | Status |
|---|---|---|---|
| General World Cuisine | `general_model_data_set` | 142 | ✅ Training |
| _(next model)_ | _(dataset name)_ | — | ⬜ Pending |
