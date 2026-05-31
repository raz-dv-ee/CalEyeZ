import os
import numpy as np
import torch
import albumentations as A
from ultralytics import YOLO

DATASET_DIR = r"E:\final project - models retrain\datasets\israeli-food-master"
PROJECT_DIR = r"E:\final project - models retrain\runs"
RUN_NAME    = "israeli_food_yolo11l"

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
        optimizer       = "AdamW",
        lr0             = 0.001,
        lrf             = 0.01,
        momentum        = 0.937,
        weight_decay    = 0.0005,
        warmup_epochs   = 5,
        warmup_momentum = 0.8,
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
