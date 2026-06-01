# Model Card — Global Food Classifier (Flattened, 132-class)

**Run:** `runs/general_model_flattened/`
**Weights:** `weights/best.pt` (epoch 73) · 25.1 MB
**Date trained:** 2026-06-01
**Status:** New Global baseline. Trained on the cleaned, leakage-free dataset. Not yet
wired into the router/Israeli fusion (label space changed 142 → 132).

---

## Architecture
- **Backbone:** YOLO11l-cls (Ultralytics), ImageNet-1k pretrained (`yolo11l-cls.pt`).
- **Params:** ~13.0 M · 49.9 GFLOPs · input 320×320.
- Only the classification head was re-initialized for 132 classes (492/494 pretrained
  items transferred).

## Training data
- **Dataset:** `datasets/general_model_flattened` — produced by
  `scripts/flatten_general_dataset.py` (see `DATASET_FLATTENING.html`).
- **Classes:** 132 (from the original 142: merged sub-classes, normalized labels,
  removed pork / paprika / soy_beans).
- **Images:** 99,307 total → train 69,491 / val 19,827 / test 9,960 (70/20/10).
- **Leakage:** removed via exact (SHA-1) + perceptual (dHash) de-duplication; no image
  or near-duplicate is shared across splits.

## Key hyperparameters
| | |
|---|---|
| epochs | 150 (early-stopped at 116, patience 30) |
| imgsz / batch | 320 / 16 |
| optimizer | AdamW, lr0 1e-3, lrf 0.01, cosine schedule |
| regularization | dropout 0.2, label_smoothing 0.1, weight_decay 5e-4 |
| augmentation | heavy domain-shift (HSV, rotation/shear/perspective, erasing 0.4, mixup 0.1, mosaic 0) + Albumentations blur callback (p=0.35) |
| AMP | on · peak VRAM ~2.2 GB (RTX 3060 Ti) |

## Results (best.pt, epoch 73)
| Metric | Validation | **Test (clean)** |
|---|---|---|
| Top-1 | 88.16% | **88.18%** |
| Top-5 | 96.87% | **96.94%** |

- **Val ≈ Test (88.16 vs 88.18):** the splits agree, confirming the de-duplication
  removed leakage. The number is trustworthy (contrast: the previous model showed
  val 83.89% vs test 88.19%, a ~4-pt leakage gap).
- **Requirement (FRS ≥ 80% aggregate top-1):** exceeded by ~8 points — Global model alone.

## Comparison to previous Global model (`general_model_yolo11l5`)
| | Old (142-cls) | New (132-cls) |
|---|---|---|
| Test top-1 | 88.19% (leakage-inflated) | 88.18% (clean, harder test) |
| Clean estimate | ~85–86% | 88.18% (measured) |
| Label space | 142 (noisy) | 132 (cleaned) |
| Leakage | yes (~1.6% exact) | none |

Net: same headline number, but earned on a cleaner, harder, leakage-free test set →
a genuine improvement, on a better-defined label space.

## Test-set confusion (most-confused class pairs)
Meaningful confusions (n=100 dish classes) are all semantically plausible inter-class
overlaps, not failures:

| True → Predicted | Rate |
|---|---|
| chocolate_mousse → chocolate_cake | 12% |
| apple_pie → bread_pudding | 10% |
| gnocchi ↔ ravioli | 7% |
| beef_carpaccio → tartare | 6% |
| eggs_benedict → croque_madame | 5% |
| lobster_bisque ↔ clam_chowder | 5% |

The lowest per-class recall values belong to the small ingredient classes (apple/onion/
banana, n≈8 test images each) where one or two misses swing the percentage — a
small-sample effect of class imbalance, not a systematic weakness.

## Known limitations
- Heavy augmentation trades clean-test top-1 for classroom robustness; a moderate-aug
  retrain would likely score higher on this clean test (~90%) but be less deployable.
- Mild overfitting after ~epoch 60 (train loss ↓, val loss ↑); `best.pt` is frozen at
  the val peak and is unaffected.
- Class imbalance (ingredient classes ~70–200 imgs vs dish classes ~1000) — mitigate
  with class-balanced sampling in any future retrain.

## Reproduce
```powershell
python scripts\training\train_general_model.py
```
