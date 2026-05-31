"""
Auto-generates data.yaml for the General World Cuisine classification model.
Scans the train split to discover all 142 class names alphabetically,
matching the order Ultralytics assigns integer indices at runtime.
"""

import os
import yaml

DATASET_ROOT = r"E:\final project - models retrain\datasets\general_model_data_set"
OUTPUT_YAML  = r"E:\final project - models retrain\data.yaml"

train_dir = os.path.join(DATASET_ROOT, "train")

classes = sorted(
    [d for d in os.listdir(train_dir)
     if os.path.isdir(os.path.join(train_dir, d))]
)

if not classes:
    raise RuntimeError(f"No subdirectories found in: {train_dir}")

data = {
    "path": DATASET_ROOT.replace("\\", "/"),
    "train": "train",
    "val":   "val",
    "test":  "test",
    "nc":    len(classes),
    "names": classes,
}

with open(OUTPUT_YAML, "w", encoding="utf-8") as f:
    yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

print(f"[OK] data.yaml written to: {OUTPUT_YAML}")
print(f"     Discovered {len(classes)} classes:")
for i, name in enumerate(classes):
    print(f"     [{i:>3}] {name}")
