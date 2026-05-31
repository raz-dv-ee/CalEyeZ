"""
generate_reranker_dataset.py
Builds a Pointwise Learning-to-Rank dataset for the XGBoost Re-Ranker.

For each image, the Top-5 candidates from the Global model and Top-5 from
the Israeli model are merged into a unique candidate set. Every candidate
gets a row with probabilities extracted from BOTH model output vectors.

v2 — Evidence-Based Routing Features
────────────────────────────────────
Beyond raw probabilities, we now engineer "confusion features" so the Arbiter
can reason about HOW CERTAIN each model is, not just WHAT it predicted:

  Per-image features (same value across all candidate rows of one image):
    global_entropy       — normalized Top-5 entropy of the Global model [0,1].
                           Low = confident/peaked, High = confused/guessing.
    local_entropy        — normalized Top-5 entropy of the Israeli model [0,1].
    global_top1_vs_top2  — Global model's (top1_prob - top2_prob) margin.
                           A large margin = decisive winner.
    local_top1_vs_top2   — Israeli model's (top1_prob - top2_prob) margin.

  Per-candidate feature:
    arbiter_dominance_score — (global_prob - israeli_prob) for THIS candidate.
                              Positive = Global favours it, negative = Israeli.

Both entropies are normalized by log(k) so the 142-class Global model and the
13-class Israeli model land on the SAME [0,1] scale and are directly comparable.
"""

import argparse
import csv
import math
from pathlib import Path

import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration — override via CLI args or edit defaults here
# ---------------------------------------------------------------------------
BASE               = r"E:\final project - models retrain"
GLOBAL_MODEL_PATH  = rf"{BASE}\runs\general_model_yolo11l5\weights\best.pt"
ISRAELI_MODEL_PATH = rf"{BASE}\runs\israeli_food_yolo11l\weights\best.pt"

# Each split pulls images from BOTH dataset trees so the re-ranker sees
# Israeli-food images and general-food images in the same training set.
SPLIT_DIRS: dict[str, list[str]] = {
    "val": [
        rf"{BASE}\datasets\general_model_data_set\val",
        rf"{BASE}\datasets\israeli-food-master\val",
    ],
    "test": [
        rf"{BASE}\datasets\general_model_data_set\test",
        rf"{BASE}\datasets\israeli-food-master\test",
    ],
}

OUTPUT_CSV     = rf"{BASE}\arbiter\data\reranker_dataset.csv"
TOP_K          = 5
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FIELDNAMES = [
    "image_path",
    "split_type",
    "ground_truth_class",
    "candidate_class",
    # ── raw probability features ──
    "global_prob",
    "israeli_prob",
    "is_global_top1",
    "is_israeli_top1",
    # ── v2 evidence-based / confusion features ──
    "global_entropy",          # normalized Top-5 entropy of Global model  [0,1]
    "local_entropy",           # normalized Top-5 entropy of Israeli model [0,1]
    "global_top1_vs_top2",     # Global  (top1_prob - top2_prob) margin
    "local_top1_vs_top2",      # Israeli (top1_prob - top2_prob) margin
    "arbiter_dominance_score", # (global_prob - israeli_prob) for this candidate
    # ── label ──
    "target_label",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(path: str):
    model = YOLO(path)
    model.to(DEVICE)
    return model


def get_class_to_idx(model) -> dict[str, int]:
    """Return {class_name: index} from the model's .names dict."""
    return {name: idx for idx, name in model.names.items()}


def collect_images(root: Path) -> list[tuple[Path, str]]:
    """
    Walk a directory whose structure is root/<class_name>/<image_files>.
    Returns [(image_path, ground_truth_class), ...].
    """
    samples = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        gt_class = class_dir.name
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in IMG_EXTENSIONS:
                samples.append((img_file, gt_class))
    return samples


def run_inference(model, img_path: Path):
    """
    Run the model on one image.
    Returns (probs_tensor, names_dict) where probs_tensor is shape [num_classes].
    """
    results = model.predict(str(img_path), verbose=False, device=DEVICE)
    result  = results[0]
    probs   = result.probs.data          # torch.Tensor [num_classes]
    return probs, model.names            # names: {idx: class_str}


def top_k_classes(probs: torch.Tensor, names: dict, k: int) -> list[str]:
    """Return top-k class name strings by descending probability."""
    topk_indices = torch.topk(probs, k=min(k, len(probs))).indices.tolist()
    return [names[i] for i in topk_indices]


def top_k_entropy(probs: torch.Tensor, k: int) -> float:
    """
    Normalized Shannon entropy of the Top-K probabilities, in [0, 1].

    The Top-K probabilities are renormalized to sum to 1, then:
        H = -Σ p·log(p)            (natural log)
        H_norm = H / log(K)        (so the scale is [0,1] regardless of K)

    WHY NORMALIZE BY log(K):
      The Global model (142 classes) and Israeli model (13 classes) have
      vastly different maximum possible entropies. Dividing by log(K) puts
      BOTH models on the same [0,1] "confusion" scale so the Arbiter can
      compare them directly. 0 = fully confident (all mass on one class),
      1 = maximally confused (uniform over the Top-K).
    """
    k_eff = min(k, len(probs))
    if k_eff <= 1:
        return 0.0
    topk = torch.topk(probs, k=k_eff).values
    p    = topk / topk.sum().clamp_min(1e-12)
    p    = p.clamp_min(1e-12)
    h    = float(-(p * p.log()).sum().item())
    return h / math.log(k_eff)


def top1_vs_top2_margin(probs: torch.Tensor) -> float:
    """
    (top1_prob - top2_prob): how decisively the model prefers its #1 choice.
    A large margin means a clear, unambiguous winner; ~0 means a coin-flip
    between the two best candidates.
    """
    if len(probs) < 2:
        return float(probs.max().item()) if len(probs) else 0.0
    top2 = torch.topk(probs, k=2).values
    return float((top2[0] - top2[1]).item())


def prob_for_class(
    class_name: str,
    probs: torch.Tensor,
    class_to_idx: dict[str, int],
) -> float:
    """
    Look up the probability of class_name in a model's output vector.
    Returns 0.0 if the class is outside this model's vocabulary.

    VECTOR ALIGNMENT LOGIC
    ----------------------
    The Global model has a 142-dim vector; the Israeli model has 13-dim.
    A candidate class may exist in one vocabulary but not the other.
    Rather than crashing or skipping, we define the probability as 0.0
    for any out-of-vocabulary class — a principled "no evidence" prior
    that lets XGBoost learn the absence signal explicitly.
    """
    idx = class_to_idx.get(class_name)
    if idx is None:
        return 0.0
    return float(probs[idx].item())


# ---------------------------------------------------------------------------
# Core dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    global_model,
    israeli_model,
    split_dirs: dict[str, list[str]],
    output_csv: str,
    top_k: int = TOP_K,
) -> None:
    global_c2i  = get_class_to_idx(global_model)
    israeli_c2i = get_class_to_idx(israeli_model)

    rows_written = 0
    images_processed = 0
    skipped = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for split_name, dir_list in split_dirs.items():
            # Collect images from every source directory for this split
            samples: list[tuple[Path, str]] = []
            for d in dir_list:
                root = Path(d)
                if not root.exists():
                    print(f"  WARN: directory not found, skipping — {root}")
                    continue
                found = collect_images(root)
                print(f"[{split_name}] {len(found):,} images in {root}")
                samples.extend(found)
            print(f"[{split_name}] Total: {len(samples):,} images\n")

            for img_path, gt_class in samples:
                try:
                    g_probs, g_names = run_inference(global_model,  img_path)
                    i_probs, i_names = run_inference(israeli_model, img_path)
                except Exception as exc:
                    print(f"  WARN: skipping {img_path} — {exc}")
                    skipped += 1
                    continue

                g_top_k = top_k_classes(g_probs, g_names, top_k)
                i_top_k = top_k_classes(i_probs, i_names, top_k)

                # Unified candidate set — ordered: global top-k first, then
                # any additional Israeli candidates not already present.
                seen: set[str] = set()
                candidates: list[str] = []
                for cls in g_top_k + i_top_k:
                    if cls not in seen:
                        seen.add(cls)
                        candidates.append(cls)

                g_top1 = g_top_k[0] if g_top_k else None
                i_top1 = i_top_k[0] if i_top_k else None

                # ── Per-image confusion features (constant across candidates) ──
                g_entropy = top_k_entropy(g_probs, top_k)
                i_entropy = top_k_entropy(i_probs, top_k)
                g_margin  = top1_vs_top2_margin(g_probs)
                i_margin  = top1_vs_top2_margin(i_probs)

                for candidate in candidates:
                    g_prob = prob_for_class(candidate, g_probs, global_c2i)
                    i_prob = prob_for_class(candidate, i_probs, israeli_c2i)
                    writer.writerow({
                        "image_path":          str(img_path),
                        "split_type":          split_name,
                        "ground_truth_class":  gt_class,
                        "candidate_class":     candidate,
                        "global_prob":         g_prob,
                        "israeli_prob":        i_prob,
                        "is_global_top1":      int(candidate == g_top1),
                        "is_israeli_top1":     int(candidate == i_top1),
                        "global_entropy":          round(g_entropy, 6),
                        "local_entropy":           round(i_entropy, 6),
                        "global_top1_vs_top2":     round(g_margin, 6),
                        "local_top1_vs_top2":      round(i_margin, 6),
                        "arbiter_dominance_score": round(g_prob - i_prob, 6),
                        "target_label":        int(candidate == gt_class),
                    })
                    rows_written += 1

                images_processed += 1
                if images_processed % 100 == 0:
                    print(f"  Processed {images_processed} images, {rows_written} rows so far…")

    print(f"\nDone. {images_processed} images processed, {skipped} skipped.")
    print(f"Total rows written: {rows_written}  →  {output_csv}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build XGBoost Re-Ranker dataset.")
    p.add_argument("--global-model",  default=GLOBAL_MODEL_PATH)
    p.add_argument("--israeli-model", default=ISRAELI_MODEL_PATH)
    p.add_argument("--output",        default=OUTPUT_CSV)
    p.add_argument("--top-k",         type=int, default=TOP_K)
    p.add_argument("--no-test",       action="store_true",
                   help="Skip the test split (faster iteration)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Device: {DEVICE}")
    print("Loading models…")
    g_model = load_model(args.global_model)
    i_model = load_model(args.israeli_model)
    print(f"  Global  model classes : {len(g_model.names)}")
    print(f"  Israeli model classes : {len(i_model.names)}")

    active_splits = {"val": SPLIT_DIRS["val"]}
    if not args.no_test:
        active_splits["test"] = SPLIT_DIRS["test"]

    build_dataset(g_model, i_model, active_splits, args.output, top_k=args.top_k)
