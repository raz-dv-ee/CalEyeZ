"""
evaluate_system.py
Full end-to-end evaluation of the Caleyez pipeline on the TEST splits.

Pipeline per image:
  1. Global YOLO11l-cls  → Top-5 candidates  (142-class vector)
  2. Israeli YOLO11l-cls → Top-5 candidates  (13-class vector)
  3. Merge → unique candidate set (5-10 classes)
  4. Extract (global_prob, israeli_prob, is_global_top1, is_israeli_top1)
  5. XGBoost Re-Ranker  → scores all candidates
  6. argmax(score)      → Arbiter final prediction

Output:
  system_evaluation_results.csv   — one row per image
  Console summary                 — overall + per-domain accuracy
"""

import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────────────────────
BASE               = r"E:\final project - models retrain"
GLOBAL_MODEL_PATH  = rf"{BASE}\runs\general_model_yolo11l5\weights\best.pt"
ISRAELI_MODEL_PATH = rf"{BASE}\runs\israeli_food_yolo11l\weights\best.pt"
XGB_MODEL_PATH     = rf"{BASE}\arbiter\train\reranker_xgb.ubj"
OUTPUT_CSV         = rf"{BASE}\arbiter\train\system_evaluation_results.csv"

TEST_DIRS = {
    "global":  rf"{BASE}\datasets\general_model_data_set\test",
    "israeli": rf"{BASE}\datasets\israeli-food-master\test",
}

TOP_K          = 5
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# MUST match FEATURE_COLS order in train_reranker_xgb.py (v5 — 15 features:
# 9 base + 6 derived interaction features appended in this exact order).
FEATURE_COLS = [
    # base 9
    "global_prob",
    "israeli_prob",
    "is_global_top1",
    "is_israeli_top1",
    "global_entropy",
    "local_entropy",
    "global_top1_vs_top2",
    "local_top1_vs_top2",
    "arbiter_dominance_score",
    # v5 derived 6
    "conf_global",
    "conf_israeli",
    "israeli_top1_p",
    "global_top1_p",
    "prob_ratio",
    "both_agree",
]

OUTPUT_FIELDNAMES = [
    "image_path",
    "domain",                # "global" or "israeli" (which dataset it came from)
    "ground_truth",
    "global_top1_class",
    "global_top1_conf",
    "israeli_top1_class",
    "israeli_top1_conf",
    "arbiter_final_class",
    "arbiter_final_score",
    "is_correct",
]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_yolo(path: str):
    m = YOLO(path)
    m.to(DEVICE)
    return m


def load_xgb(path: str) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(path)
    return booster


# ── Dataset helpers ───────────────────────────────────────────────────────────

def collect_images(root: Path, domain: str) -> list[tuple[Path, str, str]]:
    """Return [(img_path, ground_truth_class, domain), ...]."""
    samples = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        for img in class_dir.iterdir():
            if img.suffix.lower() in IMG_EXTENSIONS:
                samples.append((img, class_dir.name, domain))
    return samples


# ── Inference helpers ─────────────────────────────────────────────────────────

def run_yolo(model, img_path: Path):
    """Returns (probs_tensor [num_classes], names {idx: str})."""
    res   = model.predict(str(img_path), verbose=False, device=DEVICE)
    probs = res[0].probs.data
    return probs, model.names


def get_class_to_idx(model) -> dict[str, int]:
    return {name: idx for idx, name in model.names.items()}


def top_k_classes(probs: torch.Tensor, names: dict, k: int) -> list[str]:
    indices = torch.topk(probs, k=min(k, len(probs))).indices.tolist()
    return [names[i] for i in indices]


def prob_for_class(name: str, probs: torch.Tensor, c2i: dict) -> float:
    """Query a probability vector by class name. Returns 0.0 if OOV."""
    idx = c2i.get(name)
    return float(probs[idx].item()) if idx is not None else 0.0


def top_k_entropy(probs: torch.Tensor, k: int) -> float:
    """Normalized Top-K Shannon entropy in [0,1] (÷log k). Must match dataset script."""
    k_eff = min(k, len(probs))
    if k_eff <= 1:
        return 0.0
    topk = torch.topk(probs, k=k_eff).values
    p    = topk / topk.sum().clamp_min(1e-12)
    p    = p.clamp_min(1e-12)
    h    = float(-(p * p.log()).sum().item())
    return h / math.log(k_eff)


def top1_vs_top2_margin(probs: torch.Tensor) -> float:
    """(top1_prob - top2_prob). Must match dataset script."""
    if len(probs) < 2:
        return float(probs.max().item()) if len(probs) else 0.0
    top2 = torch.topk(probs, k=2).values
    return float((top2[0] - top2[1]).item())


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    img_path: Path,
    g_model,  i_model,
    g_c2i: dict, i_c2i: dict,
    xgb_model: xgb.Booster,
) -> dict:
    """
    Run the full Caleyez pipeline on one image.
    Returns a dict with all columns needed for the output CSV.
    """
    g_probs, g_names = run_yolo(g_model, img_path)
    i_probs, i_names = run_yolo(i_model, img_path)

    g_top_k = top_k_classes(g_probs, g_names, TOP_K)
    i_top_k = top_k_classes(i_probs, i_names, TOP_K)

    # Top-1 identities & confidences
    g_top1_cls  = g_top_k[0] if g_top_k else ""
    g_top1_conf = prob_for_class(g_top1_cls, g_probs, g_c2i)
    i_top1_cls  = i_top_k[0] if i_top_k else ""
    i_top1_conf = prob_for_class(i_top1_cls, i_probs, i_c2i)

    # Unified candidate set (Global top-k first, then additional Israeli)
    seen: set[str] = set()
    candidates: list[str] = []
    for cls in g_top_k + i_top_k:
        if cls not in seen:
            seen.add(cls)
            candidates.append(cls)

    # ── Per-image confusion features (constant across candidates) ──
    g_entropy = top_k_entropy(g_probs, TOP_K)
    i_entropy = top_k_entropy(i_probs, TOP_K)
    g_margin  = top1_vs_top2_margin(g_probs)
    i_margin  = top1_vs_top2_margin(i_probs)

    # Build feature matrix for XGBoost — order MUST match FEATURE_COLS (v5, 15 feats)
    rows = []
    for cls in candidates:
        g_prob   = prob_for_class(cls, g_probs, g_c2i)
        i_prob   = prob_for_class(cls, i_probs, i_c2i)
        is_g_t1  = int(cls == g_top1_cls)
        is_i_t1  = int(cls == i_top1_cls)
        rows.append([
            # ── base 9 ──
            g_prob,                          # global_prob
            i_prob,                          # israeli_prob
            is_g_t1,                         # is_global_top1
            is_i_t1,                         # is_israeli_top1
            g_entropy,                       # global_entropy
            i_entropy,                       # local_entropy
            g_margin,                        # global_top1_vs_top2
            i_margin,                        # local_top1_vs_top2
            g_prob - i_prob,                 # arbiter_dominance_score
            # ── v5 derived 6 (must match add_derived_features in training) ──
            g_prob * (1.0 - g_entropy),      # conf_global
            i_prob * (1.0 - i_entropy),      # conf_israeli
            i_prob * is_i_t1,                # israeli_top1_p
            g_prob * is_g_t1,                # global_top1_p
            i_prob / (g_prob + 0.05),        # prob_ratio
            is_g_t1 * is_i_t1,               # both_agree
        ])

    feature_matrix = np.array(rows, dtype=np.float32)
    dmat           = xgb.DMatrix(feature_matrix, feature_names=FEATURE_COLS)
    scores         = xgb_model.predict(dmat)        # shape [num_candidates]

    best_idx            = int(np.argmax(scores))
    arbiter_final_class = candidates[best_idx]
    arbiter_final_score = float(scores[best_idx])

    return {
        "global_top1_class":  g_top1_cls,
        "global_top1_conf":   round(g_top1_conf, 6),
        "israeli_top1_class": i_top1_cls,
        "israeli_top1_conf":  round(i_top1_conf, 6),
        "arbiter_final_class": arbiter_final_class,
        "arbiter_final_score": round(arbiter_final_score, 6),
    }


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate():
    print(f"Device : {DEVICE}")
    print("Loading YOLO models…")
    g_model = load_yolo(GLOBAL_MODEL_PATH)
    i_model = load_yolo(ISRAELI_MODEL_PATH)
    g_c2i   = get_class_to_idx(g_model)
    i_c2i   = get_class_to_idx(i_model)
    print(f"  Global  classes : {len(g_model.names)}")
    print(f"  Israeli classes : {len(i_model.names)}")

    print("Loading XGBoost Re-Ranker…")
    xgb_model = load_xgb(XGB_MODEL_PATH)
    print("  OK\n")

    # Collect all test images from both domains
    all_samples: list[tuple[Path, str, str]] = []
    for domain, dir_path in TEST_DIRS.items():
        root = Path(dir_path)
        if not root.exists():
            print(f"  WARN: test dir not found — {root}")
            continue
        found = collect_images(root, domain)
        print(f"[{domain}] {len(found):,} test images in {root}")
        all_samples.extend(found)
    print(f"\nTotal test images : {len(all_samples):,}\n")

    # Per-domain counters for the summary
    counters = {
        "global":  {"total": 0, "correct": 0},
        "israeli": {"total": 0, "correct": 0},
    }
    total_correct = 0

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()

        for idx, (img_path, gt_class, domain) in enumerate(all_samples, 1):
            try:
                result = run_pipeline(img_path, g_model, i_model,
                                      g_c2i, i_c2i, xgb_model)
            except Exception as exc:
                print(f"  WARN: skipping {img_path.name} — {exc}")
                continue

            is_correct = int(result["arbiter_final_class"] == gt_class)
            total_correct += is_correct
            counters[domain]["total"]   += 1
            counters[domain]["correct"] += is_correct

            writer.writerow({
                "image_path":          str(img_path),
                "domain":              domain,
                "ground_truth":        gt_class,
                **result,
                "is_correct":          is_correct,
            })

            if idx % 200 == 0:
                running_acc = total_correct / idx * 100
                print(f"  [{idx:,}/{len(all_samples):,}]  "
                      f"running accuracy: {running_acc:.2f}%")

    # ── Console summary ───────────────────────────────────────────────────────
    total = sum(c["total"] for c in counters.values())
    print("\n" + "═" * 58)
    print("  CALEYEZ SYSTEM — EVALUATION SUMMARY")
    print("═" * 58)
    print(f"  Total images tested   : {total:,}")
    print(f"  Overall accuracy      : {total_correct/total*100:.2f}%  "
          f"({total_correct}/{total})")
    print("─" * 58)
    for domain, c in counters.items():
        if c["total"] == 0:
            continue
        acc = c["correct"] / c["total"] * 100
        label = f"{domain.capitalize()} domain"
        print(f"  {label:<22s}: {acc:.2f}%  "
              f"({c['correct']}/{c['total']})")
    print("─" * 58)
    print(f"  Results saved to      : {OUTPUT_CSV}")
    print("═" * 58)


if __name__ == "__main__":
    evaluate()
