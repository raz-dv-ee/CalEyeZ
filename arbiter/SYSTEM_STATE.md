# Caleyez — Architecture & State Tracker

> **Purpose:** Single source of truth for the system architecture, current
> evaluation results, and a prioritised TODO list for the XGBoost Arbiter.
> Update this file after every experiment.

---

## 1. System Architecture

### 1.1 Models

| Model | Framework | Output Shape | Vocabulary |
|-------|-----------|--------------|------------|
| **Global YOLO11l-cls** | Ultralytics YOLO11l | `[142]` softmax vector | 142 general food classes |
| **Israeli YOLO11l-cls** | Ultralytics YOLO11l | `[13]` softmax vector | 13 Israeli food classes |
| **XGBoost Re-Ranker** | XGBoost `binary:logistic` | scalar score ∈ (0, 1) | N/A — scores candidates |

### 1.2 Model Paths

```
runs/general_model_yolo11l5/weights/best.pt   ← Global model
runs/israeli_food_yolo11l/weights/best.pt      ← Israeli model
reranker_xgb.ubj                               ← XGBoost Re-Ranker (current: v3)
```

### 1.3 Dataset Paths

```
datasets/
├── general_model_data_set/
│   ├── train/   (used to train Global YOLO — NOT used for Re-Ranker)
│   ├── val/     (used to build Re-Ranker training dataset)
│   └── test/    (used for final system evaluation)
└── israeli-food-master/
    ├── train/   (used to train Israeli YOLO — NOT used for Re-Ranker)
    ├── val/     (used to build Re-Ranker training dataset)
    └── test/    (used for final system evaluation)
```

### 1.4 Israeli Model Classes (13)

```
baklava, bourekas_cheese, falafel, hummus, jachnun,
malawach, meorav_yerushalmi, sabich, samosa, schnitzel,
shakshuka, shawarma, sufganiyah
```

---

## 2. Inference Pipeline

For each input image:

1. **Global YOLO** → `P_G ∈ ℝ^142` → Top-5 class names
2. **Israeli YOLO** → `P_I ∈ ℝ^13` → Top-5 class names
3. **Merge** → Unique candidate set of 5–10 classes (Global first)
4. **Vector Alignment** — for every candidate `c`:
   - `global_prob  = P_G[name_to_idx_G[c]]`  or `0.0` if OOV
   - `israeli_prob = P_I[name_to_idx_I[c]]`  or `0.0` if OOV
   - `is_global_top1  = 1 if c == argmax(P_G) else 0`
   - `is_israeli_top1 = 1 if c == argmax(P_I) else 0`
5. **XGBoost** scores the N×4 feature matrix
6. **Output** → `candidates[argmax(xgb_scores)]`

### Vector Alignment Rule

> A candidate class that exists in one model's vocabulary but **not** the
> other receives a probability of exactly `0.0` for the absent model.
> This is a principled "no evidence" prior — not a missing value.

---

## 3. Re-Ranker Training — Version History

### v1 — Baseline (Global `scale_pos_weight` only)

| Parameter | Value |
|-----------|-------|
| Objective | `binary:logistic` |
| Imbalance handling | `scale_pos_weight = n_neg / n_pos ≈ 9.0` (global) |
| `max_depth` | 4 |
| `min_child_weight` | 5 |
| `max_delta_step` | — (not set) |
| Early stopping | 30 rounds |

**Result:** Israeli domain accuracy collapsed to **28.2%**.
**Root cause:** ~25K Global images vs ~1.3K Israeli images → Israeli rows were
~5% of training data. XGBoost learned "always trust high `global_prob`" — correct
95% of the time globally, catastrophic for Israeli domain.

---

### v2 — Three-Layer Instance Weighting *(over-corrected)*

| Parameter | Value |
|-----------|-------|
| Objective | `binary:logistic` |
| Imbalance handling | Instance weights: domain balance × per-domain pos balance × confidence multiplier |
| Israeli confidence multiplier | correct row ×4.0, wrong row ×2.0 (threshold ≥ 0.90) |
| `scale_pos_weight` | Removed — baked into instance weights |
| `max_depth` | 5 |
| `min_child_weight` | 3 |
| `max_delta_step` | — (not set) |
| Early stopping | 40 rounds |

**Weight diagnostics observed:**
```
Global rows          : 149,950   weight = 1.0×
Israeli rows         : 9,640     base domain weight = 15.55×
High-conf correct    : 766       × 4.0
High-conf wrong      : 22        × 2.0
Final Sum(w_global)  : 271,268
Final Sum(w_israeli) : 593,979   ← 2.18× OVER Global (bug)
Domain weight ratio  : 0.457     ← should be 1.0
```

**Result:**
- Israeli accuracy: 28.2% → **85.98%** ✓
- Global accuracy:  88.19% → **68.67%** ✗ (severe over-correction)
- Overall Re-Rank Accuracy@1: **69.53%**

**Root cause of failure:** Three layers multiplied sequentially with no
normalization guard. The per-domain positive balance and ×4.0 confidence
multiplier stacked on top of the domain balance factor, pushing the effective
Israeli weight to 2.18× Global. The Arbiter became "paranoid" — constantly
discarding confident Global predictions to guess an Israeli dish.

---

### v3 — Normalize-Last Weighting

| Parameter | Value |
|-----------|-------|
| Objective | `binary:logistic` |
| Imbalance handling | Domain balance → confidence boost → **hard renormalization** |
| Israeli confidence multiplier | correct row ×1.5 only (threshold ≥ 0.90) |
| `scale_pos_weight` | Restored globally: `n_neg / n_pos ≈ 9.0` |
| `max_depth` | 5 |
| `min_child_weight` | 3 |
| `max_delta_step` | 1 |
| Early stopping | 40 rounds |

**Core algorithm — the guarantee v2 was missing:**
```
Step 1: weights[is_israeli] = n_global_rows / n_israeli_rows   # base balance
Step 2: weights[high_conf_correct_israeli] *= 1.5              # redistribute within budget
Step 3: weights[is_israeli] *= sum(w_global) / sum(w_israeli)  # HARD RENORMALIZE
# → sum(w_israeli) == sum(w_global) exactly, always, regardless of step 2
```

**Result:** Achieved exact domain parity. Israeli accuracy lifted to **84.14%**,
but Global accuracy still dropped to **74.9%** (from 88%). The Arbiter remained
too aggressive: with only raw probabilities as features, the *only* way it could
help the Israeli minority was to globally lower its trust in the Global model —
a zero-sum trade that inevitably hurt the (majority) Global domain.

---

### v4 — Evidence-Based Routing *(current)*

**Insight:** Stop fighting the imbalance with weights alone. Give the model
better *evidence* (engineered confusion features) so it can make a **conditional**
trust decision instead of a blanket one — breaking the v3 zero-sum trade-off.

| Parameter | Value |
|-----------|-------|
| Objective | `binary:logistic` |
| Imbalance handling | **Pure 1:1 domain parity only** — confidence multipliers REMOVED |
| Israeli confidence multiplier | ❌ Removed (signal now lives in entropy features) |
| `scale_pos_weight` | `n_neg / n_pos ≈ 9.0` (global) |
| `max_depth` | 5 |
| `min_child_weight` | 3 |
| `max_delta_step` | 1 |
| Early stopping | 40 rounds |

**New engineered features (9 total, was 4):**

| Feature | Scope | Meaning |
|---------|-------|---------|
| `global_prob`, `israeli_prob` | per-candidate | Raw aligned probabilities |
| `is_global_top1`, `is_israeli_top1` | per-candidate | Top-1 indicator flags |
| `global_entropy` | per-image | Normalized Top-5 entropy of Global, [0,1]. Low = confident |
| `local_entropy` | per-image | Normalized Top-5 entropy of Israeli, [0,1] |
| `global_top1_vs_top2` | per-image | Global decisiveness margin (p₁ − p₂) |
| `local_top1_vs_top2` | per-image | Israeli decisiveness margin (p₁ − p₂) |
| `arbiter_dominance_score` | per-candidate | `global_prob − israeli_prob` |

**Key normalization choice:** Both entropies are divided by `log(k)` so the
142-class Global model and the 13-class Israeli model land on the **same [0,1]
scale** and are directly comparable by the Arbiter.

**Target rule the model can now learn:** *"If `global_entropy` is LOW (Global
is peaked/confident) AND the Global top-1 is not an Israeli class → trust Global
unconditionally. Only when `global_entropy` is HIGH (Global is guessing) should
the Israeli signal be allowed to win."*

**Result:** *(fill after running)*

---

## 4. Evaluation Results

### v1 Evaluation (test split)

| Metric | Value |
|--------|-------|
| Overall Re-Rank Accuracy@1 | 84.90% |
| Global domain accuracy | 88.19% |
| Israeli domain accuracy | **28.2%** ← failure |

### v2 Evaluation (test split)

| Metric | Value |
|--------|-------|
| Overall Re-Rank Accuracy@1 | 69.53% |
| Global domain accuracy | **68.67%** ← over-corrected |
| Israeli domain accuracy | 85.98% |

### v3 Evaluation (test split)

| Metric | Value |
|--------|-------|
| Global domain accuracy | 74.90% ← still too low |
| Israeli domain accuracy | 84.14% |

### v4 Evaluation (test split)

> ⚠️ **To be filled after running `evaluate_system.py` with the v4 model.**

| Metric | Value |
|--------|-------|
| Overall Re-Rank Accuracy@1 | — |
| Global domain accuracy | — (target: ≥ 85%) |
| Israeli domain accuracy | — (target: ≥ 85%) |
| High-confidence error rate (score > 0.7) | — |
| Top confusion pair #1 (predicted → truth) | — |
| Top confusion pair #2 (predicted → truth) | — |
| Top confusion pair #3 (predicted → truth) | — |

---

## 5. Known Failure Modes

### 5.1 Majority-Class Overruling Bias
**Status:** Diagnosed → mitigated (v2) → over-corrected (v2) → re-balanced (v3)
→ resolved via evidence-based features (v4, pending evaluation).
See Section 3 version history for full timeline.

### 5.2 Hierarchical Misclassification
**Status:** Identified, not yet mitigated.

The Arbiter predicts a broader class (e.g., "steak") over the specific
ground-truth sub-class (e.g., "filet_mignon") because the Global model's
broad-class neuron accumulates probability mass from all visual sub-variants,
producing systematically higher confidence than the sub-class neuron. The
Re-Ranker, trained on the same flat label distribution, reinforces this behaviour.

---

## 6. Scripts Reference

| Script | Purpose | Key Output |
|--------|---------|------------|
| `arbiter/data/generate_reranker_dataset.py` | Build Re-Ranker CSV w/ evidence features (v2) | `arbiter/data/reranker_dataset.csv` |
| `arbiter/train/train_reranker_xgb.py` | Train XGBoost Re-Ranker (current: v4) | `arbiter/train/reranker_xgb.ubj` |
| `arbiter/train/evaluate_system.py` | Full pipeline evaluation on test splits | `arbiter/train/system_evaluation_results.csv` |

### Folder Structure

```
arbiter/
├── README.md
├── SYSTEM_STATE.md   (this file)
├── data/             generate_reranker_dataset.py + reranker_vector_mechanics.html
│                     (+ reranker_dataset.csv — generated, gitignored)
├── train/            train_reranker_xgb.py + evaluate_system.py
│                     + evaluation_mechanics_and_hierarchy.html + reranker_xgb.ubj
│                     (+ system_evaluation_results.csv — generated, gitignored)
└── old_versions/     superseded prototypes (router design, early arbiter)
```

> Scripts use absolute paths rooted at `BASE = E:\final project - models retrain`,
> so they run correctly regardless of the working directory. The v4 feature set
> (`global_entropy`, `local_entropy`, `global_top1_vs_top2`, `local_top1_vs_top2`,
> `arbiter_dominance_score`) is kept in the SAME column order across all three
> scripts — changing one requires changing all three.

---

## 7. TODO — XGBoost Arbiter Optimisations

### 7.1 Confidence Entropy Feature ✅ DONE (v4)
- **What:** Added `global_entropy` + `local_entropy` (normalized Top-5 entropy),
  plus `global_top1_vs_top2`, `local_top1_vs_top2` margins and
  `arbiter_dominance_score`.
- **Status:** `[x] Implemented in v4 — pending evaluation`

### 7.2 Class-Specific Score Penalties
- **What:** After evaluating, identify the top-N "attractor" classes.
  Apply a multiplicative discount `δ < 1` to their XGBoost scores at inference.
- **Why:** Targeted fix for known confusion pairs without full retraining.
- **How:** Build a `PENALTY_MAP` dict from confusion-pair analysis of the CSV.
- **Effort:** Low — post-processing only.
- **Status:** `[ ] Blocked — need v4 evaluation results first`

### 7.3 Hierarchical Label Smoothing
- **What:** Assign `target_label = 0.4` for candidates that are a known
  parent/sibling of the ground truth class.
- **Effort:** Medium — requires manual taxonomy definition.
- **Status:** `[ ] Not started`

### 7.4 Pairwise Re-Ranking Loss
- **What:** Switch objective from `binary:logistic` to `rank:pairwise`.
- **Why:** Directly optimises relative ordering, not absolute scores.
- **How:** Add `qid` column per image group; switch objective in training script.
- **Effort:** Medium.
- **Status:** `[ ] Not started`

### 7.5 Israeli-Specific Probability Scaling
- **What:** Scale up `israeli_prob` by a learned factor for images where
  the Israeli model has any non-zero output.
- **Why:** 13-class softmax produces higher raw confidence per class than
  142-class softmax — the Arbiter may still be underweighting this signal.
- **Effort:** Low.
- **Status:** `[ ] Not started`

---

## 8. Experiment Log

| Date | Version | Change | Global Acc | Israeli Acc | Overall Acc@1 |
|------|---------|--------|-----------|-------------|---------------|
| May 2026 | v1 | Baseline — global `scale_pos_weight` only | 88.19% | 28.20% | 84.90% |
| May 2026 | v2 | Three-layer instance weights (no normalization guard) | 68.67% | 85.98% | 69.53% |
| May 2026 | v3 | Normalize-last weighting + `max_delta_step=1` | 74.90% | 84.14% | — |
| May 2026 | v4 | Evidence-based routing (entropy/margin/dominance features) + pure 1:1 parity | — | — | — |

---

*Last updated: May 2026*
