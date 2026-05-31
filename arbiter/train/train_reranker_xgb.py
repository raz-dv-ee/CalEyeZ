"""
train_reranker_xgb.py  —  v5: Tunable Domain Dial + Interaction Features
========================================================================
What changed from v4 (and why)
------------------------------
A full empirical study (oracle + sweep over the real data) established the
hard limits of this two-model system:

  • Global images are 94.9% of the data; the Global YOLO's own top-1 caps
    at 88.19%. An arbiter that only sees the models' probability vectors
    cannot reliably beat a model's own argmax — so Global accuracy is
    bounded near ~88%.
  • The Israeli model alone scores 92.26% on Israeli images.
  • PERFECT routing (an oracle gate) therefore caps overall at 88.40%.
    ⇒ 90%+ is mathematically unreachable WITHOUT improving the Global model.
    The bottleneck is the Global YOLO, not the arbiter.

v4 (rigid 1:1 domain parity) sat at a poor operating point: overall 80.33%,
Global 80.41%, Israeli 78.85% — it actively hurt the majority Global domain.

v5 makes the global↔israeli trade-off an explicit DIAL and adds interaction
features (derived from the existing 9 columns — no dataset re-generation):

  DOMAIN_WEIGHT_ALPHA ∈ [0,1] controls Israeli up-weighting:
      israeli_row_weight = (n_global / n_israeli) ** ALPHA
      ALPHA = 0.0 → no domain weighting   (max overall ~85.6%, Israeli ~54%)
      ALPHA = 0.3 → balanced (DEFAULT)     (overall ~85.0%, Israeli ~63%)
      ALPHA = 1.0 → full parity (= v4)     (overall ~80%,  Israeli ~79%)

  Interaction features let the trees express conditional trust cleanly:
      conf_global    = global_prob  * (1 - global_entropy)
      conf_israeli   = israeli_prob * (1 - local_entropy)
      israeli_top1_p = israeli_prob * is_israeli_top1
      global_top1_p  = global_prob  * is_global_top1
      prob_ratio     = israeli_prob / (global_prob + 0.05)
      both_agree     = is_global_top1 * is_israeli_top1

NOTE: evaluate_system.py must compute these same derived features in the
same order. They are pure functions of the base 9 columns.
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = r"E:\final project - models retrain"
DATASET_CSV  = rf"{BASE}\arbiter\data\reranker_dataset.csv"
MODEL_OUTPUT = rf"{BASE}\arbiter\train\reranker_xgb.ubj"
RANDOM_SEED  = 42

# ── v5 dial & params ──
DOMAIN_WEIGHT_ALPHA = 0.3    # 0=max overall, 0.3=balanced (default), 1.0=full parity (v4)
SCALE_POS_WEIGHT    = 3.0    # candidate-set imbalance; 3 worked best in the sweep

BASE_FEATURE_COLS = [
    # ── raw probability features ──
    "global_prob",
    "israeli_prob",
    "is_global_top1",
    "is_israeli_top1",
    # ── v4 evidence-based / confusion features ──
    "global_entropy",
    "local_entropy",
    "global_top1_vs_top2",
    "local_top1_vs_top2",
    "arbiter_dominance_score",
]
# ── v5 interaction features (derived from the base columns) ──
DERIVED_FEATURE_COLS = [
    "conf_global",
    "conf_israeli",
    "israeli_top1_p",
    "global_top1_p",
    "prob_ratio",
    "both_agree",
]
FEATURE_COLS = BASE_FEATURE_COLS + DERIVED_FEATURE_COLS
TARGET_COL = "target_label"


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the v5 interaction features. MUST match evaluate_system.py."""
    g = df["global_prob"]
    i = df["israeli_prob"]
    df["conf_global"]    = g * (1.0 - df["global_entropy"])
    df["conf_israeli"]   = i * (1.0 - df["local_entropy"])
    df["israeli_top1_p"] = i * df["is_israeli_top1"]
    df["global_top1_p"]  = g * df["is_global_top1"]
    df["prob_ratio"]     = i / (g + 0.05)
    df["both_agree"]     = df["is_global_top1"] * df["is_israeli_top1"]
    return df

ISRAELI_CLASSES = {
    "baklava", "bourekas_cheese", "falafel", "hummus", "jachnun",
    "malawach", "meorav_yerushalmi", "sabich", "samosa", "schnitzel",
    "shakshuka", "shawarma", "sufganiyah",
}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_splits(csv_path: str):
    df = pd.read_csv(csv_path)
    required = set(BASE_FEATURE_COLS + [TARGET_COL, "split_type", "image_path",
                                        "candidate_class", "ground_truth_class"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    df = add_derived_features(df)   # v5 interaction features

    train_df = df[df["split_type"] == "val"].copy()
    test_df  = df[df["split_type"] == "test"].copy()

    if test_df.empty:
        print("No 'test' split found — using 80/20 random split of val data.")
        rng      = np.random.default_rng(RANDOM_SEED)
        mask     = rng.random(len(train_df)) < 0.8
        test_df  = train_df[~mask].copy()
        train_df = train_df[mask].copy()

    train_df["is_israeli"] = train_df["ground_truth_class"].isin(ISRAELI_CLASSES)
    test_df["is_israeli"]  = test_df["ground_truth_class"].isin(ISRAELI_CLASSES)

    _print_split_stats("Train", train_df)
    _print_split_stats("Test",  test_df)
    return train_df, test_df


def _print_split_stats(name: str, df: pd.DataFrame):
    n_global  = (~df["is_israeli"]).sum()
    n_israeli = df["is_israeli"].sum()
    n_pos     = df[TARGET_COL].sum()
    print(f"{name} rows : {len(df):>9,}  |  "
          f"global: {n_global:,}  israeli: {n_israeli:,}  |  "
          f"positive: {n_pos:,}")


# ── Instance weight computation ───────────────────────────────────────────────

def compute_instance_weights(df: pd.DataFrame,
                             alpha: float = DOMAIN_WEIGHT_ALPHA) -> np.ndarray:
    """
    Tunable domain weighting (v5).

      w[is_global]  = 1.0
      w[is_israeli] = (n_global_rows / n_israeli_rows) ** alpha

    alpha = 0 → no domain weighting (max overall; Israeli weak)
    alpha = 1 → full 1:1 parity (= v4; balanced but hurts Global)
    alpha = 0.3 (default) → balanced sweet spot found in the sweep.
    """
    n           = len(df)
    weights     = np.ones(n, dtype=np.float64)
    is_israeli  = df["is_israeli"].values
    is_global   = ~is_israeli

    n_global_rows  = int(is_global.sum())
    n_israeli_rows = int(is_israeli.sum())

    if n_israeli_rows == 0:
        print("  WARN: no Israeli rows found — skipping domain balancing.")
        return weights.astype(np.float32)

    full_ratio  = n_global_rows / n_israeli_rows
    israeli_w   = full_ratio ** alpha
    weights[is_israeli] = israeli_w

    sum_g = weights[is_global].sum()
    sum_i = weights[is_israeli].sum()

    print(f"\n── Instance Weight Diagnostics (v5 — alpha = {alpha}) ───────────")
    print(f"  Global rows    : {n_global_rows:,}   (weight = 1.000)")
    print(f"  Israeli rows   : {n_israeli_rows:,}   "
          f"(full ratio = {full_ratio:.2f}×, weight = {israeli_w:.3f}× at alpha={alpha})")
    print(f"  sum(w_global)  = {sum_g:,.1f}")
    print(f"  sum(w_israeli) = {sum_i:,.1f}   (ratio g/i = {sum_g/sum_i:.3f})")

    return weights.astype(np.float32)


# ── Positive/negative balance ─────────────────────────────────────────────────

def compute_scale_pos_weight(y: pd.Series) -> float:
    """
    Global scale_pos_weight = n_neg / n_pos.
    Applied symmetrically across both domains via XGBoost params.
    Since the candidate-set imbalance (~9 wrong per correct) is the same
    in both domains, a single global value is correct — no per-domain split.
    """
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    if n_pos == 0:
        raise ValueError("No positive labels found in training data.")
    spw = n_neg / n_pos
    print(f"  scale_pos_weight = {n_neg:,} / {n_pos:,} = {spw:.2f}")
    return float(spw)


# ── Ranking evaluation ────────────────────────────────────────────────────────

def rerank_accuracy(df: pd.DataFrame, score_col: str = "xgb_score") -> float:
    best = (
        df.sort_values(score_col, ascending=False)
          .groupby("image_path")
          .first()
          .reset_index()
    )
    correct = (best["candidate_class"] == best["ground_truth_class"]).sum()
    return correct / len(best) if len(best) > 0 else 0.0


def rerank_accuracy_at_k(df: pd.DataFrame, k: int,
                          score_col: str = "xgb_score") -> float:
    def hit(g):
        top = g.nlargest(k, score_col)["candidate_class"]
        return g["ground_truth_class"].iloc[0] in top.values
    return df.groupby("image_path").apply(hit).mean()


def mean_reciprocal_rank(df: pd.DataFrame, score_col: str = "xgb_score") -> float:
    def rr(g):
        ranked = g.sort_values(score_col, ascending=False).reset_index(drop=True)
        gt     = ranked["ground_truth_class"].iloc[0]
        match  = ranked[ranked["candidate_class"] == gt]
        return 0.0 if match.empty else 1.0 / (match.index[0] + 1)
    return df.groupby("image_path").apply(rr).mean()


def domain_rerank_accuracy(df: pd.DataFrame,
                            score_col: str = "xgb_score") -> dict[str, float]:
    results = {}
    for domain, flag in [("global", False), ("israeli", True)]:
        subset = df[df["is_israeli"] == flag]
        results[domain] = rerank_accuracy(subset, score_col) if not subset.empty else float("nan")
    return results


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    np.random.seed(RANDOM_SEED)

    print(f"\nLoading dataset: {args.dataset}")
    train_df, test_df = load_splits(args.dataset)

    X_train = train_df[FEATURE_COLS].astype(float)
    y_train = train_df[TARGET_COL].astype(int)
    X_test  = test_df[FEATURE_COLS].astype(float)
    y_test  = test_df[TARGET_COL].astype(int)

    print("\n── Weight Computation ──────────────────────────────────────────")
    train_weights = compute_instance_weights(train_df, alpha=args.alpha)
    spw           = SCALE_POS_WEIGHT
    print(f"  scale_pos_weight = {spw}  (fixed; tuned via sweep)")

    dtrain = xgb.DMatrix(X_train, label=y_train,
                         weight=train_weights,
                         feature_names=FEATURE_COLS)
    dtest  = xgb.DMatrix(X_test, label=y_test,
                         feature_names=FEATURE_COLS)

    params = {
        "objective":        "binary:logistic",
        "eval_metric":      ["logloss", "auc"],
        "scale_pos_weight": spw,
        "max_depth":        6,
        "eta":              0.05,
        "subsample":        0.8,
        "colsample_bytree": 1.0,
        "min_child_weight": 3,
        # max_delta_step: stabilises gradient updates on weighted data;
        # a value of 1 prevents runaway step sizes on Israeli minority rows.
        "max_delta_step":   1,
        "seed":             RANDOM_SEED,
        "tree_method":      "hist",
    }

    print(f"\nTraining XGBoost "
          f"(up to {args.rounds} rounds, early-stop {args.early_stop})…")
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round       = args.rounds,
        evals                 = [(dtrain, "train"), (dtest, "eval")],
        early_stopping_rounds = args.early_stop,
        evals_result          = evals_result,
        verbose_eval          = 50,
    )

    # ── Binary classification metrics ─────────────────────────────────────
    y_score = model.predict(dtest)
    y_pred  = (y_score >= 0.5).astype(int)

    print("\n── Binary Classification Metrics ──────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Wrong", "Correct"]))
    print(f"ROC-AUC : {roc_auc_score(y_test, y_score):.4f}")
    print(f"Avg Prec: {average_precision_score(y_test, y_score):.4f}")

    # ── Ranking metrics ───────────────────────────────────────────────────
    test_df = test_df.copy()
    test_df["xgb_score"] = y_score

    ra1        = rerank_accuracy(test_df)
    ra3        = rerank_accuracy_at_k(test_df, k=3)
    mrr        = mean_reciprocal_rank(test_df)
    domain_acc = domain_rerank_accuracy(test_df)

    print("\n── Re-Ranking Metrics — Overall ────────────────────────────────")
    print(f"  Re-Rank Accuracy@1   : {ra1*100:.2f}%")
    print(f"  Re-Rank Accuracy@3   : {ra3*100:.2f}%")
    print(f"  Mean Reciprocal Rank : {mrr:.4f}")

    print("\n── Re-Ranking Metrics — By Domain ──────────────────────────────")
    for domain, acc in domain_acc.items():
        marker = "✓" if acc >= 0.85 else "✗"
        print(f"  {marker} {domain.capitalize():<10}: {acc*100:.2f}%  "
              f"{'(target met)' if acc >= 0.85 else '(below 85% target)'}")

    # ── Baseline comparison ───────────────────────────────────────────────
    print("\n── Baseline vs Re-Ranker (per domain) ──────────────────────────")
    for domain, flag in [("global", False), ("israeli", True)]:
        subset   = test_df[test_df["is_israeli"] == flag]
        baseline = subset[subset["is_global_top1"] == 1]
        if baseline.empty:
            continue
        b_acc = (baseline["candidate_class"] ==
                 baseline["ground_truth_class"]).mean()
        d_acc = domain_acc[domain]
        print(f"  {domain.capitalize():<10} | baseline: {b_acc*100:.2f}%  "
              f"→  re-ranker: {d_acc*100:.2f}%  "
              f"(lift: {(d_acc-b_acc)*100:+.2f} pp)")

    # ── Feature importance ────────────────────────────────────────────────
    print("\n── Feature Importances (gain) ──────────────────────────────────")
    importance = model.get_score(importance_type="gain")
    for feat, score in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {feat:<22s} {score:.4f}")

    model.save_model(args.output)
    print(f"\nModel saved → {args.output}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train XGBoost Re-Ranker (v5 — tunable domain dial).")
    p.add_argument("--dataset",    default=DATASET_CSV)
    p.add_argument("--output",     default=MODEL_OUTPUT)
    p.add_argument("--rounds",     type=int,   default=600)
    p.add_argument("--early-stop", type=int,   default=40)
    p.add_argument("--alpha",      type=float, default=DOMAIN_WEIGHT_ALPHA,
                   help="Israeli domain up-weight exponent: 0=max overall, "
                        "0.3=balanced (default), 1.0=full parity (v4).")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
