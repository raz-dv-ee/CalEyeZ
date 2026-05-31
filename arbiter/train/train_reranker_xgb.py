"""
train_reranker_xgb.py  —  v4: Evidence-Based Routing
=====================================================
Why v4 replaces v3
------------------
v3 achieved exact domain parity (sum(w_israeli) == sum(w_global)) and lifted
Israeli accuracy to 84.14%, but Global accuracy still dropped (88% → 74.9%).
The remaining problem: with ONLY raw probabilities as features, the only way
the model could help the Israeli minority was to globally lower its trust in
the Global model — which inevitably hurt the (majority) Global domain.

The v4 insight — STOP fighting the imbalance with weights alone; give the
model better EVIDENCE. We now feed engineered "confusion features" so the
Arbiter can make a *conditional* decision instead of a blanket one:

    "If global_entropy is LOW (Global is very confident / peaked) AND the
     Global top-1 is not an Israeli class → trust Global unconditionally.
     Only when global_entropy is HIGH (Global is guessing) should the
     Israeli signal be allowed to win."

New features consumed (produced by generate_reranker_dataset.py v2):
    global_entropy, local_entropy          — normalized Top-5 entropy [0,1]
    global_top1_vs_top2, local_top1_vs_top2 — decisiveness margins
    arbiter_dominance_score                 — (global_prob - israeli_prob)

Weighting policy in v4
----------------------
  • Keep ONLY the clean 1:1 domain parity instance weighting from v3.
  • REMOVE the high-confidence multipliers entirely (they caused over-
    correction and are now redundant — the entropy features carry that
    signal in a learnable, conditional form).
  • Keep the global scale_pos_weight (~9) for the candidate-set imbalance.

Net effect: the model balances domains via weights, but learns WHEN to
override the Global model via features — no longer a blunt instrument.
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

FEATURE_COLS = [
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
TARGET_COL = "target_label"

ISRAELI_CLASSES = {
    "baklava", "bourekas_cheese", "falafel", "hummus", "jachnun",
    "malawach", "meorav_yerushalmi", "sabich", "samosa", "schnitzel",
    "shakshuka", "shawarma", "sufganiyah",
}

# Confidence redistribution parameters
# NOTE (v4): The high-confidence multipliers (CONF_THRESHOLD / CONF_MULTIPLIER)
# from v2/v3 have been REMOVED. That signal is now carried by the engineered
# entropy / margin features, which the model can use *conditionally* instead of
# as a blanket weight bias. Weighting is now pure 1:1 domain parity.


# ── Data loading ──────────────────────────────────────────────────────────────

def load_splits(csv_path: str):
    df = pd.read_csv(csv_path)
    required = set(FEATURE_COLS + [TARGET_COL, "split_type", "image_path",
                                   "candidate_class", "ground_truth_class"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

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

def compute_instance_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Pure 1:1 domain-parity weighting (v4).

      w[is_global]  = 1.0
      w[is_israeli] = n_global_rows / n_israeli_rows

    This makes sum(w_israeli) == sum(w_global) EXACTLY by construction — no
    renormalization step is needed because there are no other multipliers.
    All conditional "trust" logic now lives in the engineered features, not
    in the weights. The weights only ensure the optimiser pays equal attention
    to both domains; the features decide WHEN to favour each model.
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

    base_factor = n_global_rows / n_israeli_rows
    weights[is_israeli] = base_factor

    sum_g = weights[is_global].sum()
    sum_i = weights[is_israeli].sum()
    ratio = sum_g / sum_i

    print(f"\n── Instance Weight Diagnostics (v4 — pure 1:1 parity) ──────────")
    print(f"  Global rows    : {n_global_rows:,}   (weight = 1.000)")
    print(f"  Israeli rows   : {n_israeli_rows:,}   "
          f"(domain factor = {base_factor:.3f}×)")
    print(f"  sum(w_global)  = {sum_g:,.1f}")
    print(f"  sum(w_israeli) = {sum_i:,.1f}")
    status = "✓ EXACT PARITY" if abs(ratio - 1.0) < 1e-6 else f"⚠ ratio = {ratio:.6f}"
    print(f"  Domain ratio   = {ratio:.6f}  {status}")

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
    train_weights = compute_instance_weights(train_df)
    spw           = compute_scale_pos_weight(y_train)

    dtrain = xgb.DMatrix(X_train, label=y_train,
                         weight=train_weights,
                         feature_names=FEATURE_COLS)
    dtest  = xgb.DMatrix(X_test, label=y_test,
                         feature_names=FEATURE_COLS)

    params = {
        "objective":        "binary:logistic",
        "eval_metric":      ["logloss", "auc"],
        "scale_pos_weight": spw,
        "max_depth":        5,
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
        description="Train XGBoost Re-Ranker (v4 — evidence-based routing).")
    p.add_argument("--dataset",    default=DATASET_CSV)
    p.add_argument("--output",     default=MODEL_OUTPUT)
    p.add_argument("--rounds",     type=int, default=600)
    p.add_argument("--early-stop", type=int, default=40)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
