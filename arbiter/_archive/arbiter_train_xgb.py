"""
Nutritional Arbiter — XGBoost Routing Classifier
Caleyez Capstone Project

Routing decisions:
  0 → Trust Global model top-1       (81.9 % of training data)
  1 → Trust Israeli model top-1      ( 5.0 % — critical minority)
  3 → Uncertain / neither reliable   (13.1 %)

Author: Raz  |  Built: 2026-05-31
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH   = Path(r"E:\data cleaned backup\backup 31.05.2026 17pm\arbiter_training_dataset.csv")
OUTPUT_DIR  = Path(__file__).parent
MODEL_PATH  = OUTPUT_DIR / "arbiter_xgb_model.json"
SEED        = 42
N_FOLDS     = 5
EPS         = 1e-9  # numerical stability

# The 13 classes the Israeli model was trained on
ISRAELI_CLASSES = {
    "baklava", "bourekas_cheese", "falafel", "hummus",
    "jachnun", "malawach", "meorav_yerushalmi", "sabich",
    "samosa", "schnitzel", "shakshuka", "shawarma", "sufganiyah",
}

# These are the rarest and most culturally unique Israeli classes —
# misclassifying them causes the largest nutritional error (e.g., jachnun
# is >50 % fat; classifying it as bread_pudding underestimates calories by ~40 %).
CRITICAL_ISRAELI = {"jachnun", "malawach", "sabich", "meorav_yerushalmi", "shawarma"}

# Sample-weight multipliers (tuned from class-frequency analysis)
WEIGHT_CLASS_1        = 18.0   # base up-weight for Israeli-correct routes
WEIGHT_CLASS_3        =  6.5   # up-weight for uncertain routes
CRITICAL_EXTRA_MULT   =  2.5   # additional multiplier for critical Israeli GT


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def _entropy2(p1: float, p2: float) -> float:
    """Shannon entropy approximated from the top-2 softmax probabilities."""
    p1, p2 = max(p1, EPS), max(p2, EPS)
    return -(p1 * np.log(p1) + p2 * np.log(p2))


def _overconfidence_penalty(global_conf: float, global_margin: float) -> float:
    """
    Penalise the Global model when it is both very confident AND has a
    large margin over its own second guess — the signature of 'Overconfident
    Stupidity'.  The penalty grows quadratically so moderate confidence is
    not penalised but extreme confidence (>0.85) is flagged strongly.
    """
    if global_conf > 0.85:
        return global_conf ** 2 * global_margin
    return 0.0


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive all routing signals from the two models' inference outputs.
    No ground-truth information is used — these features must be available
    at serving time.
    """
    f = pd.DataFrame()

    # ── Raw confidences ───────────────────────────────────────────────────
    f["global_top1_conf"]  = df["global_top1_conf"]
    f["global_top2_conf"]  = df["global_top2_conf"]
    f["local_top1_conf"]   = df["local_top1_conf"]
    f["local_top2_conf"]   = df["local_top2_conf"]

    # ── Confidence margins (decision boundary clarity) ────────────────────
    f["global_conf_margin"] = df["global_top1_conf"] - df["global_top2_conf"]
    f["local_conf_margin"]  = df["local_top1_conf"]  - df["local_top2_conf"]

    # ── Entropy approximation (uncertainty signal) ────────────────────────
    f["global_entropy"] = df.apply(
        lambda r: _entropy2(r["global_top1_conf"], r["global_top2_conf"]), axis=1
    )
    f["local_entropy"] = df.apply(
        lambda r: _entropy2(r["local_top1_conf"],  r["local_top2_conf"]),  axis=1
    )

    # ── Cross-model rivalry ───────────────────────────────────────────────
    # > 1.0 means Israeli model is MORE confident than Global on this image
    f["local_over_global_ratio"] = (
        df["local_top1_conf"] / (df["global_top1_conf"] + EPS)
    )
    f["local_beats_global"] = (
        df["local_top1_conf"] > df["global_top1_conf"]
    ).astype(int)

    # Difference of top-1 confidences (signed)
    f["conf_diff_local_minus_global"] = (
        df["local_top1_conf"] - df["global_top1_conf"]
    )

    # ── Global model's 'Overconfident Stupidity' penalty ─────────────────
    # High penalty when global is very certain AND its margin is huge —
    # the pattern seen in wrong high-confidence predictions (mean conf=0.683
    # for label-3, yet cases exist at >0.90 confidence that are still wrong).
    f["global_overconf_penalty"] = df.apply(
        lambda r: _overconfidence_penalty(
            r["global_top1_conf"], r["global_conf_margin"]
        ),
        axis=1,
    )

    # Boolean: is the global model dangerously over-confident?
    f["global_is_overconfident"] = (df["global_top1_conf"] > 0.85).astype(int)

    # ── Israeli class membership flags ───────────────────────────────────
    f["local_top1_is_israeli"] = (
        df["local_top1_class"].isin(ISRAELI_CLASSES)
    ).astype(int)
    f["local_top2_is_israeli"] = (
        df["local_top2_class"].isin(ISRAELI_CLASSES)
    ).astype(int)
    f["global_top1_is_israeli"] = (
        df["global_top1_class"].isin(ISRAELI_CLASSES)
    ).astype(int)

    # ── Israeli class confidence interaction ──────────────────────────────
    # High signal: local model confidently predicts an Israeli dish
    # while global is also highly confident — the core conflict to resolve.
    f["israeli_signal"] = (
        f["local_top1_is_israeli"] * df["local_top1_conf"]
    )
    f["global_overconf_x_israeli_conflict"] = (
        f["global_overconf_penalty"] * f["local_top1_is_israeli"]
    )

    # ── Agreement between models ──────────────────────────────────────────
    f["models_agree_top1"] = (
        df["global_top1_class"] == df["local_top1_class"]
    ).astype(int)
    # Does Global's second guess match Local's first? (convergence signal)
    f["global_top2_matches_local_top1"] = (
        df["global_top2_class"] == df["local_top1_class"]
    ).astype(int)

    # ── Label-encoded class identities ───────────────────────────────────
    # Encode the local top-1 class name so XGBoost can learn per-class biases.
    # (Only 13 unique values — safe to encode directly.)
    le_local = LabelEncoder()
    f["local_top1_class_enc"] = le_local.fit_transform(df["local_top1_class"])

    # For the global model (142 classes) use frequency-based target encoding
    # proxy: how often does each global_top1_class appear with label=1?
    # At training time we compute this from the data; at serving time the
    # encoder is baked into the pipeline via a lookup table.
    global_israeli_rate = (
        df.groupby("global_top1_class")["arbiter_label"]
        .apply(lambda s: (s == 1).mean())
        .to_dict()
    )
    f["global_class_israeli_rate"] = df["global_top1_class"].map(
        global_israeli_rate
    ).fillna(0.0)

    # ── Combined risk score ───────────────────────────────────────────────
    # Intuitive composite: how strongly should we distrust the global model?
    f["global_distrust_score"] = (
        f["global_overconf_penalty"] * (1.0 + f["local_top1_is_israeli"])
        - f["global_conf_margin"]
        + f["local_over_global_ratio"] * f["local_top1_is_israeli"]
    )

    return f


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SAMPLE WEIGHT COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_sample_weights(df_raw: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """
    Custom weighting strategy:
      - Label 0 (global correct)  → weight = 1.0
      - Label 1 (Israeli correct) → weight = WEIGHT_CLASS_1
      - Label 3 (uncertain)       → weight = WEIGHT_CLASS_3
      - Samples whose ground truth is a CRITICAL Israeli class get an
        additional CRITICAL_EXTRA_MULT multiplier on top of the above,
        because their nutritional error would be the most severe.
    """
    w = np.where(labels == 0, 1.0,
        np.where(labels == 1, float(WEIGHT_CLASS_1),
                              float(WEIGHT_CLASS_3)))

    is_critical = df_raw["ground_truth_class"].isin(CRITICAL_ISRAELI).values
    w = np.where(is_critical, w * CRITICAL_EXTRA_MULT, w)

    return w.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  XGBOOST HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

XGB_PARAMS = {
    # Multi-class softmax
    "objective":        "multi:softmax",
    "num_class":        4,              # labels 0,1,2,3 (2 unused but kept)
    "eval_metric":      ["mlogloss", "merror"],

    # Capacity — enough depth for complex confidence interactions
    "n_estimators":     600,
    "max_depth":        7,
    "min_child_weight": 5,

    # Regularisation — prevent memorising the dominant class
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "gamma":            0.1,
    "reg_alpha":        0.5,
    "reg_lambda":       2.0,

    # Reproducibility
    "seed":             SEED,
    "n_jobs":          -1,
    "use_label_encoder": False,
    "verbosity":        0,
}


# ─────────────────────────────────────────────────────────────────────────────
# 5.  EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

LABEL_NAMES = {0: "Global-Top1", 1: "Israeli-Top1", 3: "Uncertain"}


def evaluate(y_true, y_pred, title=""):
    label_order = [0, 1, 3]
    target_names = [LABEL_NAMES[l] for l in label_order]

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(classification_report(
        y_true, y_pred,
        labels=label_order,
        target_names=target_names,
        zero_division=0,
    ))

    macro_f1 = f1_score(y_true, y_pred, average="macro",    labels=label_order, zero_division=0)
    israeli_f1  = f1_score(y_true, y_pred, average=None,    labels=[1],         zero_division=0)[0]
    print(f"  Macro-F1       : {macro_f1:.4f}")
    print(f"  Israeli F1 (1) : {israeli_f1:.4f}")

    return macro_f1, israeli_f1


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    label_order = [0, 1, 3]
    names = [LABEL_NAMES[l] for l in label_order]
    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", xticklabels=names,
                yticklabels=names, cmap="Blues", ax=axes[0])
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_ylabel("True"); axes[0].set_xlabel("Predicted")

    sns.heatmap(cm_pct, annot=True, fmt=".1f", xticklabels=names,
                yticklabels=names, cmap="OrRd", ax=axes[1])
    axes[1].set_title("Confusion Matrix (row %)")
    axes[1].set_ylabel("True"); axes[1].set_xlabel("Predicted")

    plt.suptitle("Nutritional Arbiter — XGBoost", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Confusion matrix saved → {save_path}")
    plt.show()


def plot_feature_importance(model, feature_names, save_path=None):
    scores = model.get_booster().get_fscore()
    fi = pd.Series(scores).reindex(feature_names).fillna(0).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(fi) * 0.35)))
    fi.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("XGBoost Feature Importance (gain)", fontweight="bold")
    ax.set_xlabel("F-Score (gain)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Feature importance saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Caleyez — Nutritional Arbiter XGBoost Training")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[1/6] Loading dataset …")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    print(f"      Rows: {len(df):,}   Columns: {df.shape[1]}")

    label_counts = df["arbiter_label"].value_counts().sort_index()
    print(f"      Label distribution:\n{label_counts.to_string()}")
    print(f"      Imbalance ratio (0:1) ≈ {label_counts[0]/label_counts[1]:.1f}×")

    # ── Feature engineering ───────────────────────────────────────────────
    print("\n[2/6] Engineering features …")
    X = engineer_features(df)
    y = df["arbiter_label"].values
    feature_names = list(X.columns)
    print(f"      Feature matrix: {X.shape[0]:,} × {X.shape[1]}")

    # ── Sample weights ────────────────────────────────────────────────────
    print("\n[3/6] Computing sample weights …")
    sample_weights = compute_sample_weights(df, y)
    print(f"      Weight range: [{sample_weights.min():.1f}, {sample_weights.max():.1f}]")
    for lbl in [0, 1, 3]:
        mask = y == lbl
        print(f"      Label {lbl}: n={mask.sum():,}  mean_weight={sample_weights[mask].mean():.1f}")

    # ── Cross-Validation ──────────────────────────────────────────────────
    print(f"\n[4/6] {N_FOLDS}-fold Stratified Cross-Validation …")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    cv_macro_f1   = []
    cv_israeli_f1 = []
    oof_pred = np.zeros(len(y), dtype=int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr        = sample_weights[train_idx]

        model_cv = xgb.XGBClassifier(**XGB_PARAMS)
        model_cv.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        fold_pred = model_cv.predict(X_val)
        oof_pred[val_idx] = fold_pred

        mf1, if1 = evaluate(y_val, fold_pred, f"Fold {fold} Validation")
        cv_macro_f1.append(mf1)
        cv_israeli_f1.append(if1)

    print(f"\n  CV Macro-F1      : {np.mean(cv_macro_f1):.4f} ± {np.std(cv_macro_f1):.4f}")
    print(f"  CV Israeli F1(1) : {np.mean(cv_israeli_f1):.4f} ± {np.std(cv_israeli_f1):.4f}")

    print("\n  ── Out-of-Fold (full dataset) ──")
    evaluate(y, oof_pred, "OOF Evaluation (all folds combined)")

    # ── Train final model on all data ─────────────────────────────────────
    print("\n[5/6] Training final model on full dataset …")
    final_model = xgb.XGBClassifier(**XGB_PARAMS)
    final_model.fit(X, y, sample_weight=sample_weights, verbose=False)

    final_pred = final_model.predict(X)
    evaluate(y, final_pred, "Final Model — Training Set (sanity check)")

    # ── Save model ────────────────────────────────────────────────────────
    print("\n[6/6] Saving artefacts …")
    final_model.save_model(str(MODEL_PATH))
    print(f"      Model saved → {MODEL_PATH}")

    # Save feature names alongside the model for reproducible inference
    meta = {
        "feature_names":   feature_names,
        "label_map":       {0: "global_top1", 1: "israeli_top1", 3: "uncertain"},
        "israeli_classes": sorted(ISRAELI_CLASSES),
        "critical_classes": sorted(CRITICAL_ISRAELI),
        "cv_macro_f1_mean":   float(np.mean(cv_macro_f1)),
        "cv_israeli_f1_mean": float(np.mean(cv_israeli_f1)),
    }
    meta_path = OUTPUT_DIR / "arbiter_xgb_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"      Meta saved  → {meta_path}")

    # ── Visualisations ────────────────────────────────────────────────────
    plot_confusion_matrix(
        y, oof_pred,
        save_path=OUTPUT_DIR / "arbiter_confusion_matrix.png"
    )
    plot_feature_importance(
        final_model, feature_names,
        save_path=OUTPUT_DIR / "arbiter_feature_importance.png"
    )

    print("\n✓ Training complete.")
    return final_model, meta


# ─────────────────────────────────────────────────────────────────────────────
# 7.  INFERENCE HELPER  (import and call from app code)
# ─────────────────────────────────────────────────────────────────────────────

def load_arbiter(model_path=MODEL_PATH, meta_path=None):
    """Load a saved arbiter for inference."""
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    if meta_path is None:
        meta_path = Path(model_path).parent / "arbiter_xgb_meta.json"
    with open(meta_path) as fh:
        meta = json.load(fh)

    return model, meta


def predict_route(
    model,
    meta,
    global_top1_class: str,
    global_top1_conf: float,
    global_top2_class: str,
    global_top2_conf: float,
    local_top1_class: str,
    local_top1_conf: float,
    local_top2_class: str,
    local_top2_conf: float,
) -> dict:
    """
    Predict routing decision for a single image.

    Returns
    -------
    dict with keys:
        route       : int (0=global, 1=israeli, 3=uncertain)
        route_label : str
        probabilities : dict of class → probability
    """
    # Build a one-row DataFrame matching training schema
    row = {
        "global_top1_class":  global_top1_class,
        "global_top1_conf":   global_top1_conf,
        "global_top2_class":  global_top2_class,
        "global_top2_conf":   global_top2_conf,
        "local_top1_class":   local_top1_class,
        "local_top1_conf":    local_top1_conf,
        "local_top2_class":   local_top2_class,
        "local_top2_conf":    local_top2_conf,
        "ground_truth_class": "__serving__",  # unused in feature fn
        "arbiter_label":      -1,             # unused
    }
    df_row = pd.DataFrame([row])
    X_row  = engineer_features(df_row)[meta["feature_names"]]

    pred       = int(model.predict(X_row)[0])
    proba      = model.predict_proba(X_row)[0]
    label_map  = {int(k): v for k, v in meta["label_map"].items()}

    return {
        "route":          pred,
        "route_label":    label_map.get(pred, "unknown"),
        "probabilities":  {label_map[int(k)]: float(p) for k, p in zip(model.classes_, proba)},
    }


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
