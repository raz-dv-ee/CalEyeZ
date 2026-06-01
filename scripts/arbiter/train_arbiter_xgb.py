"""
Arbiter (router) training — CalEyeZ.

Learns to route each image to the correct expert model from both models' output
signals. Because the two label spaces are disjoint, the routing target is the image
DOMAIN: route_to_israeli = (domain == 'israeli'). The arbiter never sees the domain —
it predicts it from confidence / entropy / margin features of both models.

Honest evaluation:
  - Trained on the rows whose images came from the global model's VAL split.
  - Evaluated on the rows from the TEST split (held out from training the arbiter AND
    from training the global model) -> no leakage.

Reports routing accuracy, ROC-AUC, and the money metric: the final ROUTED system top-1
vs. the "always global" baseline and the oracle ceiling.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
CSV = ROOT / "datasets" / "arbiter_dataset.csv"
MODEL_OUT = ROOT / "scripts" / "arbiter" / "arbiter_xgb.json"
ALPHA = 0.5  # domain-weight dial: scale_pos_weight = (n_neg/n_pos)**ALPHA

BASE_FEATS = ([f"g_conf{k}" for k in range(1, 6)] + ["g_entropy", "g_margin"]
              + [f"i_conf{k}" for k in range(1, 6)] + ["i_entropy", "i_margin"])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction features (pure functions of the base signals; no label leakage)."""
    df = df.copy()
    df["conf_gap"]    = df["g_conf1"] - df["i_conf1"]          # global vs israeli top-1 conf
    df["conf_ratio"]  = df["g_conf1"] / (df["i_conf1"] + 1e-6)
    df["entropy_gap"] = df["i_entropy"] - df["g_entropy"]      # >0 => israeli more uncertain
    df["margin_gap"]  = df["g_margin"] - df["i_margin"]
    df["both_unsure"] = ((df["g_conf1"] < 0.5) & (df["i_conf1"] < 0.5)).astype(int)
    return df


FEATS = BASE_FEATS + ["conf_gap", "conf_ratio", "entropy_gap", "margin_gap", "both_unsure"]


def main() -> None:
    df = pd.read_csv(CSV)
    df["split"] = np.where(df["image_path"].str.contains(r"[\\/]val[\\/]"), "val", "test")
    df["y"] = (df["domain"] == "israeli").astype(int)
    df = add_features(df)

    tr = df[df.split == "val"]
    te = df[df.split == "test"]
    print(f"train rows (val):  {len(tr):>6}  (israeli {tr.y.sum()}, {tr.y.mean()*100:.1f}%)")
    print(f"test  rows (test): {len(te):>6}  (israeli {te.y.sum()}, {te.y.mean()*100:.1f}%)")

    Xtr, ytr = tr[FEATS], tr["y"]
    Xte, yte = te[FEATS], te["y"]

    n_pos, n_neg = int(ytr.sum()), int((ytr == 0).sum())
    spw = (n_neg / n_pos) ** ALPHA
    print(f"scale_pos_weight = ({n_neg}/{n_pos})^{ALPHA} = {spw:.2f}")

    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="auc",
        min_child_weight=3, reg_lambda=1.0, n_jobs=4,
    )
    clf.fit(Xtr, ytr)

    proba = clf.predict_proba(Xte)[:, 1]
    pred_israeli = proba >= 0.5

    # ---- routing metrics ----
    auc = roc_auc_score(yte, proba)
    route_acc = (pred_israeli == yte.values).mean()
    # domain-wise recall
    isr = yte.values == 1
    isr_recall = (pred_israeli[isr]).mean()
    glb_recall = (~pred_israeli[~isr]).mean()

    # ---- money metric: final routed system top-1 ----
    g_corr = te["g_correct"].values.astype(bool)
    i_corr = te["i_correct"].values.astype(bool)
    routed_correct = np.where(pred_israeli, i_corr, g_corr)
    routed_acc  = routed_correct.mean()
    base_acc    = g_corr.mean()                 # always trust global
    oracle_acc  = (g_corr | i_corr).mean()      # perfect router

    print("\n=== ROUTING (domain detection) on test ===")
    print(f"  routing accuracy : {route_acc*100:.2f}%")
    print(f"  ROC-AUC          : {auc:.4f}")
    print(f"  israeli recall   : {isr_recall*100:.2f}%")
    print(f"  global  recall   : {glb_recall*100:.2f}%")

    print("\n=== FINAL ROUTED SYSTEM top-1 on test ===")
    print(f"  always-global baseline : {base_acc*100:.2f}%")
    print(f"  ARBITER (routed)       : {routed_acc*100:.2f}%")
    print(f"  oracle ceiling         : {oracle_acc*100:.2f}%")
    gain = (routed_acc - base_acc) * 100
    closed = (routed_acc - base_acc) / (oracle_acc - base_acc) * 100 if oracle_acc > base_acc else 0
    print(f"  -> arbiter vs baseline : {gain:+.2f} pts  ({closed:.0f}% of the way to oracle)")

    print("\n=== top features by gain ===")
    imp = sorted(zip(FEATS, clf.feature_importances_), key=lambda x: -x[1])[:8]
    for f, v in imp:
        print(f"  {f:<14} {v:.3f}")

    clf.save_model(str(MODEL_OUT))
    print(f"\nSaved arbiter -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
