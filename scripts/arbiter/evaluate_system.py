"""
Full-system evaluation — CalEyeZ.

Applies the trained arbiter to the TEST split only and records, per image, the routing
decision and the final system prediction. Writes a per-image CSV and prints the overall
system top-1.

Why TEST only: the arbiter was TRAINED on the val rows, so val is not a valid
generalization metric. The test split is held out from BOTH the global model and the
arbiter -> the system top-1 here is the honest, reportable number.

No GPU needed: both models' outputs are already in arbiter_dataset.csv; we only apply
the arbiter's routing decision on top.
"""

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent.parent
CSV = ROOT / "datasets" / "arbiter_dataset.csv"
ARBITER = ROOT / "scripts" / "arbiter" / "arbiter_xgb.json"
OUT = ROOT / "datasets" / "system_evaluation.csv"

sys.path.insert(0, str(ROOT / "scripts" / "arbiter"))
from train_arbiter_xgb import add_features, FEATS  # reuse identical feature build


def summarize(d: pd.DataFrame, label: str) -> None:
    n = len(d)
    sys_acc = d["final_correct"].mean() * 100
    base = d["g_correct"].mean() * 100
    oracle = (d["g_correct"].astype(bool) | d["i_correct"].astype(bool)).mean() * 100
    print(f"\n=== {label}  (n={n}) ===")
    print(f"  system top-1 (routed) : {sys_acc:.2f}%")
    print(f"  always-global baseline: {base:.2f}%")
    print(f"  oracle ceiling        : {oracle:.2f}%")
    for dom in ("global", "israeli"):
        m = d["domain"] == dom
        if m.any():
            print(f"    {dom:8} n={m.sum():5}  routed={d.loc[m,'final_correct'].mean()*100:.2f}%")


def main() -> None:
    df = pd.read_csv(CSV)
    df["split"] = np.where(df["image_path"].str.contains(r"[\\/]val[\\/]"), "val", "test")
    df = df[df.split == "test"].reset_index(drop=True)   # TEST only (held out)
    feat = add_features(df)

    clf = xgb.XGBClassifier()
    clf.load_model(str(ARBITER))
    df["route_prob_israeli"] = clf.predict_proba(feat[FEATS])[:, 1]
    df["route"] = np.where(df["route_prob_israeli"] >= 0.5, "israeli", "global")
    df["final_pred"] = np.where(df["route"] == "israeli", df["i_top1"], df["g_top1"])
    df["final_correct"] = (df["final_pred"] == df["true_class"]).astype(int)

    out = df[["image_path", "split", "domain", "true_class",
              "g_top1", "g_conf1", "i_top1", "i_conf1",
              "route", "route_prob_israeli", "final_pred", "final_correct",
              "g_correct", "i_correct"]]
    out.to_csv(OUT, index=False)
    print(f"Wrote per-image evaluation -> {OUT}  ({len(out)} rows, TEST only)")

    summarize(df, "TEST  (held out from global model + arbiter — VALID system metric)")


if __name__ == "__main__":
    main()
