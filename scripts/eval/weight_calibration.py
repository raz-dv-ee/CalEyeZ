"""
Weight-channel calibration (Experiment B): does the scale reading match true mass, and
can a simple software correction fix it?

Two accepted CSV formats (auto-detected):
  A) single reading per row:   label,true_g,read_g
  B) repeated readings:        item,true_g,r1,r2,r3,r4,r5   (any number of r* columns)

Format B additionally reports REPEATABILITY (spread of the 5 reads) and fits a linear
software correction  corrected_g = a + b * read  by least squares over the items, then
reports the residual error AFTER correction. Calories scale linearly with weight, so a
corrected weight directly fixes the calorie channel.

Usage:  py -3 scripts/eval/weight_calibration.py scripts/eval/weights_repeat.csv
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", type=Path, nargs="?", default=Path("scripts/eval/weights_cal.csv"))
    args = ap.parse_args()
    if not args.weights.exists():
        sys.exit(f"No file at {args.weights}. Create it with columns: item,true_g,r1..r5  (or label,true_g,read_g)")

    df = pd.read_csv(args.weights, comment="#")
    df.columns = [c.strip() for c in df.columns]
    df = df[df["true_g"].notna()]
    if df.empty:
        sys.exit("weights file has no filled-in rows yet.")

    name_col = "item" if "item" in df.columns else "label"
    read_cols = [c for c in df.columns if c.lower().startswith("r") and c != name_col and c != "true_g"]
    if "read_g" in df.columns:
        read_cols = ["read_g"]
    if not read_cols:
        sys.exit("No reading columns found (need r1..r5 or read_g).")

    reads = df[read_cols].apply(pd.to_numeric, errors="coerce")
    if reads.isna().all().all():
        sys.exit("Reading columns are empty - fill in the scale readings first.")

    df["mean_read"] = reads.mean(axis=1)
    df = df[df["mean_read"].notna()].reset_index(drop=True)   # skip rows not filled in yet
    reads = reads.loc[df.index] if False else df[read_cols].apply(pd.to_numeric, errors="coerce")
    if df.empty:
        sys.exit("No filled-in rows yet (add the scale readings).")
    df["std_read"]  = reads.std(axis=1, ddof=0)
    df["range_read"] = reads.max(axis=1) - reads.min(axis=1)
    df["err_g"] = df["mean_read"] - df["true_g"]
    df["abs_%"] = (df["err_g"].abs() / df["true_g"] * 100)

    n = len(df)
    print(f"\n==== WEIGHT CALIBRATION  ({n} items, {len(read_cols)} reads each) ====")
    print("\n--- RAW (before correction) ---")
    print(f"mean abs error : {df['err_g'].abs().mean():.1f} g   ({df['abs_%'].mean():.2f}%)")
    print(f"bias (signed)  : {df['err_g'].mean():+.1f} g")
    if len(read_cols) > 1:
        print(f"repeatability  : mean spread {df['range_read'].mean():.1f} g (max {df['range_read'].max():.0f} g), "
              f"mean std {df['std_read'].mean():.1f} g")

    print(f"\n{name_col:<14}{'true':>7}{'mean':>7}{'err_g':>7}{'abs_%':>7}{'spread':>8}")
    for _, r in df.iterrows():
        print(f"{str(r[name_col])[:13]:<14}{r['true_g']:>7.0f}{r['mean_read']:>7.1f}"
              f"{r['err_g']:>+7.1f}{r['abs_%']:>7.2f}{r['range_read']:>8.1f}")

    # ---- fit software correction and compare two models ----
    x = df["mean_read"].to_numpy(float)
    y = df["true_g"].to_numpy(float)
    if n >= 2:
        def report(name, pred, params):
            res = pred - y
            mae = float(np.abs(res).mean()); mx = float(np.abs(res).max())
            wpe = float((np.abs(res) / y * 100).mean())            # weight %-error, matches the calorie test
            print(f"\n--- {name} ---")
            print(f"  {params}")
            print(f"  residual: mean abs {mae:.1f} g ({wpe:.1f}%),  max {mx:.1f} g")
            print(f"  {'per-item err(g):':<18}" + " ".join(f"{v:+.0f}" for v in res))
            return mae

        # 2-parameter line: corrected = a + b*raw
        b, a = np.polyfit(x, y, 1)
        mae_lin = report(f"MODEL A  corrected = a + b*raw", a + b * x, f"a = {a:+.3f}   b = {b:.5f}")

        # through-origin (pure span/gain, intercept forced to 0): corrected = k*raw
        k = float((x * y).sum() / (x * x).sum())
        mae_org = report(f"MODEL B  corrected = k*raw  (no intercept, physical span error)",
                         k * x, f"k = {k:.5f}")

        best = "B (through-origin)" if mae_org <= mae_lin + 0.5 else "A (with intercept)"
        print(f"\n=> RECOMMENDED: Model {best}. "
              + ("Through-origin is simpler and won't over-read light items."
                 if best.startswith("B") else "The intercept genuinely improves the fit here."))
        print("   Paste the chosen coefficients into the app decode (CAL_A/CAL_B, or set CAL_A=0 for Model B).")
    else:
        print("\n(add at least 2 different weights to fit a correction line)")


if __name__ == "__main__":
    main()
