"""
Illustrate the background-class OOD separation for the book:
  proj-book/figures/ood_separation.png
Histogram of the Israeli model's p_background (i_p_background) split by the true domain,
straight from datasets/arbiter_dataset.csv. Shows the near-linear split (~0.03 on genuine
Israeli food vs ~0.50 on global food) that makes p_bg the arbiter's top feature.
Run:  py -3 scripts/eval/build_ood_fig.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV = os.path.join(ROOT, "datasets", "arbiter_dataset.csv")
OUT = os.path.join(ROOT, "proj-book", "figures", "ood_separation.png")
GREEN, RED = "#2E7D32", "#C0392B"
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})

df = pd.read_csv(CSV)
col = "i_p_background" if "i_p_background" in df.columns else None
assert col, "arbiter_dataset.csv has no i_p_background column"
isr = df[df["domain"] == "israeli"][col].to_numpy()
glb = df[df["domain"] == "global"][col].to_numpy()

fig, ax = plt.subplots(figsize=(8.4, 3.9))
bins = np.linspace(0, 1, 41)
ax.hist(isr, bins=bins, density=True, color=GREEN, alpha=.75,
        label=f"genuine Israeli food  (median p$_{{bg}}$ = {np.median(isr):.2f})")
ax.hist(glb, bins=bins, density=True, color=RED, alpha=.55,
        label=f"global food  (median p$_{{bg}}$ = {np.median(glb):.2f})")
ax.set_xlabel("Israeli model's  p$_{bg}$  =  P(“this is not one of my dishes”)")
ax.set_ylabel("relative frequency")
ax.set_title("The background class separates the two domains almost linearly")
ax.legend(frameon=False, loc="upper center")
ax.set_xlim(0, 1)
ax.grid(axis="y", color="#ddd", lw=.6)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print(f"israeli n={len(isr)} median={np.median(isr):.3f} | global n={len(glb)} median={np.median(glb):.3f}")
print("wrote", OUT)
