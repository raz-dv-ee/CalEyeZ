"""
Render the two App-vs-Market figures for the project book from the same numbers
shown in index.html (#tab-compare) and scripts/eval/calorie_comparison.xlsx.
  figures/compare_permeal.png  - per-meal absolute calorie error, three apps
  figures/compare_portion.png  - portion test: reported kcal vs actual weight
Regenerate:  py -3 scripts/eval/build_compare_figs.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "proj-book", "figures")
OUT = os.path.abspath(OUT)

GREEN, BLUE, RED, GREY = "#2E7D32", "#1F6FEB", "#C0392B", "#8b949e"
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans",
                     "axes.edgecolor": "#888", "axes.linewidth": 0.8})

# ---- Figure 1: per-meal absolute % error (8 shared meals) ----
foods = ["bell\npepper", "cucumber", "tuna\n(can)", "french\nfries",
         "hummus", "Domino's", "Pizza\nHut", "malawach"]
caleyez = [25, 31, 8, 135, 39, 42, 5, 8]
mfp     = [54, 54, 83, 126, 13, 24, 34, 1]
calai   = [56, 9, 29, 52, 89, 80, 55, 139]

x = np.arange(len(foods)); w = 0.27
fig, ax = plt.subplots(figsize=(8.4, 3.7))
ax.bar(x - w, caleyez, w, label=f"CalEyeZ (mean {np.mean(caleyez):.0f}%)", color=GREEN)
ax.bar(x,     mfp,     w, label=f"MyFitnessPal (mean {np.mean(mfp):.0f}%)", color=BLUE)
ax.bar(x + w, calai,   w, label=f"Cal.ai (mean {np.mean(calai):.0f}%)",     color=RED)
ax.set_ylabel("absolute calorie error (%)")
ax.set_xticks(x); ax.set_xticklabels(foods)
ax.set_title("Per-meal calorie error — lower is better (8 meals all three apps were tested on)")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
ax.set_ylim(0, 160); ax.grid(axis="y", color="#ddd", lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "compare_permeal.png"), dpi=200)
plt.close(fig)

# ---- Figure 2: portion test (canned tuna, two portions) ----
fig, ax = plt.subplots(figsize=(6.6, 4.0))
gx = np.array([0, 170])
ax.plot(gx, gx * 1.04, "--", color=GREY, lw=1.6,
        label="ground truth (label): 1.04 kcal/g")
ax.scatter([14, 162], [15, 182], s=90, color=GREEN, zorder=5,
           label="CalEyeZ (measured weight)")
ax.scatter([14, 162], [45, 120], s=90, color=RED, marker="s", zorder=5,
           label="Cal.ai (photo estimate)")
ax.annotate("3% err", (14, 15), (26, 6), color=GREEN, fontsize=9)
ax.annotate("209% err", (14, 45), (26, 58), color=RED, fontsize=9)
ax.annotate("8% err", (162, 182), (120, 190), color=GREEN, fontsize=9)
ax.annotate("29% err", (162, 120), (120, 104), color=RED, fontsize=9)
ax.set_xlabel("actual weight on the plate (g)")
ax.set_ylabel("reported calories (kcal)")
ax.set_title("Portion test — does the calorie number follow the real portion?")
ax.set_xlim(0, 175); ax.set_ylim(0, 205)
ax.legend(frameon=False, loc="upper left")
ax.grid(color="#ddd", lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "compare_portion.png"), dpi=200)
plt.close(fig)

print("saved compare_permeal.png and compare_portion.png to", OUT)
