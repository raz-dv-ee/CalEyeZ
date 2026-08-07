"""
Catastrophic forgetting, measured: proj-book/figures/forgetting.png

Every number here is parsed straight out of the fine-tune logs in the repo root
(ft_hummus*.log, ft_global2.log), from the lines the regression gate prints:

    ORIGINAL  test top-1: 92.26%
    FINETUNED test top-1: 49.89%   (delta -42.38 pts)

Five completed fine-tunes, five refusals. Run:  py -3 scripts/eval/build_forgetting_fig.py
"""
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "proj-book", "figures", "forgetting.png")
GREEN, RED, GREY = "#2E7D32", "#C0392B", "#9aa4b2"
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})

# (log file, short label describing the recipe that was tried)
RUNS = [
    ("ft_hummus.log",  "Israeli\nFT-1\nfull unfreeze\n15 ep, lr 1e-4"),
    ("ft_hummus2.log", "Israeli\nFT-2\nfreeze 10\n25 ep, lr 1e-4"),
    ("ft_hummus3.log", "Israeli\nFT-3\nfull unfreeze\n20 ep, lr 1e-4"),
    ("ft_hummus4.log", "Israeli\nFT-4\nfull unfreeze\n15 ep, lr 5e-5"),
    ("ft_global2.log", "Global\nFT-5\nfreeze 10\n6 ep, lr 1e-4"),
]

PAT_BEFORE = re.compile(rb"ORIGINAL\s+test top-1:\s*([\d.]+)%")
PAT_AFTER = re.compile(rb"FINETUNED\s+test top-1:\s*([\d.]+)%")

labels, before, after = [], [], []
for fname, label in RUNS:
    blob = open(os.path.join(ROOT, fname), "rb").read()
    b, a = PAT_BEFORE.search(blob), PAT_AFTER.search(blob)
    assert b and a, f"no regression-gate verdict in {fname}"
    labels.append(label)
    before.append(float(b.group(1)))
    after.append(float(a.group(1)))

fig, ax = plt.subplots(figsize=(8.4, 3.6))
x = range(len(labels))
w = 0.38
ax.bar([i - w / 2 for i in x], before, w, color=GREEN, alpha=.85,
       label="clean test top-1 before the fine-tune")
ax.bar([i + w / 2 for i in x], after, w, color=RED, alpha=.85,
       label="clean test top-1 after it")

for i, (b, a) in enumerate(zip(before, after)):
    ax.text(i - w / 2, b + 1.5, f"{b:.2f}", ha="center", fontsize=8.5, color=GREEN)
    ax.text(i + w / 2, a + 1.5, f"{a:.2f}", ha="center", fontsize=8.5, color=RED)
    ax.annotate(f"-{b - a:.1f} pts", xy=(i, max(b, a) + 9), ha="center",
                fontsize=8.5, color="#444")

ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("top-1 on the original clean test split (%)")
ax.set_ylim(0, 128)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_title("Every fine-tune learned the new photos and destroyed the old classes")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=2,
          fontsize=9, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="#e5e7eb", lw=.8)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print("wrote", OUT)
for l, b, a in zip(labels, before, after):
    print(l.replace("\n", " "), b, "->", a)
