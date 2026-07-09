"""
Render the fp16-vs-fp32 bit-layout illustration for the book/report:
  proj-book/figures/fp_bits.png
Shows sign / exponent / mantissa fields for both formats, the 10-vs-23 mantissa bits,
and the "room to grow" point: a product of two 10-bit mantissas needs ~20 bits, which
fits in fp32's 23 but not fp16's 10.
Run:  py -3 scripts/eval/build_fp_bits_fig.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "proj-book", "figures"))
SIGN, EXP, MANT = "#C0392B", "#1F6FEB", "#2E7D32"
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})

fig, ax = plt.subplots(figsize=(9.2, 3.6))
ax.set_xlim(0, 32); ax.set_ylim(0, 4.2); ax.axis("off")


def row(y, sign, expb, mantb, label):
    x = 0
    for n, col, name in ((sign, SIGN, "sign"), (expb, EXP, f"exponent ({expb})"),
                         (mantb, MANT, f"mantissa ({mantb})")):
        ax.add_patch(Rectangle((x, y), n, 0.8, facecolor=col, edgecolor="white", lw=1.2))
        # per-bit ticks
        for i in range(1, n):
            ax.plot([x + i, x + i], [y, y + 0.8], color="white", lw=0.4, alpha=.6)
        ax.text(x + n / 2, y + 0.4, name, ha="center", va="center", color="white",
                fontsize=9, fontweight="bold")
        x += n
    ax.text(-0.4, y + 0.4, label, ha="right", va="center", fontsize=11, fontweight="bold")
    ax.text(x + 0.3, y + 0.4, f"= {x} bits", ha="left", va="center", fontsize=10, color="#444")


row(2.7, 1, 8, 23, "fp32")
row(1.3, 1, 5, 10, "fp16")

# "room to grow" annotation: 20-bit product bracket over the fp32 mantissa region
mx0 = 1 + 8            # start of fp32 mantissa
ax.annotate("", xy=(mx0, 3.62), xytext=(mx0 + 20, 3.62),
            arrowprops=dict(arrowstyle="<->", color="#6a1b9a", lw=1.6))
ax.text(mx0 + 10, 3.78, "a 10×10-bit product needs ~20 bits — fits here (fp32), not in fp16's 10",
        ha="center", va="bottom", fontsize=9, color="#6a1b9a")

ax.text(16, 0.35, "fp16 mantissa ≈ 3–4 decimal digits   ·   fp32 mantissa ≈ 7 decimal digits   "
                  "·   value = sign × mantissa × 2^exponent",
        ha="center", va="center", fontsize=9.5, color="#333")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fp_bits.png"), dpi=200, bbox_inches="tight")
print("wrote", os.path.join(OUT, "fp_bits.png"))
