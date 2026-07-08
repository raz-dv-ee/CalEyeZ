"""
Experiment E analysis — Cal.ai vs CalEyeZ: viewpoint repeatability + portion monotonicity.

Reads:
  scripts/eval/repeatability.csv   (P1: same food/grams, many angles, BOTH apps)
  scripts/eval/portion_sweep.csv   (P2: same food, increasing grams, BOTH apps)

Prints per-food statistics and writes figures for the book/site:
  proj-book/figures/repeatability.png   (per-photo kcal: Cal.ai scatter vs CalEyeZ flat vs truth)
  proj-book/figures/portion_sweep.png   (reported kcal vs actual grams, both apps vs truth diagonal)

Run:  py -3 scripts/eval/repeatability_analysis.py
Robust to partially-filled sheets: rows with no app reading are skipped; the CalEyeZ arm is
optional and simply omitted from a chart if its column is blank.
"""
import csv, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIGS = os.path.abspath(os.path.join(HERE, "..", "..", "proj-book", "figures"))
GREEN, RED, GREY = "#2E7D32", "#C0392B", "#8b949e"


def read_rows(path):
    """Read a CSV skipping the leading '#' comment lines; return list of dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def fnum(v):
    v = (v or "").strip()
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def spread_stats(name, k):
    mean = st.mean(k)
    sd = st.pstdev(k) if len(k) < 2 else st.stdev(k)
    cv = 100 * sd / mean if mean else 0
    print(f"    {name}: n={len(k)}  mean {mean:.0f}  SD {sd:.1f}  CV {cv:.0f}%  "
          f"range {min(k):.0f}-{max(k):.0f} ({max(k)-min(k):.0f} kcal spread)")
    return mean, sd, cv


# ---------------- P1: repeatability ----------------
def analyse_repeatability():
    rows = read_rows(os.path.join(HERE, "repeatability.csv"))
    foods = {}
    for r in rows:
        if r["food"].endswith("fill_me"):
            continue
        cal, ce = fnum(r.get("calai_kcal")), fnum(r.get("caleyez_kcal"))
        if cal is None and ce is None:
            continue
        w, per100 = fnum(r["true_weight_g"]), fnum(r["gt_per100_kcal"])
        d = foods.setdefault(r["food"], {"calai": [], "caleyez": [],
                                         "gt": (per100 * w / 100) if (w and per100) else None})
        if cal is not None:
            d["calai"].append(cal)
        if ce is not None:
            d["caleyez"].append(ce)

    if not foods:
        print("[repeatability] no filled rows yet — skipping.")
        return

    print("\n=== P1 · Viewpoint repeatability (same food, same grams, camera varies) ===")
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    first = True
    for i, (food, d) in enumerate(foods.items()):
        gt = d["gt"]
        print(f"  {food}" + (f" (truth {gt:.0f} kcal)" if gt else ""))
        if gt:
            ax.hlines(gt, i - .33, i + .33, color=GREY, lw=2, linestyles="--",
                      zorder=3, label="ground truth" if first else None)
        if d["calai"]:
            spread_stats("Cal.ai ", d["calai"])
            ax.scatter([i] * len(d["calai"]), d["calai"], s=62, color=RED, alpha=.85,
                       zorder=5, label="Cal.ai (photo estimate)" if first else None)
        if d["caleyez"]:
            spread_stats("CalEyeZ", d["caleyez"])
            ax.scatter([i] * len(d["caleyez"]), d["caleyez"], s=62, color=GREEN, marker="D",
                       alpha=.85, zorder=6, label="CalEyeZ (scale-measured)" if first else None)
        first = False

    ax.set_xticks(range(len(foods)))
    ax.set_xticklabels([f.replace("_", " ") for f in foods])
    ax.set_ylabel("reported calories (kcal)")
    ax.set_title("Same food, same weight — only the camera moves")
    ax.legend(frameon=False); ax.grid(axis="y", color="#ddd", lw=.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "repeatability.png"), dpi=200)
    plt.close(fig)
    print(f"  -> wrote {os.path.join(FIGS, 'repeatability.png')}")


# ---------------- P2: portion monotonicity ----------------
def analyse_portion():
    rows = read_rows(os.path.join(HERE, "portion_sweep.csv"))
    foods = {}
    for r in rows:
        if r["food"].endswith("fill_me"):
            continue
        w = fnum(r.get("true_weight_g"))
        if w is None:
            continue
        d = foods.setdefault(r["food"], {"w": [], "calai": [], "caleyez": [],
                                         "per100": fnum(r["gt_per100_kcal"])})
        d["w"].append(w)
        d["calai"].append(fnum(r.get("calai_kcal")))
        d["caleyez"].append(fnum(r.get("caleyez_kcal")))

    if not foods:
        print("\n[portion] no filled rows yet — skipping.")
        return

    print("\n=== P2 · Portion monotonicity (weight grows; does the number follow?) ===")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    firsttruth = True
    for food, d in foods.items():
        order = sorted(range(len(d["w"])), key=lambda j: d["w"][j])
        ws = [d["w"][j] for j in order]
        for key, colour, marker, lbl in (("calai", RED, "s", "Cal.ai"),
                                         ("caleyez", GREEN, "D", "CalEyeZ")):
            pts = [(d["w"][j], d[key][j]) for j in order if d[key][j] is not None]
            if len(pts) >= 2:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker + "-", color=colour,
                        label=f"{lbl} — {food.replace('_',' ')}")
                print(f"  {food} · {lbl}: {min(xs):.0f}->{max(xs):.0f} g ({max(xs)-min(xs):.0f} g) "
                      f"moved it {max(ys)-min(ys):.0f} kcal ({min(ys):.0f}-{max(ys):.0f})")
        if d["per100"]:
            gx = [0, max(ws)]
            ax.plot(gx, [d["per100"] * x / 100 for x in gx], "--", color=GREY,
                    label="ground truth" if firsttruth else None)
            firsttruth = False
    ax.set_xlabel("actual weight on the plate (g)")
    ax.set_ylabel("reported calories (kcal)")
    ax.set_title("Portion sweep — Cal.ai (guess) stays flat, CalEyeZ (measure) tracks the grams")
    ax.legend(frameon=False); ax.grid(color="#ddd", lw=.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "portion_sweep.png"), dpi=200)
    plt.close(fig)
    print(f"  -> wrote {os.path.join(FIGS, 'portion_sweep.png')}")


if __name__ == "__main__":
    analyse_repeatability()
    analyse_portion()
    print("\nDone. Re-run any time as you add rows; blank readings are ignored.")
