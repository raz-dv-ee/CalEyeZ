"""
CalEyeZ · Photo Tester  —  offline evidence-card generator for field photos.
============================================================================
WHY THIS EXISTS
    The live demo reads a webcam. For the robustness study you shoot food in the
    field with a PHONE, then evaluate at the desk on the 3060 Ti. This script feeds
    those phone photos through the *byte-for-byte identical* inference path as the
    live demo (it imports the demo's own predict()), so nobody can claim the test
    pipeline differed from the deployed one.

WHAT IT PRODUCES, per image, in one go (no manual screenshotting):
    1. <stem>__card.png   — a single evidence card: the image + both models' exact
                            top-5 softmax + the arbiter's P(israeli)/logit/SHAP
                            "why" + every one of the 20 features + the decision.
    2. eval_results.csv   — one row per image, every number flattened (append mode).
    3. <stem>__features.json — the full dump (top-5 lists, 20 features, SHAP reasons,
                            flags, EXIF capture-time) for that image.

EXIF capture-time is pulled from the ORIGINAL file before OpenCV strips it — that
timestamp is your "this photo was taken AFTER the model was trained" novelty proof.

USAGE
    python scripts/eval/photo_tester.py  path/to/photo.jpg          # one image
    python scripts/eval/photo_tester.py  path/to/folder/            # whole folder
    python scripts/eval/photo_tester.py                             # GUI file-picker
    optional:  --out results_dir   --show   (open each card window as it's made)

Deps: the same env as the demo (ultralytics torch opencv-python pillow xgboost
      numpy pandas matplotlib).
"""
from __future__ import annotations
import os
import sys
import csv
import json
import glob
import time
import argparse
from datetime import datetime

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Import the DEMO module so inference is guaranteed identical to the live system.
# Importing it loads both YOLO models + the arbiter once (the BLE thread is NOT
# started — that only happens under the demo's __main__ guard).
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.abspath(os.path.join(HERE, "..", "demo"))
sys.path.insert(0, DEMO_DIR)
try:
    import caleyez_demo as cz   # noqa: E402  (must come after sys.path insert)
except Exception as e:
    print(f"[FATAL] could not import the demo module from {DEMO_DIR}\n        {e}")
    sys.exit(1)

if cz.model_g is None or cz.model_i is None or cz.arbiter is None:
    print("[FATAL] models/arbiter did not load (check the weight paths in caleyez_demo.py).")
    sys.exit(1)

# theme (matches the project)
BG, PANEL, INK, MUT = "#0d1117", "#161b22", "#e6edf3", "#8b949e"
ACCENT, TEAL, PURPLE, AMBER, RED, GREEN = "#58a6ff", "#39d0c4", "#bc8cff", "#d29922", "#f85149", "#3fb950"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def exif_datetime(path: str) -> str:
    """Capture time from the photo's EXIF (DateTimeOriginal). '' if unavailable."""
    try:
        from PIL import Image, ExifTags
        img = Image.open(path)
        ex = img.getexif()
        if not ex:
            return ""
        tagmap = {ExifTags.TAGS.get(k, k): v for k, v in ex.items()}
        return str(tagmap.get("DateTimeOriginal") or tagmap.get("DateTime") or "")
    except Exception:
        return ""


def load_image_bgr(path: str):
    """Load an image as a BGR ndarray, honouring the EXIF orientation flag.

    cv2.imread() ignores EXIF orientation, so phone photos taken in portrait come in sideways and the
    center-ROI then crops the wrong region -> the classic ~10% 'field collapse' that looks like a model
    failure but is really a capture bug (see handoff notes TODO #1). We load via PIL and apply
    ImageOps.exif_transpose() so the pixels are upright BEFORE inference, exactly as a human sees them.
    Falls back to cv2.imread for anything PIL cannot open. Returns None if the image is unreadable.
    """
    try:
        from PIL import Image, ImageOps
        im = Image.open(path)
        im = ImageOps.exif_transpose(im)          # bake the rotation into the pixels
        rgb = np.array(im.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return cv2.imread(path)                    # last resort (e.g. exotic formats); EXIF not applied


def load_truth(path: str) -> dict:
    """Read a ground-truth CSV (columns: image,true_food[,...]) into {basename: true_food}.

    'image' may be a filename or a full path; we key on the basename so it matches regardless of how the
    photo was passed. Empty/missing file -> empty dict (scoring columns then stay blank).
    """
    truth = {}
    if not path:
        return truth
    if not os.path.exists(path):
        print(f"[WARN] --truth file not found: {path} (scoring skipped)")
        return truth
    with open(path, newline="", encoding="utf-8-sig") as tf:
        for r in csv.DictReader(tf):
            key = (r.get("image") or r.get("file") or "").strip()
            food = (r.get("true_food") or r.get("truth") or r.get("label") or "").strip()
            if key and food:
                truth[os.path.basename(key).lower()] = food.lower()
    print(f"[SYSTEM] loaded {len(truth)} ground-truth label(s) from {path}")
    return truth


def _barh(ax, pairs, color, title):
    """Horizontal bar chart of (label, value) pairs, biggest on top."""
    pairs = pairs[:5]
    labels = [p[0].replace("_", " ") for p in pairs][::-1]
    vals   = [float(p[1]) for p in pairs][::-1]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=color, height=.6)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8, color=INK)
    ax.set_xlim(0, 1)
    for yi, v in zip(y, vals):
        ax.text(min(v + .02, .98), yi, f"{v*100:.1f}%", va="center", fontsize=7.5, color=MUT)
    ax.set_title(title, fontsize=9, color=MUT, loc="left")
    ax.tick_params(axis="x", labelsize=7, colors=MUT)
    for s in ax.spines.values():
        s.set_color("#30363d")
    ax.set_facecolor(PANEL)


def _reasons(ax, reasons, feats):
    """SHAP-style diverging bars: right(+)=push to Israeli, left(-)=push to Global."""
    rows = (reasons or [])[:5][::-1]
    if not rows:
        ax.axis("off"); return
    y = np.arange(len(rows))
    vals = [r[1] for r in rows]
    names = [f"{r[0]} ({feats.get(r[0],0):.2f})" for r in rows]
    colors = [TEAL if v > 0 else ACCENT for v in vals]
    ax.barh(y, vals, color=colors, height=.6)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.5, color=INK)
    ax.axvline(0, color=MUT, lw=.8, ls=(0, (2, 2)))
    ax.set_title("ARBITER 'WHY'   ◀ Global   |   Israeli ▶", fontsize=9, color=MUT, loc="left")
    ax.tick_params(axis="x", labelsize=7, colors=MUT)
    for s in ax.spines.values():
        s.set_color("#30363d")
    ax.set_facecolor(PANEL)


def render_card(path: str, bgr: np.ndarray, d: dict, exif_dt: str, show: bool, out_png: str):
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    g, i = d["global"], d["israeli"]
    f = d["features"]
    route = "ISRAELI" if d["route_israeli"] else "GLOBAL"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    roi = cv2.cvtColor(d["roi"], cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(13, 8), facecolor=BG)
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1.15, 1, 1],
                           hspace=.45, wspace=.32, left=.05, right=.97, top=.93, bottom=.05)

    # --- input image + ROI ---
    ax_img = fig.add_subplot(gs[0, 0:2]); ax_img.imshow(rgb); ax_img.axis("off")
    ax_img.set_title(os.path.basename(path), fontsize=10, color=INK, loc="left")
    ax_roi = fig.add_subplot(gs[0, 2]); ax_roi.imshow(roi); ax_roi.axis("off")
    ax_roi.set_title("ROI fed to models", fontsize=8.5, color=MUT, loc="left")

    # --- top-5 softmax for each expert ---
    _barh(fig.add_subplot(gs[1, 0]), g["top5"], ACCENT, f"GLOBAL expert (132)  ·  top: {g['label'].replace('_',' ')}")
    _barh(fig.add_subplot(gs[1, 1]), i["top5"], TEAL,  f"ISRAELI expert (13)  ·  top: {i['label'].replace('_',' ')}")

    # --- decision panel ---
    axd = fig.add_subplot(gs[1, 2]); axd.axis("off")
    flags = []
    if d["confident"]:           flags.append(("CONFIDENT", GREEN))
    if d["arbiter_unsure"]:      flags.append(("ARBITER UNDECIDED", AMBER))
    if d["both_unsure"]:         flags.append(("BOTH UNSURE (OOD)", AMBER))
    if d["israeli_abstain"]:     flags.append(("ISRAELI ABSTAIN (bg)", AMBER))
    if not flags:                flags.append(("LOW CONFIDENCE → Gemini", RED))
    decision = (f"FINAL : {d['label'].replace('_',' ')}\n"
                f"route : {route}   conf {d['conf']*100:.1f}%\n\n"
                f"P(israeli) = {d['p_israeli']:.3f}\n"
                f"logit = {d['logit']:+.3f}\n"
                f"sigmoid({d['logit']:+.2f}) {'≥' if d['p_israeli']>=.5 else '<'} 0.50")
    axd.text(0, 1, decision, va="top", ha="left", fontsize=11, color=INK,
             family="monospace", transform=axd.transAxes)
    for k, (txt, col) in enumerate(flags):
        axd.text(0, .18 - k*.11, "● " + txt, va="top", ha="left", fontsize=9, color=col,
                 family="monospace", transform=axd.transAxes)

    # --- SHAP reasons ---
    _reasons(fig.add_subplot(gs[2, 0:2]), d.get("reasons", []), f)

    # --- full 20-feature vector ---
    axf = fig.add_subplot(gs[2, 2]); axf.axis("off")
    lines = ["20-FEATURE VECTOR"]
    for k in range(1, 6):
        lines.append(f"g_conf{k}={f.get(f'g_conf{k}',0):.2f}  i_conf{k}={f.get(f'i_conf{k}',0):.2f}")
    lines += [
        f"g_ent={f.get('g_entropy',0):.2f}  g_mar={f.get('g_margin',0):.2f}",
        f"i_ent={f.get('i_entropy',0):.2f}  i_mar={f.get('i_margin',0):.2f}",
        f"i_p_bg={f.get('i_p_background',0):.2f}",
        f"conf_gap={f.get('conf_gap',0):+.2f}  ratio={f.get('conf_ratio',0):.2f}",
        f"ent_gap={f.get('entropy_gap',0):+.2f}  mar_gap={f.get('margin_gap',0):+.2f}",
        f"both_unsure={int(f.get('both_unsure',0))}",
    ]
    if exif_dt:
        lines += ["", f"EXIF taken: {exif_dt}"]
    axf.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=8.2, color=MUT,
             family="monospace", transform=axf.transAxes)

    fig.suptitle("CalEyeZ · evidence card", color=MUT, fontsize=11, x=.05, ha="left")
    fig.savefig(out_png, facecolor=BG, dpi=130)
    if show:
        plt.show()
    plt.close(fig)


def folder_label(path: str, root: str) -> str:
    """Ground-truth label inferred from the image's parent folder (ImageFolder layout).

    For 'real images test/hamburger/x.jpg' this returns 'hamburger'. Returns '' for images that sit
    directly in the root folder (flat layout, no class subfolder) so a mixed tree degrades gracefully.
    """
    if not root:
        return ""
    parent = os.path.dirname(os.path.abspath(path))
    if os.path.abspath(parent) == os.path.abspath(root):
        return ""                              # image is loose in the root -> no class folder
    return os.path.basename(parent).lower()


def process(path: str, out_dir: str, show: bool, writer, csv_file, truth: dict, root: str = "",
            use_folder: bool = True):
    bgr = load_image_bgr(path)                # EXIF-corrected load (upright pixels)
    if bgr is None:
        print(f"[skip] cannot read {path}")
        return
    # Latency = wall-clock time for the FULL edge decision (both YOLO forward passes + the arbiter),
    # i.e. exactly the input->answer cost the live system pays per Analyze. Measured with a monotonic
    # clock around predict() only, so image I/O and card rendering are excluded.
    t0 = time.perf_counter()
    d = cz.predict(bgr)                       # IDENTICAL to the live demo's edge decision
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    exif_dt = exif_datetime(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    out_png = os.path.join(out_dir, stem + "__card.png")
    render_card(path, bgr, d, exif_dt, show, out_png)

    # ground-truth scoring: explicit --truth CSV wins, else the parent-folder name (ImageFolder layout),
    # else blank (image not scored). Lets you simply drop photos into class subfolders.
    true_food = truth.get(os.path.basename(path).lower(), "")
    if not true_food and use_folder:
        true_food = folder_label(path, root)
    correct = "" if not true_food else (d["label"].lower() == true_food)

    g, i, f = d["global"], d["israeli"], d["features"]
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image": os.path.basename(path), "exif_datetime": exif_dt,
        "true_food": true_food, "correct": correct, "latency_ms": latency_ms,
        "final_label": d["label"], "route": "israeli" if d["route_israeli"] else "global",
        "p_israeli": round(d["p_israeli"], 4), "logit": round(d["logit"], 4),
        "confident": d["confident"], "both_unsure": d["both_unsure"],
        "arbiter_unsure": d["arbiter_unsure"], "israeli_abstain": d["israeli_abstain"],
        "g_top1": g["label"], "g_top1_conf": round(g["conf"][0], 4),
        "i_top1": i["label"], "i_top1_conf": round(i["conf"][0], 4),
    }
    # i_p_background and the rest of the 20 features come from cz.FEATS (which already includes
    # i_p_background) -> written exactly once, fixing the old duplicate-column header.
    for k in cz.FEATS:
        row[k] = round(float(f.get(k, 0)), 5)
    row["g_top5"] = json.dumps(g["top5"]); row["i_top5"] = json.dumps(i["top5"])
    row["reasons"] = json.dumps(d.get("reasons", []))

    writer.writerow(row); csv_file.flush()

    with open(os.path.join(out_dir, stem + "__features.json"), "w", encoding="utf-8") as jf:
        json.dump(row, jf, indent=2, ensure_ascii=False)

    verdict = "" if correct == "" else ("  ✓" if correct else f"  ✗ (truth: {true_food})")
    print(f"[ok] {os.path.basename(path):<28} -> {d['label']:<18} "
          f"({route_str(d)})  {latency_ms:>6.1f} ms{verdict}")


def route_str(d):
    return ("ISRAELI" if d["route_israeli"] else "GLOBAL") + (
        "" if d["confident"] else " → Gemini")


def gather(path: str) -> list[str]:
    if os.path.isdir(path):
        out = []
        for ext in IMG_EXTS:
            out += glob.glob(os.path.join(path, "**", "*" + ext), recursive=True)
        return sorted(out)
    return [path] if path.lower().endswith(IMG_EXTS) else []


def pick_with_dialog() -> list[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        files = filedialog.askopenfilenames(
            title="Pick food photo(s) to test",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
        root.destroy()
        return list(files)
    except Exception as e:
        print(f"[FATAL] no path given and GUI picker failed: {e}")
        return []


def already_processed(csv_path: str) -> set:
    """Basenames already present in the CSV's 'image' column (for idempotent re-runs)."""
    done = set()
    if not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0):
        return done
    try:
        with open(csv_path, newline="", encoding="utf-8") as cf:
            reader = csv.reader(cf)
            header = next(reader, [])
            idx = header.index("image") if "image" in header else 1
            for r in reader:
                if len(r) > idx:
                    done.add(os.path.basename(r[idx]).lower())
    except Exception as e:
        print(f"[WARN] could not read existing CSV ({e}); not skipping anything.")
    return done


def main():
    ap = argparse.ArgumentParser(description="CalEyeZ offline photo evidence-card generator")
    ap.add_argument("path", nargs="?", help="image file or folder (omit for a GUI picker)")
    ap.add_argument("--out", default=os.path.join(cz.ROOT, "field_eval"), help="output directory")
    ap.add_argument("--show", action="store_true", help="open each card window as it is made")
    ap.add_argument("--truth", default="", help="ground-truth CSV (image,true_food) -> adds correct/accuracy")
    ap.add_argument("--no-folder-labels", action="store_true",
                    help="do NOT use parent-folder names as ground truth (on by default for folder input)")
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild eval_results.csv from scratch instead of appending (no skip)")
    ap.add_argument("--reprocess", action="store_true",
                    help="process images even if already in the CSV (still appends)")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="in the summary, score only predictions with chosen-expert conf >= this "
                         "(simulates the confidence gate; below it would go to Gemini live)")
    args = ap.parse_args()

    if not args.show:
        import matplotlib
        matplotlib.use("Agg")             # headless: just save PNGs, no windows

    paths = gather(args.path) if args.path else pick_with_dialog()
    if not paths:
        print("No images to process."); return

    # ImageFolder-style ground truth: when the input is a directory, each image's parent folder is its
    # true class (real images test/hamburger/x.jpg -> 'hamburger'). Disable with --no-folder-labels.
    root = args.path if (args.path and os.path.isdir(args.path)) else ""
    use_folder = bool(root) and not args.no_folder_labels
    if use_folder:
        print(f"[SYSTEM] using parent-folder names as ground truth under: {root}")

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "eval_results.csv")
    truth = load_truth(args.truth)

    # deterministic column order (so appends across runs stay aligned). NOTE: i_p_background is NOT listed
    # explicitly here — it lives inside cz.FEATS, so listing it again is what caused the duplicate column.
    fieldnames = (["timestamp", "image", "exif_datetime", "true_food", "correct", "latency_ms",
                   "final_label", "route", "p_israeli", "logit", "confident", "both_unsure",
                   "arbiter_unsure", "israeli_abstain", "g_top1", "g_top1_conf", "i_top1", "i_top1_conf"]
                  + list(cz.FEATS) + ["g_top5", "i_top5", "reasons"])

    # skip images already in the CSV (idempotent re-runs) unless asked to reprocess/overwrite
    done = set() if (args.overwrite or args.reprocess) else already_processed(csv_path)
    if done:
        skipped = [p for p in paths if os.path.basename(p).lower() in done]
        paths = [p for p in paths if os.path.basename(p).lower() not in done]
        print(f"[SYSTEM] skipping {len(skipped)} image(s) already in CSV "
              f"(use --reprocess to force, --overwrite to rebuild).")
    if not paths:
        print("Nothing new to process."); return

    mode = "w" if args.overwrite else "a"
    write_header = args.overwrite or not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    csv_file = open(csv_path, mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    print(f"[SYSTEM] processing {len(paths)} image(s) -> {args.out}")
    for p in paths:
        try:
            process(p, args.out, args.show, writer, csv_file, truth, root, use_folder)
        except Exception as e:
            print(f"[err] {p}: {e}")
    csv_file.close()

    # latency + accuracy summary across the whole CSV (cheap second pass over the file)
    _summarise(csv_path, truth, args.min_conf)
    print(f"\nDone. Cards + JSON in {args.out}\\ , summary in {csv_path}")


def wilson_ci(hits: int, n: int, z: float = 1.96):
    """Wilson score 95% confidence interval for a binomial proportion (lo, hi) in percent.

    WHY Wilson and not the textbook normal/Wald interval (p ± z*sqrt(p(1-p)/n)): Wald assumes the
    sampling distribution is normal, which only holds when both n*p and n*(1-p) are large (>~5). For a
    GOOD model the error count is tiny (e.g. 3 misses in 30), so Wald is wrong here and can even spill
    past 0% or 100%. Wilson is the standard small-sample / extreme-proportion interval and never does
    that. (This is also why '30 samples -> CLT -> normal' is the wrong tool for an accuracy proportion.)
    """
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5)) / denom
    return (max(0.0, centre - half) * 100, min(1.0, centre + half) * 100)


def _summarise(csv_path: str, truth: dict, min_conf: float = 0.0):
    """Print latency stats and accuracy with Wilson 95% CIs (overall + per-class) across the CSV.

    min_conf > 0 keeps only predictions whose chosen-expert top-1 confidence >= min_conf (everything
    below would defer to Gemini in the live system), so you can see how accuracy + CI move as the gate
    rises — a direct, honest picture of the confidence-gate / Gemini tradeoff.
    """
    try:
        with open(csv_path, newline="", encoding="utf-8") as cf:
            rows = list(csv.DictReader(cf))
    except Exception:
        return
    if not rows:
        return
    lats = sorted(float(r["latency_ms"]) for r in rows if r.get("latency_ms"))
    if lats:
        med = lats[len(lats)//2]
        p95 = lats[min(len(lats)-1, int(round(0.95*(len(lats)-1))))]
        print(f"\n[LATENCY] n={len(lats)}  median={med:.1f} ms  p95={p95:.1f} ms  "
              f"min={lats[0]:.1f}  max={lats[-1]:.1f}  (first call includes model warm-up)")

    scored = [r for r in rows if r.get("correct") in ("True", "False")]

    def _chosen_conf(r):
        try:
            return float(r["i_top1_conf"] if r.get("route") == "israeli" else r["g_top1_conf"])
        except Exception:
            return 0.0

    if min_conf > 0:
        kept = [r for r in scored if _chosen_conf(r) >= min_conf]
        print(f"\n[GATE] --min-conf {min_conf:.2f}: keeping {len(kept)}/{len(scored)} predictions "
              f"(the other {len(scored)-len(kept)} would defer to Gemini in the live system).")
        scored = kept

    if scored:
        hits = sum(1 for r in scored if r["correct"] == "True")
        lo, hi = wilson_ci(hits, len(scored))
        line = f"\n[ACCURACY] overall {hits}/{len(scored)} = {100*hits/len(scored):.1f}%   95% CI [{lo:.1f}%, {hi:.1f}%] (Wilson)"
        conf = [r for r in scored if r.get("confident") == "True"]
        if conf:
            chits = sum(1 for r in conf if r["correct"] == "True")
            clo, chi = wilson_ci(chits, len(conf))
            line += (f"\n           confident-only {chits}/{len(conf)} = {100*chits/len(conf):.1f}%"
                     f"   95% CI [{clo:.1f}%, {chi:.1f}%]")
        print(line)
        # per-class breakdown with Wilson CIs (sorted worst-first so weak classes jump out)
        classes = {}
        for r in scored:
            c = classes.setdefault(r["true_food"], [0, 0])
            c[1] += 1
            if r["correct"] == "True":
                c[0] += 1
        print("[PER-CLASS]   (CI is wide until n is large — pooled overall above is the solid number)")
        for cls, (h, n) in sorted(classes.items(), key=lambda kv: kv[1][0]/kv[1][1]):
            lo, hi = wilson_ci(h, n)
            flag = "  <-- n<30, directional only" if n < 30 else ""
            print(f"   {cls:<22} {h:>3}/{n:<3} = {100*h/n:5.1f}%   95% CI [{lo:5.1f}%, {hi:5.1f}%]{flag}")


if __name__ == "__main__":
    main()
