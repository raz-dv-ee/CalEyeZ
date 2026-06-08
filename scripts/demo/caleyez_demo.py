"""
CalEyeZ live demo  ·  Global model + Israeli model + XGBoost arbiter + BLE scale + Gemini fallback.

Pipeline per capture:
    webcam frame (any resolution)
      -> fixed center ROI crop (user-guided, scale-consistent)
      -> lighting normalisation (gray-world white balance + CLAHE)
      -> Global model (132 cls, imgsz 320) and Israeli model (13 cls, imgsz 224), both via
         Ultralytics predict() so letterboxing makes the result resolution independent
      -> 19-feature vector -> XGBoost arbiter -> P(israeli) -> pick the expert
      -> confidence gate: if the chosen expert is weak OR both experts are unsure,
         fall back to the Gemini vision API (covers the "both models wrong" case)
      -> nutrition: local DB first, else USDA, scaled by the BLE weight in grams.

Design choices are documented inline and in the project doc (Live Demo tab in index.html).
Run:  python scripts/demo/caleyez_demo.py
Deps: ultralytics torch opencv-python pillow xgboost requests google-generativeai bleak matplotlib numpy pandas
"""
from __future__ import annotations

import asyncio
import csv
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

# ---- heavy / optional deps are imported defensively so the UI still opens if one is missing ----
try:
    import torch  # noqa: F401  (ultralytics needs it; import surfaces a clear error early)
    from ultralytics import YOLO
    import xgboost as xgb
    _ML_OK = True
except Exception as e:  # pragma: no cover
    print(f"[WARN] ML stack unavailable: {e}")
    _ML_OK = False

try:
    import requests
except Exception:
    requests = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from bleak import BleakScanner, BleakClient
    _BLE_OK = True
    _BLE_ERR = ""
except Exception as e:
    _BLE_OK = False
    _BLE_ERR = str(e)
    print(f"[WARN] bleak unavailable, BLE weight disabled: {e}\n"
          f"       install it in THIS python env:  pip install bleak\n"
          f"       (you can still demo by typing the weight in the box).")

# ==========================================================================================
# 0 · CONFIG
# ==========================================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
# Path root. When packaged by PyInstaller, model/arbiter data files are unpacked under the bundle dir
# (sys._MEIPASS for onefile; the _internal dir for onedir). In dev it is the repo root two levels up.
FROZEN = getattr(sys, "frozen", False)
EXE_DIR = os.path.dirname(sys.executable) if FROZEN else HERE   # folder the .exe sits in (for config/log files)
if FROZEN:
    ROOT = getattr(sys, "_MEIPASS", EXE_DIR)
else:
    ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

GLOBAL_W = os.path.join(ROOT, "runs", "general_model_flattened", "weights", "best.pt")
ISRAELI_W = os.path.join(ROOT, "runs", "israeli_food_yolo11l_v2", "weights", "best.pt")  # V2: 13 dishes + background
ARBITER_J = os.path.join(ROOT, "scripts", "arbiter", "arbiter_xgb.json")

GLOBAL_IMGSZ = 320            # matches how the global model was trained / the arbiter dataset
ISRAELI_IMGSZ = 224          # matches how the Israeli model was trained
ROI_FRAC = 0.60              # central square fraction used as the food region

# Confidence gate -> Gemini fallback. These target the "both models wrong" pool (~11% of images).
GATE_GLOBAL = 0.55           # global model is broad (132 cls); raised so borderline guesses auto-defer to Gemini
GATE_ISRAELI = 0.60          # Israeli model is narrow (13 cls): demand a higher top-1
BOTH_UNSURE = 0.50           # if BOTH experts' top-1 < this, treat as out-of-distribution -> Gemini
ARBITER_BAND = 0.15          # if |P(israeli) - 0.5| < this, the arbiter cannot decide which expert -> Gemini

# best-of-N analysis: run more frames on a GPU, fewer on a CPU / packaged edge build so ANALYZE stays
# responsive (each frame is 2 model forward passes; on CPU 6 frames would mean a ~10 s wait per click).
try:
    BEST_OF_N = 2 if (FROZEN or not (_ML_OK and torch.cuda.is_available())) else 6
except Exception:
    BEST_OF_N = 2

# Keys: env first, then a plain-text `caleyez_keys.txt` next to the program (USDA_API_KEY=... per line),
# then a dev fallback literal. The packaged .exe is FROZEN, so it NEVER uses the literal: a shipped exe
# carries no secret, the user sets an env var or drops a caleyez_keys.txt beside the .exe.
def _read_key(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if v:
        return v
    cfg = os.path.join(EXE_DIR, "caleyez_keys.txt")
    if os.path.exists(cfg):
        try:
            for line in open(cfg, encoding="utf-8"):
                if line.strip().startswith(name + "="):
                    return line.split("=", 1)[1].strip()
        except Exception:
            pass
    return ""
# dev-only fallback (ignored when frozen so the literal is never shipped in the exe)
_DEV_USDA = "" if FROZEN else "YOUR_USDA_API_KEY_HERE"
_DEV_GEM  = "" if FROZEN else "YOUR_GEMINI_API_KEY_HERE"
USDA_API_KEY = _read_key("USDA_API_KEY") or _DEV_USDA
GEMINI_API_KEY = _read_key("GEMINI_API_KEY") or _DEV_GEM

# Writable outputs go next to the .exe when frozen (the bundle dir is temporary/read-only).
WRITE_DIR = EXE_DIR if FROZEN else ROOT
LOG_FILE = os.path.join(WRITE_DIR, "nutrition_log.csv")
NOTIFY_UUID = "0000ffb2-0000-1000-8000-00805f9b34fb"   # SWAN scale notify characteristic
MAX_HISTORY = 100

# Data-tagging tool: every tagged shot becomes (1) a labelled image for fine-tuning, (2) a real-world
# verification row, and (3) the 512-D penultimate embedding ("DNA") of each model for an
# embeddings-router / OSR study. Layout is ImageFolder-compatible so the images are training-ready.
TAG_DIR = os.path.join(WRITE_DIR, "tagging_data")
TAG_IMG_DIR = os.path.join(TAG_DIR, "images")
TAG_EMB_DIR = os.path.join(TAG_DIR, "embeddings")
TAG_CSV = os.path.join(TAG_DIR, "tagging_log.csv")

# theme
BG, PANEL, CARD, INK, MUT = "#0d1117", "#161b22", "#1c2230", "#e6edf3", "#8b949e"
ACCENT, GREEN, AMBER, RED, TEAL, PURPLE = "#58a6ff", "#3fb950", "#d29922", "#f85149", "#39d0c4", "#bc8cff"

# 19 arbiter features, in the exact training order (scripts/arbiter/train_arbiter_xgb.py)
# 20 arbiter features, EXACT training order (scripts/arbiter/train_arbiter_xgb.py): note i_p_background
# sits right after i_margin, before the interaction terms.
FEATS = ([f"g_conf{k}" for k in range(1, 6)] + ["g_entropy", "g_margin"]
         + [f"i_conf{k}" for k in range(1, 6)] + ["i_entropy", "i_margin", "i_p_background"]
         + ["conf_gap", "conf_ratio", "entropy_gap", "margin_gap", "both_unsure"])

# Small local nutrition table (per 100 g) for the Israeli dishes the USDA database does not cover well.
LOCAL_DB = {
    "shakshuka": ("Shakshuka, eggs poached in tomato sauce", 85, 5.5, 4.0, 5.0),
    "sufganiyah": ("Sufganiyah, Israeli jelly donut", 350, 5.0, 40.0, 18.0),
    "jachnun": ("Jachnun, Yemenite rolled pastry", 380, 7.0, 42.0, 20.0),
    "malawach": ("Malawach, fried Yemenite flatbread", 330, 6.5, 45.0, 14.0),
    "sabich": ("Sabich, pita with eggplant and egg", 220, 7.0, 25.0, 10.0),
    "meorav_yerushalmi": ("Jerusalem mixed grill", 210, 18.0, 5.0, 12.0),
    "baklava": ("Baklava, nut and syrup pastry", 430, 6.0, 45.0, 25.0),
    "bourekas_cheese": ("Cheese bourekas", 360, 9.0, 30.0, 23.0),
    "falafel": ("Falafel balls", 333, 13.3, 31.8, 17.8),
    "hummus": ("Hummus, chickpea dip", 177, 7.9, 20.0, 8.6),
    "samosa": ("Samosa, fried savory pastry", 262, 5.0, 32.0, 13.0),
    "schnitzel": ("Chicken schnitzel", 290, 18.0, 16.0, 16.0),
    "shawarma": ("Shawarma, spiced rotisserie meat", 250, 17.0, 5.0, 18.0),
    # common global foods so the demo shows macros even with no API keys configured
    "bell_pepper": ("Bell pepper, raw", 26, 1.0, 6.0, 0.3),
    "tomato": ("Tomato, raw", 18, 0.9, 3.9, 0.2),
    "cucumber": ("Cucumber, raw", 15, 0.7, 3.6, 0.1),
    "apple": ("Apple, raw", 52, 0.3, 14.0, 0.2),
    "banana": ("Banana, raw", 89, 1.1, 23.0, 0.3),
    "orange": ("Orange, raw", 47, 0.9, 12.0, 0.1),
    "carrot": ("Carrot, raw", 41, 0.9, 10.0, 0.2),
    "potato": ("Potato, raw", 77, 2.0, 17.0, 0.1),
    "egg": ("Egg, whole", 143, 13.0, 1.1, 9.5),
    "pizza": ("Pizza, cheese", 266, 11.0, 33.0, 10.0),
    "hamburger": ("Hamburger", 295, 17.0, 24.0, 14.0),
    "french_fries": ("French fries", 312, 3.4, 41.0, 15.0),
    "rice": ("White rice, cooked", 130, 2.7, 28.0, 0.3),
    "white_rice": ("White rice, cooked", 130, 2.7, 28.0, 0.3),
    "bread": ("Bread, white", 265, 9.0, 49.0, 3.2),
    "steak": ("Beef steak, cooked", 271, 25.0, 0.0, 19.0),
    "sushi": ("Sushi", 150, 5.8, 30.0, 0.5),
    "chocolate_cake": ("Chocolate cake", 371, 5.0, 50.0, 16.0),
}

# ==========================================================================================
# 1 · MODELS
# ==========================================================================================
print("[SYSTEM] loading models...")
model_g = model_i = arbiter = None
if _ML_OK:
    try:
        model_g = YOLO(GLOBAL_W)
        model_i = YOLO(ISRAELI_W)
        arbiter = xgb.XGBClassifier()
        arbiter.load_model(ARBITER_J)
        print("[SYSTEM] models + arbiter ready.")
    except Exception as e:
        print(f"[SYSTEM ERROR] could not load models: {e}")

# Every class both models know (132 global + 13 Israeli) -> autocomplete for the tagging field.
ALL_CLASSES = []
if model_g is not None:
    ALL_CLASSES = sorted(set(model_g.names.values()) | set(model_i.names.values()))

if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini = genai.GenerativeModel("gemini-flash-latest")
    except Exception as e:
        gemini = None
        print(f"[WARN] Gemini unavailable: {e}")
else:
    gemini = None


# ==========================================================================================
# 2 · IMAGE ROBUSTNESS  (why: a live demo faces any webcam resolution and any room lighting)
# ==========================================================================================
def open_camera():
    """Open the webcam robustly across Windows machines.

    cv2.VideoCapture(0) defaults to the MSMF backend on Windows, which on many laptops fails with
    'CvCapture_MSMF::grabFrame ... can't grab frame. Error: -1072875772' (it opens but never delivers
    frames). DirectShow (CAP_DSHOW) is far more reliable for consumer webcams, so we try it first, then
    MSMF, then the plain default, across the first few device indices. Returns the first capture that
    actually yields a frame (a handle that merely 'opens' is not enough — MSMF lies about that).
    """
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for idx in range(3):              # device 0,1,2 (external cams / virtual cams shift the index)
        for be in backends:
            cap = cv2.VideoCapture(idx, be)
            if cap.isOpened():
                ok, _ = cap.read()    # actually pull one frame to confirm the device delivers
                if ok:
                    print(f"[CAMERA] opened index {idx} via backend {be}")
                    return cap
            cap.release()
    print("[CAMERA] WARNING: no working webcam found; preview will be blank.")
    return cv2.VideoCapture(0)        # last resort, keeps the app alive


def center_roi(bgr: np.ndarray) -> np.ndarray:
    """Fixed central square crop.

    Rationale: automatic plate detection (HoughCircles / contours) is brittle on arbitrary
    backgrounds, cluttered tables and changing light, and its failures are silent. A fixed,
    user-guided ROI (the on-screen "place food here" box) is deterministic, keeps the food at
    a consistent scale regardless of camera resolution, and removes that whole failure mode.
    """
    h, w = bgr.shape[:2]
    s = int(min(h, w) * ROI_FRAC)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return bgr[y0:y0 + s, x0:x0 + s].copy()


# NOTE on approaches tried and removed (kept here as a record, because both looked reasonable but
# measurably HURT and were cut after testing):
#   - GrabCut food segmentation: the models are robust to background (a pepper scores 99.97% on a
#     dark or wood background), and GrabCut crops were unreliable - a clean hummus at 96% dropped to
#     malawach at 92% after GrabCut. Removed; we feed the plain centre ROI.
#   - Test-time augmentation (averaging raw + CLAHE softmax): it shifted the confidence values away
#     from the distribution the arbiter was trained on and broke routing (confident Israeli dishes
#     were sent to the global model). Removed; features now come from a single predict(), exactly as
#     in generate_arbiter_dataset.py.


# ==========================================================================================
# 3 · INFERENCE  (features computed identically to generate_arbiter_dataset.py)
# ==========================================================================================
# Ultralytics models are NOT thread-safe: a predict()/embed() from the tag worker overlapping one from
# the analyze worker (or a re-entrant click) deadlocks the shared predictor. This lock serialises every
# model call so only one runs at a time, which is what made tagging hang after the first sample.
MODEL_LOCK = threading.Lock()


def expert_features(model, bgr: np.ndarray, imgsz: int) -> dict:
    """Top-5 conf, entropy, margin from one expert.

    Computed with a SINGLE predict() exactly as in generate_arbiter_dataset.py, because the arbiter is
    a precise function of these confidence values. Test-time averaging (raw + CLAHE) was tried and
    removed: averaging shifts the confidences away from the distribution the arbiter was trained on and
    broke routing (it sent confident Israeli dishes to the global model). predict() also letterboxes
    internally, so the result is already resolution independent.
    """
    with MODEL_LOCK:
        r = model.predict(bgr, imgsz=imgsz, verbose=False)[0]
    p = r.probs
    order = list(p.top5)
    conf = [float(c) for c in p.top5conf.tolist()]
    while len(conf) < 5:
        conf.append(0.0)
    full = np.clip(p.data.detach().cpu().numpy().astype(np.float64), 1e-12, 1.0)
    entropy = float(-(full * np.log(full)).sum())
    margin = float(conf[0] - conf[1])
    # P(background): the V2 Israeli model's "this is not one of my dishes" probability. 0.0 for the global
    # model (no such class). High value = strong "route to global" signal (the arbiter's top feature).
    bg_idx = next((k for k, v in r.names.items() if v == "background"), None)
    p_background = float(full[bg_idx]) if bg_idx is not None else 0.0
    return {"label": r.names[int(order[0])], "conf": conf, "entropy": entropy, "margin": margin,
            "p_background": p_background,
            "top5": [(r.names[int(i)], float(c)) for i, c in zip(order, conf)]}


def build_feature_row(g: dict, i: dict) -> pd.DataFrame:
    row = {}
    for k in range(5):
        row[f"g_conf{k+1}"] = g["conf"][k]
        row[f"i_conf{k+1}"] = i["conf"][k]
    row["g_entropy"], row["g_margin"] = g["entropy"], g["margin"]
    row["i_entropy"], row["i_margin"] = i["entropy"], i["margin"]
    row["i_p_background"] = i.get("p_background", 0.0)   # V2 arbiter feature
    row["conf_gap"] = g["conf"][0] - i["conf"][0]
    row["conf_ratio"] = g["conf"][0] / (i["conf"][0] + 1e-6)
    row["entropy_gap"] = i["entropy"] - g["entropy"]
    row["margin_gap"] = g["margin"] - i["margin"]
    row["both_unsure"] = int((g["conf"][0] < 0.5) and (i["conf"][0] < 0.5))
    return pd.DataFrame([row], columns=FEATS)


def embed_vectors(roi: np.ndarray):
    """Penultimate 512-D embeddings ('DNA') of both experts for one image, as numpy arrays.

    These are the features just before each model's classifier. Stored per tagged sample, they let us
    later train an embeddings-based router or an OSR detector (the V2 directions) on real-world data.
    """
    with MODEL_LOCK:
        g = model_g.embed(roi, imgsz=GLOBAL_IMGSZ, verbose=False)[0].detach().cpu().numpy().astype(np.float32)
        i = model_i.embed(roi, imgsz=ISRAELI_IMGSZ, verbose=False)[0].detach().cpu().numpy().astype(np.float32)
        # CRITICAL: embed() leaves model.predictor in 'embed' mode, so the NEXT predict() would return
        # raw tensors (probs=None) and every later analyze/tag would fail. Reset so predict() rebuilds a
        # classification predictor. This is the real cause of "tagging hangs after the first sample".
        model_g.predictor = None
        model_i.predictor = None
    return g, i


def predict(bgr: np.ndarray) -> dict:
    """Full edge decision. Returns everything the UI needs to explain itself."""
    # Feed the plain centre ROI. We do NOT segment the food with GrabCut: testing showed the models
    # are robust to background (a pepper scores 99.97% on a dark or wood background), while GrabCut
    # crops were unreliable and actively harmful (a clean hummus at 96% became malawach at 92% after
    # GrabCut). The simplest crop is the most accurate here.
    roi = center_roi(bgr)
    food = roi
    g = expert_features(model_g, roi, GLOBAL_IMGSZ)
    i = expert_features(model_i, roi, ISRAELI_IMGSZ)

    X = build_feature_row(g, i)
    p_israeli = float(arbiter.predict_proba(X)[0, 1])
    route_israeli = p_israeli >= 0.5
    # The arbiter's raw output is a log-odds (logit); the sigmoid of it is P(israeli). Recover it so the
    # UI can show the actual math: logit = sum(SHAP contributions) + bias, P = 1 / (1 + e^-logit).
    pc = min(max(p_israeli, 1e-7), 1 - 1e-7)
    logit = float(np.log(pc / (1 - pc)))

    chosen = i if route_israeli else g
    gate = GATE_ISRAELI if route_israeli else GATE_GLOBAL
    both_unsure = (g["conf"][0] < BOTH_UNSURE) and (i["conf"][0] < BOTH_UNSURE)
    # "arbiter undecided": P(israeli) sits near 0.5, so the router cannot confidently pick an expert.
    arbiter_unsure = abs(p_israeli - 0.5) < ARBITER_BAND
    # If the Israeli model was routed to but it says 'background' (= "not one of my dishes"), that is an
    # abstention, not a real food label -> treat as not confident so it falls back to the global guess/Gemini.
    israeli_abstain = route_israeli and chosen["label"] == "background"
    confident = (chosen["conf"][0] >= gate) and not both_unsure and not arbiter_unsure and not israeli_abstain

    return {"global": g, "israeli": i, "p_israeli": p_israeli, "route_israeli": route_israeli,
            "logit": logit, "chosen": chosen, "label": chosen["label"], "conf": chosen["conf"][0],
            "confident": confident, "both_unsure": both_unsure, "arbiter_unsure": arbiter_unsure,
            "israeli_abstain": israeli_abstain,
            "roi": roi, "food": food,
            "features": {f: float(X.iloc[0][f]) for f in FEATS},
            "reasons": arbiter_reasons(X)}


def predict_best(frames):
    """Run the full edge decision on several recent frames and keep the strongest one.

    Each frame is a genuine single-frame prediction, so the arbiter features are never averaged and
    routing is identical to the single-frame path (we do NOT distort the confidence distribution the
    arbiter was trained on). Taking the best-of-N simply raises the chance that at least one in-focus,
    well-lit, in-distribution frame is captured, which is the cheapest no-retrain accuracy win. Prefers
    a confident result; otherwise the highest top-1 confidence."""
    results = []
    for f in frames:
        try:
            results.append(predict(f))
        except Exception as e:
            print(f"[WARN] predict frame failed: {e}")
    if not results:
        return None
    confident = [r for r in results if r["confident"]]
    pool = confident if confident else results
    return max(pool, key=lambda r: r["conf"])


def arbiter_reasons(X: pd.DataFrame, k: int = 5):
    """Per-prediction feature contributions (TreeSHAP from the XGBoost booster).

    pred_contribs returns, for this single image, how many log-odds each feature added toward the
    'route to Israeli' decision. Positive pushes toward Israeli, negative toward Global. This is the
    exact, faithful 'why' behind the routing, not a global importance average.
    """
    try:
        booster = arbiter.get_booster()
        dm = xgb.DMatrix(X, feature_names=FEATS)
        contribs = booster.predict(dm, pred_contribs=True)[0]   # len = n_features + 1 (bias last)
        pairs = [(f, float(c)) for f, c in zip(FEATS, contribs[:-1])]
        pairs.sort(key=lambda kv: -abs(kv[1]))
        return [(f, c, "Israeli" if c > 0 else "Global") for f, c in pairs[:k]]
    except Exception as e:
        print(f"[WARN] arbiter_reasons failed: {e}")
        return []


def gemini_identify(bgr: np.ndarray) -> str | None:
    if gemini is None:
        return None
    try:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resp = gemini.generate_content(
            ["Identify the single main food. Reply with only its common English name, no punctuation.",
             Image.fromarray(rgb)])
        return resp.text.strip().lower().replace(" ", "_")
    except Exception as e:
        print(f"[WARN] Gemini failed: {e}")
        return None


# ==========================================================================================
# 4 · NUTRITION
# ==========================================================================================
def _usda_lookup(query: str):
    """Return (desc, cal, pro, carb, fat) per 100 g from USDA, or None."""
    if not (requests and USDA_API_KEY):
        return None
    try:
        r = requests.get("https://api.nal.usda.gov/fdc/v1/foods/search",
                         params={"api_key": USDA_API_KEY, "query": query, "pageSize": 5,
                                 "dataType": ["Foundation", "SR Legacy"]}, timeout=8)
        foods = r.json().get("foods", [])
        if not foods:
            return None
        ids = {1008: "cal", 2047: "cal", 1003: "pro", 1005: "carb", 1004: "fat"}
        got = {}
        for n in foods[0]["foodNutrients"]:
            nm = ids.get(n.get("nutrientId"))
            if nm and nm not in got:
                got[nm] = n.get("value", 0)
        if not got.get("cal"):
            return None
        return (foods[0]["description"], got.get("cal", 0), got.get("pro", 0),
                got.get("carb", 0), got.get("fat", 0))
    except Exception as e:
        print(f"[WARN] USDA failed: {e}")
        return None


def _gemini_nutrition(query: str):
    """Ask Gemini for per-100 g macros when local DB and USDA both miss. Returns tuple or None."""
    if gemini is None:
        return None
    try:
        import json
        resp = gemini.generate_content(
            f"Give typical nutrition per 100 g of '{query.replace('_', ' ')}'. "
            f"Reply ONLY as compact JSON: {{\"cal\":<kcal>,\"pro\":<g>,\"carb\":<g>,\"fat\":<g>}}.")
        txt = resp.text.strip().strip("`")
        txt = txt[txt.find("{"):txt.rfind("}") + 1]
        d = json.loads(txt)
        return (query.replace("_", " ").title() + " (est.)",
                float(d["cal"]), float(d["pro"]), float(d["carb"]), float(d["fat"]))
    except Exception as e:
        print(f"[WARN] Gemini nutrition failed: {e}")
        return None


def nutrition(label: str, grams: int) -> dict:
    """Resolve macros: local DB -> USDA -> Gemini estimate. Always returns numbers if any source works."""
    key = label.lower().replace(" ", "_")
    ratio = max(grams, 0) / 100.0
    if key in LOCAL_DB:
        desc, cal, pro, carb, fat = LOCAL_DB[key]
        src = "local DB"
    else:
        res = _usda_lookup(label.replace("_", " "))
        src = "USDA"
        if res is None:
            res = _gemini_nutrition(label)   # cloud estimate so the demo never shows blank macros
            src = "Gemini est."
        if res is None:
            desc, cal, pro, carb, fat = label.replace("_", " ").title(), 0, 0, 0, 0
            src = "no source (set USDA_API_KEY / GEMINI_API_KEY)"
        else:
            desc, cal, pro, carb, fat = res
    return {"name": label.replace("_", " ").title(), "desc": desc, "src": src, "weight": grams,
            "cal": int(cal * ratio), "pro": round(pro * ratio, 1),
            "carb": round(carb * ratio, 1), "fat": round(fat * ratio, 1)}


# ==========================================================================================
# 5 · BLE WEIGHT  (protocol recovered by reverse engineering; see scripts/ble/scale_reader.py)
# ==========================================================================================
weight_buffer = deque([0] * MAX_HISTORY, maxlen=MAX_HISTORY)
raw_buffer = deque(maxlen=5)   # last few raw decodes, median-filtered to reject flicker/overshoot
current_weight = 0
manual_weight = 0          # used when the BLE scale is not connected
ble_status = "no BLE" if not _BLE_OK else "scanning"
stable = False
last_rx = 0.0              # time of the last notification, for the staleness watchdog
STALE_SEC = 4.0           # no packets for this long -> force a reconnect


def get_weight() -> int:
    """Live BLE weight when connected, otherwise the value typed in the UI."""
    return int(current_weight) if ble_status == "connected" else int(manual_weight)


def on_notify(_sender, data):
    """Decode one packet to grams.

    weight = low_byte + high_byte * 256, with the high byte recovered as in the reverse-engineered
    protocol. We do NOT latch the high byte (the old "sticky high" hack caused a transient overshoot
    to add a permanent +256 g offset, so a removed-then-replaced load read double). Instead we take
    the MEDIAN of the last few raw decodes: a single bad frame (flicker or momentary overshoot) is
    outvoted, and the reading follows the scale all the way back down to 0.
    """
    global current_weight, ble_status, stable, last_rx
    b = list(data)
    if len(b) < 8:
        return
    low = b[4]
    # high byte = number of 256s, carried in b[3]. Do NOT mask bit0: the old `& 0xFE` cleared the
    # low bit of the high byte, so every ODD multiple of 256 lost 256 g (e.g. 272 g -> 16 g). b[5] is
    # status/zero in practice and is OR-ed in only defensively.
    high = b[5] | b[3]
    raw = low + high * 256
    raw_buffer.append(raw)
    s = sorted(raw_buffer)
    w = s[len(s) // 2]                 # median of the last few raw decodes
    current_weight = w
    weight_buffer.append(w)
    last10 = list(weight_buffer)[-10:]
    stable = len(last10) == 10 and (max(last10) - min(last10)) <= 2
    ble_status = "connected"
    last_rx = time.time()


async def _ble_loop():
    global ble_status, last_rx
    while True:
        try:
            dev = await BleakScanner.find_device_by_name("SWAN")
            if dev:
                async with BleakClient(dev) as c:
                    await c.start_notify(NOTIFY_UUID, on_notify)
                    last_rx = time.time()
                    # Staleness watchdog: some scales stop notifying when idle / power-saving, and the
                    # link can silently go quiet (is_connected stays True). If no packet arrives for
                    # STALE_SEC, break out to disconnect and rescan, which re-subscribes and makes the
                    # reading resume the moment a load is placed again. Without this, a removed-then-
                    # replaced object would never update.
                    while c.is_connected:
                        await asyncio.sleep(0.4)
                        if time.time() - last_rx > STALE_SEC:
                            ble_status = "reconnecting"
                            raw_buffer.clear()
                            break
            else:
                ble_status = "searching"
                await asyncio.sleep(2)
        except Exception:
            ble_status = "retrying"
            await asyncio.sleep(2)


def start_ble():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_ble_loop())


# ==========================================================================================
# 6 · GUI
# ==========================================================================================
class AutocompleteEntry(tk.Entry):
    """Entry that suggests matching class names in a dropdown as you type (substring match)."""

    def __init__(self, parent, textvar, options, **kw):
        super().__init__(parent, textvariable=textvar, **kw)
        self.var = textvar
        self.options = options
        self.pop = None
        self.lb = None
        self.bind("<KeyRelease>", self._on_key)
        self.bind("<FocusOut>", lambda e: self.after(150, self._close))
        self.bind("<Escape>", lambda e: self._close())
        self.bind("<Return>", lambda e: self._close())

    def _norm(self, s):
        return s.strip().lower().replace(" ", "_")

    def _on_key(self, e):
        if e.keysym in ("Return", "Escape", "Up", "Down"):
            return
        txt = self._norm(self.var.get())
        if not txt:
            self._close()
            return
        starts = [o for o in self.options if o.startswith(txt)]
        contains = [o for o in self.options if txt in o and not o.startswith(txt)]
        matches = (starts + contains)[:8]
        if not matches or (len(matches) == 1 and matches[0] == txt):
            self._close()
            return
        self._show(matches)

    def _show(self, matches):
        if self.pop is None:
            self.pop = tk.Toplevel(self)
            self.pop.wm_overrideredirect(True)
            self.lb = tk.Listbox(self.pop, bg=CARD, fg=INK, selectbackground=PURPLE,
                                 font=("Segoe UI", 10), bd=0, highlightthickness=1,
                                 highlightbackground="#30363d", activestyle="none")
            self.lb.pack(fill="both", expand=True)
            self.lb.bind("<ButtonRelease-1>", self._pick)
        self.lb.delete(0, tk.END)
        for m in matches:
            self.lb.insert(tk.END, m)
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height()
        self.pop.wm_geometry(f"{max(self.winfo_width(), 160)}x{20 * len(matches)}+{x}+{y}")

    def _pick(self, e):
        sel = self.lb.curselection()
        if sel:
            self.var.set(self.lb.get(sel[0]))
        self._close()
        self.icursor(tk.END)

    def _close(self):
        if self.pop is not None:
            self.pop.destroy()
            self.pop = None
            self.lb = None


class CalEyeZDemo:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.running = True
        self.frame_bgr = None
        self.frames = deque(maxlen=BEST_OF_N)   # recent frames for best-of-N analysis (2 on CPU/edge, 6 on GPU)
        self.last = None
        self.last = None
        root.title("CalEyeZ  ·  Live Demo")
        root.geometry("1240x820")
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self._init_log()
        self._style()
        self._build_left(root)
        self._build_right(root)

        self.cap = open_camera()
        self._tick_camera()
        self._tick_weight()

    # ---------- styling ----------
    def _style(self):
        st = ttk.Style()
        st.theme_use("clam")
        st.configure("TCombobox", fieldbackground=CARD, background=CARD, foreground=INK)
        st.configure("Treeview", background=CARD, fieldbackground=CARD, foreground=INK,
                     rowheight=24, font=("Segoe UI", 10))
        st.configure("Treeview.Heading", background=PANEL, foreground=MUT, font=("Segoe UI", 9, "bold"))
        st.map("Treeview", background=[("selected", ACCENT)])

    def _chip(self, parent, text, fg):
        return tk.Label(parent, text=text, bg=CARD, fg=fg, font=("Segoe UI", 9, "bold"),
                        padx=8, pady=2)

    # ---------- left: camera + decision ----------
    def _build_left(self, root):
        left = tk.Frame(root, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=12, pady=12)

        self.video = tk.Label(left, bg="black", bd=1, relief="solid")
        self.video.pack(pady=(0, 10))

        # live decision panel (explains every prediction)
        dec = tk.Frame(left, bg=PANEL, bd=1, relief="solid")
        dec.pack(fill="x")
        tk.Label(dec, text="ARBITER DECISION", bg=PANEL, fg=MUT,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=12, pady=(8, 2))
        self.bar_global = self._expert_row(dec, "Global (132)", ACCENT)
        self.bar_israeli = self._expert_row(dec, "Israeli (13)", TEAL)
        self.route_lbl = tk.Label(dec, text="P(israeli) = -", bg=PANEL, fg=INK,
                                  font=("Segoe UI", 10, "bold"))
        self.route_lbl.pack(anchor="w", padx=12, pady=(2, 2))
        tk.Label(dec, text="WHY  (push toward  ◀ Global   |   Israeli ▶)", bg=PANEL, fg=MUT,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=12)
        self.reasons_canvas = tk.Canvas(dec, height=104, bg=PANEL, highlightthickness=0)
        self.reasons_canvas.pack(fill="x", padx=12, pady=(2, 6))

        # live arbiter math: the actual 19 features and the logit -> sigmoid -> P(israeli) computation
        tk.Label(dec, text="ARBITER MATH  (live 19-feature vector -> P)", bg=PANEL, fg=MUT,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=12)
        self.math_lbl = tk.Label(dec, text="(press Analyze)", bg=CARD, fg=INK, justify="left",
                                  anchor="w", font=("Consolas", 9), padx=8, pady=6)
        self.math_lbl.pack(fill="x", padx=12, pady=(2, 10))

    def _expert_row(self, parent, name, color):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", padx=12, pady=3)
        tk.Label(row, text=name, bg=PANEL, fg=MUT, width=12, anchor="w",
                 font=("Segoe UI", 10)).pack(side="left")
        canvas = tk.Canvas(row, height=16, bg=CARD, highlightthickness=0)
        canvas.pack(side="left", fill="x", expand=True, padx=8)
        lbl = tk.Label(row, text="-", bg=PANEL, fg=INK, width=22, anchor="w",
                       font=("Segoe UI", 10))
        lbl.pack(side="left")
        return {"canvas": canvas, "lbl": lbl, "color": color}

    def _set_bar(self, bar, label, conf):
        c = bar["canvas"]
        c.delete("all")
        w = max(c.winfo_width(), 1)
        c.create_rectangle(0, 0, int(w * conf), 16, fill=bar["color"], width=0)
        bar["lbl"].config(text=f"{label.replace('_', ' ')}  {conf*100:.0f}%")

    def _draw_reasons(self, reasons, feats):
        """SHAP-style force chart: each feature is a bar from the centre, left = pushed toward Global,
        right = pushed toward Israeli, length proportional to its contribution."""
        c = self.reasons_canvas
        c.delete("all")
        if not reasons:
            return
        w = max(c.winfo_width(), 700)
        name_w = 150                         # left column for "feature (value)"
        cx = name_w + (w - name_w - 90) / 2  # centre of the force area
        half = (w - name_w - 90) / 2 - 6
        mx = max(abs(r[1]) for r in reasons) or 1.0
        rows = reasons[:4]
        rh = 24
        c.create_line(cx, 6, cx, 6 + rh * len(rows), fill=MUT, dash=(2, 2))
        for i, (f, contrib, direction) in enumerate(rows):
            y = 14 + i * rh
            c.create_text(6, y, anchor="w", text=f"{f} ({feats.get(f, 0):.2f})",
                          fill=INK, font=("Consolas", 10))
            length = abs(contrib) / mx * half
            if contrib > 0:                  # toward Israeli (right, teal)
                c.create_rectangle(cx, y - 7, cx + length, y + 7, fill=TEAL, width=0)
                c.create_text(cx + length + 5, y, anchor="w", text=f"+{contrib:.2f}",
                              fill=TEAL, font=("Consolas", 9))
            else:                            # toward Global (left, blue)
                c.create_rectangle(cx - length, y - 7, cx, y + 7, fill=ACCENT, width=0)
                c.create_text(cx - length - 5, y, anchor="e", text=f"{contrib:.2f}",
                              fill=ACCENT, font=("Consolas", 9))

    def _update_math(self, d):
        """Show the arbiter's live arithmetic: the 19 input features, then the logit -> sigmoid -> P.

        This is the exact computation the XGBoost arbiter runs every capture, made visible so you can
        read the numbers that produced the routing decision on stage."""
        f = d.get("features", {})
        g5 = " ".join(f"{f.get(f'g_conf{k}',0):.2f}" for k in range(1, 6))
        i5 = " ".join(f"{f.get(f'i_conf{k}',0):.2f}" for k in range(1, 6))
        p = d.get("p_israeli", 0.0)
        logit = d.get("logit", 0.0)
        decision = "ISRAELI" if d.get("route_israeli") else "GLOBAL"
        flag = ""
        if d.get("arbiter_unsure"):
            flag = "   <- UNDECIDED (near 0.5)"
        txt = (
            f"GLOBAL : conf1-5 [{g5}]\n"
            f"         entropy={f.get('g_entropy',0):.2f}  margin={f.get('g_margin',0):.2f}\n"
            f"ISRAELI: conf1-5 [{i5}]\n"
            f"         entropy={f.get('i_entropy',0):.2f}  margin={f.get('i_margin',0):.2f}"
            f"  P(background)={f.get('i_p_background',0):.2f}\n"
            f"INTERACT conf_gap={f.get('conf_gap',0):+.2f}  conf_ratio={f.get('conf_ratio',0):.2f}\n"
            f"         entropy_gap={f.get('entropy_gap',0):+.2f}  margin_gap={f.get('margin_gap',0):+.2f}"
            f"  both_unsure={int(f.get('both_unsure',0))}\n"
            f"--------------------------------------------\n"
            f"logit(sum SHAP+bias) = {logit:+.3f}\n"
            f"P(israeli) = sigmoid({logit:+.2f}) = {p:.3f}\n"
            f"P {'>=' if p>=0.5 else '<'} 0.50  ->  route to {decision}{flag}"
        )
        self.math_lbl.config(text=txt)

    # ---------- right: weight + controls + interactive log ----------
    def _build_right(self, root):
        right = tk.Frame(root, bg=PANEL, width=460)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self.weight_lbl = tk.Label(right, text="0 g", bg=PANEL, fg=TEAL, font=("Segoe UI", 46, "bold"))
        self.weight_lbl.pack(pady=(14, 0))
        self.ble_lbl = self._chip(right, "scanning", MUT)
        self.ble_lbl.pack()

        # manual weight fallback (used automatically when the scale is not connected)
        man = tk.Frame(right, bg=PANEL)
        man.pack(pady=(6, 0))
        tk.Label(man, text="manual g:", bg=PANEL, fg=MUT, font=("Segoe UI", 9)).pack(side="left")
        self.manual_var = tk.StringVar(value="0")
        e = tk.Entry(man, textvariable=self.manual_var, width=7, justify="center",
                     bg=CARD, fg=INK, insertbackground=INK, relief="flat")
        e.pack(side="left", padx=6)
        e.bind("<KeyRelease>", self._set_manual)

        ctl = tk.Frame(right, bg=PANEL)
        ctl.pack(fill="x", padx=24, pady=12)
        tk.Label(ctl, text="Preparation", bg=PANEL, fg=MUT, font=("Segoe UI", 9)).pack(anchor="w")
        self.prep = tk.StringVar(value="Raw / Standard")
        ttk.Combobox(ctl, textvariable=self.prep, state="readonly",
                     values=["Raw / Standard", "Grilled / Roasted", "Fried / Oil"]).pack(fill="x", pady=4)

        self.btn = tk.Button(right, text="ANALYZE", bg=ACCENT, fg="#08111f",
                             font=("Segoe UI", 14, "bold"), bd=0, command=self.analyze)
        self.btn.pack(fill="x", padx=24, pady=(4, 8))

        # result card
        self.result = tk.Label(right, text="Place food in the box and press Analyze.",
                               bg=CARD, fg=INK, font=("Consolas", 11), justify="left",
                               anchor="w", padx=12, pady=12)
        self.result.pack(fill="x", padx=24)
        self.src_chip = self._chip(right, "", MUT)
        self.src_chip.pack(anchor="w", padx=24, pady=(4, 0))

        # Manual cloud override: the confidence gate cannot catch a confident-WRONG local guess (a
        # top-down object the model reads as the wrong food at high confidence). This button lets you
        # force the Gemini vision identifier on the current view, so the demo is never stuck on a bad
        # local prediction. Requires GEMINI_API_KEY.
        self.btn_gem = tk.Button(right, text="ASK GEMINI (override)", bg=CARD, fg=PURPLE,
                                 font=("Segoe UI", 10, "bold"), bd=0, command=self.ask_gemini)
        self.btn_gem.pack(fill="x", padx=24, pady=(10, 0))

        self.btn_save = tk.Button(right, text="SAVE TO LOG", bg=CARD, fg=GREEN,
                                  font=("Segoe UI", 10, "bold"), bd=0, state="disabled", command=self.save)
        self.btn_save.pack(fill="x", padx=24, pady=(6, 4))

        # ---- DATA TAGGING: type what it really is, then capture image + predictions + 512-D DNA ----
        tk.Label(right, text="DATA TAGGING (ground truth)", bg=PANEL, fg=PURPLE,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=24, pady=(6, 0))
        tagrow = tk.Frame(right, bg=PANEL)
        tagrow.pack(fill="x", padx=24)
        self.gt_var = tk.StringVar()
        AutocompleteEntry(tagrow, self.gt_var, ALL_CLASSES, bg=CARD, fg=INK, insertbackground=INK,
                          relief="flat", font=("Segoe UI", 11)).pack(side="left", fill="x", expand=True, ipady=3)
        self.btn_tag = tk.Button(tagrow, text="TAG", bg=PURPLE, fg="#0d1117", font=("Segoe UI", 10, "bold"),
                                 bd=0, command=self.tag_sample, width=6)
        self.btn_tag.pack(side="left", padx=(6, 0))
        self.tag_status = tk.Label(right, text="tagged: 0", bg=PANEL, fg=MUT, font=("Consolas", 9))
        self.tag_status.pack(anchor="w", padx=24, pady=(2, 6))

        # interactive log embedded in the main window (no popups)
        tk.Label(right, text="TODAY'S LOG", bg=PANEL, fg=MUT,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=24)
        cols = ("time", "food", "g", "kcal")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=8)
        for c, w in zip(cols, (60, 150, 50, 60)):
            self.tree.heading(c, text=c.upper())
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill="x", padx=24, pady=6)
        self.tree.tag_configure("isr", foreground=TEAL)
        self.tree.tag_configure("glb", foreground=ACCENT)

        self.totals = tk.Label(right, text="", bg=PANEL, fg=INK, font=("Segoe UI", 10, "bold"), justify="left")
        self.totals.pack(anchor="w", padx=24, pady=(2, 6))

        bar = tk.Frame(right, bg=PANEL)
        bar.pack(fill="x", padx=24)
        tk.Button(bar, text="Delete selected", bg=CARD, fg=RED, bd=0,
                  font=("Segoe UI", 9, "bold"), command=self.delete_selected).pack(side="left")
        self._refresh_log()

    # ---------- camera / weight loops ----------
    def _tick_camera(self):
        if not self.running:
            return
        ok, frame = self.cap.read()
        if ok:
            self.frame_bgr = frame.copy()
            self.frames.append(frame.copy())
            h, w = frame.shape[:2]
            s = int(min(h, w) * ROI_FRAC)
            y0, x0 = (h - s) // 2, (w - s) // 2
            cv2.rectangle(frame, (x0, y0), (x0 + s, y0 + s), (204, 255, 0), 2)
            cv2.putText(frame, "PLACE FOOD HERE", (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (204, 255, 0), 2)
            # The models were trained on side / eye-level food photos. A top-down webcam view is
            # out-of-distribution; aiming the camera at ~45 deg / eye level matches training and is the
            # single biggest accuracy win, no retraining needed.
            cv2.putText(frame, "tip: aim camera at eye level / ~45deg", (x0, y0 + s + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 200, 255), 1)
            # NOTE: food isolation (GrabCut) runs only inside predict() on ANALYZE, never in this live
            # loop. GrabCut is an iterative GMM and is CPU-heavy; running it per frame here would stutter
            # the webcam feed while two YOLO models are resident. The live feed shows only the guide box.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((620, 465), Image.Resampling.LANCZOS)
            tkimg = ImageTk.PhotoImage(img)
            self.video.imgtk = tkimg
            self.video.configure(image=tkimg)
        self.root.after(30, self._tick_camera)

    def _set_manual(self, _evt=None):
        global manual_weight
        try:
            manual_weight = max(0, int(float(self.manual_var.get() or 0)))
        except ValueError:
            manual_weight = 0

    def _tick_weight(self):
        if not self.running:
            return
        connected = ble_status == "connected"
        self.weight_lbl.config(text=f"{get_weight()} g",
                               fg=TEAL if (connected and stable) else AMBER)
        if connected:
            self.ble_lbl.config(text="BLE: connected" + ("  (stable)" if stable else ""), fg=GREEN)
        elif not _BLE_OK:
            self.ble_lbl.config(text="BLE off (pip install bleak) - using manual g", fg=AMBER)
        else:
            self.ble_lbl.config(text=f"BLE: {ble_status} - using manual g", fg=MUT)
        self.root.after(120, self._tick_weight)

    # ---------- manual Gemini override ----------
    def ask_gemini(self):
        if gemini is None:
            messagebox.showinfo("Gemini", "Set GEMINI_API_KEY (and pip install google-generativeai) "
                                          "to enable the cloud fallback.")
            return
        if self.frame_bgr is None or model_g is None:
            return
        self.btn_gem.config(state="disabled", text="ASKING GEMINI...", fg=MUT)
        threading.Thread(target=self._gemini_worker,
                         args=(self.frame_bgr.copy(), get_weight()), daemon=True).start()

    def _gemini_worker(self, bgr, grams):
        try:
            d = self.last["decision"] if self.last else predict(bgr)
            food = d.get("food") if isinstance(d, dict) else center_roi(bgr)
            name = gemini_identify(food if food is not None else center_roi(bgr))
            if not name:
                self.root.after(0, lambda: self._gemini_done(None, None))
                return
            info = nutrition(name, grams)
            info["used_gemini"] = True
            info["decision"] = d
            self.root.after(0, lambda: self._gemini_done(info, None))
        except Exception as e:
            self.root.after(0, lambda: self._gemini_done(None, str(e)))

    def _gemini_done(self, info, err):
        self.btn_gem.config(state="normal", text="ASK GEMINI (override)", fg=PURPLE)
        if info is None:
            self.src_chip.config(text=f"Gemini override failed ({err})" if err else
                                 "Gemini could not identify the food", fg=AMBER)
            return
        self._show(info)

    # ---------- analyze ----------
    def analyze(self):
        if self.frame_bgr is None or model_g is None:
            messagebox.showwarning("Not ready", "Camera or models not ready.")
            return
        self.btn.config(state="disabled", text="THINKING...", bg=MUT)
        self.result.config(text="Running edge inference...")
        frames = [f.copy() for f in self.frames] or [self.frame_bgr.copy()]
        threading.Thread(target=self._analyze_worker,
                         args=(frames, get_weight(), self.prep.get()),
                         daemon=True).start()

    def _analyze_worker(self, frames, grams, prep):
        try:
            d = predict_best(frames)
            if d is None:
                raise RuntimeError("inference produced no result")
            used_gemini = False
            label = d["label"]
            if not d["confident"]:
                g = gemini_identify(d["food"])
                if g:
                    label, used_gemini = g, True
            info = nutrition(label, grams)
            info["used_gemini"] = used_gemini
            info["decision"] = d
            self.root.after(0, lambda: self._show(info))
        except Exception as e:
            self.root.after(0, lambda: self._fail(str(e)))

    def _show(self, info):
        d = info["decision"]
        self._set_bar(self.bar_global, d["global"]["label"], d["global"]["conf"][0])
        self._set_bar(self.bar_israeli, d["israeli"]["label"], d["israeli"]["conf"][0])
        route_txt = f"P(israeli) = {d['p_israeli']*100:.0f}%  ->  route to "
        route_txt += "ISRAELI" if d["route_israeli"] else "GLOBAL"
        self.route_lbl.config(text=route_txt, fg=TEAL if d["route_israeli"] else ACCENT)
        self._draw_reasons(d.get("reasons", []), d.get("features", {}))
        self._update_math(d)

        self.btn.config(state="normal", text="ANALYZE", bg=ACCENT)
        self.last = info
        self.btn_save.config(state="normal")
        self.result.config(text=(
            f"FOOD : {info['name']}\n"
            f"{info['desc'][:34]}\n"
            f"---------------------------\n"
            f"WEIGHT : {info['weight']} g\n"
            f"CAL    : {info['cal']}\n"
            f"PROTEIN: {info['pro']} g\n"
            f"CARBS  : {info['carb']} g\n"
            f"FAT    : {info['fat']} g"))
        if info["used_gemini"]:
            why = ("arbiter undecided" if d.get("arbiter_unsure") else
                   "both experts unsure" if d.get("both_unsure") else "low confidence")
            self.src_chip.config(text=f"source: Gemini fallback ({why})", fg=PURPLE)
        elif not d["confident"]:
            # the system flagged this input as uncertain. Three distinct reasons, shown explicitly:
            if d.get("arbiter_unsure"):
                reason = f"ARBITER UNDECIDED  P(israeli)={d['p_israeli']*100:.0f}% (near 50/50)"
            elif d.get("both_unsure"):
                reason = "BOTH EXPERTS UNSURE (out-of-distribution)"
            else:
                reason = f"LOW CONFIDENCE {d['conf']*100:.0f}%"
            hint = "set GEMINI_API_KEY for cloud fallback" if gemini is None else "Gemini fallback failed"
            self.src_chip.config(text=f"{reason} ({hint})", fg=AMBER)
        else:
            self.src_chip.config(text=f"source: {'Israeli' if d['route_israeli'] else 'Global'} expert  ·  {info['src']}",
                                 fg=GREEN)

    def _fail(self, msg):
        self.btn.config(state="normal", text="ANALYZE", bg=ACCENT)
        self.result.config(text=f"Error:\n{msg}", fg=RED)

    # ---------- logging ----------
    def _init_log(self):
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w", newline="") as f:
                csv.writer(f).writerow(["Date", "Time", "Food", "Weight(g)", "Calories", "Protein", "Carbs", "Fat", "Route"])
        os.makedirs(TAG_IMG_DIR, exist_ok=True)
        os.makedirs(TAG_EMB_DIR, exist_ok=True)
        if not os.path.exists(TAG_CSV):
            with open(TAG_CSV, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "id", "datetime", "ground_truth", "weight_g",
                    "global_pred", "global_conf", "israeli_pred", "israeli_conf",
                    "p_israeli", "route", "final_pred",
                    "global_correct", "israeli_correct", "system_correct",
                    "global_top5", "israeli_top5", "image_path", "embed_path"])
        # restore the running tagged/correct count from any existing log
        self.tag_n = self.tag_ok = 0
        if os.path.exists(TAG_CSV):
            with open(TAG_CSV, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    self.tag_n += 1
                    self.tag_ok += (r.get("system_correct") == "True")

    def tag_sample(self):
        gt = self.gt_var.get().strip().lower().replace(" ", "_")
        if not gt:
            messagebox.showwarning("Ground truth", "Type what the food actually is, then press TAG.")
            return
        if self.frame_bgr is None or model_g is None:
            return
        self.btn_tag.config(state="disabled")
        self.tag_status.config(text="tagging...", fg=AMBER)
        threading.Thread(target=self._tag_worker,
                         args=(self.frame_bgr.copy(), get_weight(), gt), daemon=True).start()

    def _tag_worker(self, bgr, grams, gt):
        try:
            import json
            d = predict(bgr)
            roi = center_roi(bgr)
            g_emb, i_emb = embed_vectors(roi)
            sid = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            # 1) labelled image, saved as the centre ROI = exactly what the models saw and embedded,
            #    in ImageFolder layout so it is directly usable for fine-tuning.
            cls_dir = os.path.join(TAG_IMG_DIR, gt)
            os.makedirs(cls_dir, exist_ok=True)
            img_path = os.path.join(cls_dir, sid + ".jpg")
            cv2.imwrite(img_path, roi)
            # 2) the 512-D DNA of both experts
            emb_path = os.path.join(TAG_EMB_DIR, sid + ".npz")
            np.savez_compressed(emb_path, g=g_emb, i=i_emb,
                                ground_truth=gt, route=("israeli" if d["route_israeli"] else "global"))
            # 3) verification row
            g_pred, i_pred = d["global"]["label"], d["israeli"]["label"]
            final = d["label"]
            g_ok, i_ok = (g_pred == gt), (i_pred == gt)
            sys_ok = (final == gt)
            row = [sid, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), gt, grams,
                   g_pred, f"{d['global']['conf'][0]:.4f}", i_pred, f"{d['israeli']['conf'][0]:.4f}",
                   f"{d['p_israeli']:.4f}", "israeli" if d["route_israeli"] else "global", final,
                   g_ok, i_ok, sys_ok,
                   json.dumps(d["global"]["top5"]), json.dumps(d["israeli"]["top5"]),
                   os.path.relpath(img_path, ROOT), os.path.relpath(emb_path, ROOT)]
            with open(TAG_CSV, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
            self.tag_n += 1
            self.tag_ok += int(sys_ok)
            self.root.after(0, lambda: self._tag_done(gt, final, sys_ok))
        except Exception as e:
            msg = str(e)
            self.root.after(0, lambda: (self.tag_status.config(text=f"tag failed: {msg}", fg=RED),
                                        self.btn_tag.config(state="normal")))

    def _tag_done(self, gt, final, ok):
        acc = 100.0 * self.tag_ok / max(self.tag_n, 1)
        mark = "OK" if ok else f"MISS (got {final})"
        self.tag_status.config(
            text=f"tagged: {self.tag_n}  ·  sys acc: {acc:.0f}%  ·  last: {gt} {mark}",
            fg=GREEN if ok else AMBER)
        # keep the label so you can take multiple shots of the same dish quickly; clear it yourself
        # (or autocomplete a new one) when you switch dishes.
        self.btn_tag.config(state="normal")

    def save(self):
        if not self.last:
            return
        d = self.last["decision"]
        now = datetime.now()
        route = "israeli" if d["route_israeli"] else "global"
        with open(LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"),
                                    self.last["name"], self.last["weight"], self.last["cal"],
                                    self.last["pro"], self.last["carb"], self.last["fat"], route])
        self.btn_save.config(state="disabled")
        self._refresh_log()

    def _refresh_log(self):
        for r in self.tree.get_children():
            self.tree.delete(r)
        today = datetime.now().strftime("%Y-%m-%d")
        tc = tp = tcb = tf = 0
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE) as f:
                for row in csv.DictReader(f):
                    if row["Date"] != today:
                        continue
                    tag = "isr" if row.get("Route") == "israeli" else "glb"
                    self.tree.insert("", "end",
                                     values=(row["Time"][:5], row["Food"], row["Weight(g)"], row["Calories"]),
                                     tags=(tag,))
                    tc += int(float(row["Calories"])); tp += float(row["Protein"])
                    tcb += float(row["Carbs"]); tf += float(row["Fat"])
        self.totals.config(text=f"TODAY   {tc} kcal   ·   P {tp:.0f}g   C {tcb:.0f}g   F {tf:.0f}g")

    def delete_selected(self):
        sel = self.tree.selection()
        if not sel:
            return
        wanted = {self.tree.item(s)["values"][0] + "|" + str(self.tree.item(s)["values"][1]) for s in sel}
        rows, header = [], None
        with open(LOG_FILE) as f:
            reader = csv.reader(f)
            header = next(reader)
            for r in reader:
                key = r[1][:5] + "|" + r[2]
                if r[0] == datetime.now().strftime("%Y-%m-%d") and key in wanted:
                    continue
                rows.append(r)
        with open(LOG_FILE, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        self._refresh_log()

    def close(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    if _BLE_OK:
        threading.Thread(target=start_ble, daemon=True).start()
    root = tk.Tk()
    CalEyeZDemo(root)
    root.mainloop()
