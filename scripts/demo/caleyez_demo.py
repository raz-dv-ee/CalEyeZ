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
import threading
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
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

GLOBAL_W = os.path.join(ROOT, "runs", "general_model_flattened", "weights", "best.pt")
ISRAELI_W = os.path.join(ROOT, "runs", "israeli_food_yolo11l", "weights", "best.pt")
ARBITER_J = os.path.join(ROOT, "scripts", "arbiter", "arbiter_xgb.json")

GLOBAL_IMGSZ = 320            # matches how the global model was trained / the arbiter dataset
ISRAELI_IMGSZ = 224          # matches how the Israeli model was trained
ROI_FRAC = 0.60              # central square fraction used as the food region

# Confidence gate -> Gemini fallback. These target the "both models wrong" pool (~11% of images).
GATE_GLOBAL = 0.45           # global model is broad (132 cls): a modest top-1 is still trustworthy
GATE_ISRAELI = 0.60          # Israeli model is narrow (13 cls): demand a higher top-1
BOTH_UNSURE = 0.50           # if BOTH experts' top-1 < this, treat as out-of-distribution -> Gemini

# Keys are read from the environment so they are not committed. Fallbacks keep the demo runnable.
USDA_API_KEY = os.environ.get("USDA_API_KEY", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

LOG_FILE = os.path.join(ROOT, "nutrition_log.csv")
NOTIFY_UUID = "0000ffb2-0000-1000-8000-00805f9b34fb"   # SWAN scale notify characteristic
MAX_HISTORY = 100

# theme
BG, PANEL, CARD, INK, MUT = "#0d1117", "#161b22", "#1c2230", "#e6edf3", "#8b949e"
ACCENT, GREEN, AMBER, RED, TEAL, PURPLE = "#58a6ff", "#3fb950", "#d29922", "#f85149", "#39d0c4", "#bc8cff"

# 19 arbiter features, in the exact training order (scripts/arbiter/train_arbiter_xgb.py)
FEATS = ([f"g_conf{k}" for k in range(1, 6)] + ["g_entropy", "g_margin"]
         + [f"i_conf{k}" for k in range(1, 6)] + ["i_entropy", "i_margin"]
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


def food_bbox(roi: np.ndarray):
    """Bounding box of the food object inside the ROI, in ROI coordinates, or None.

    Why this is the key fix: the models were trained on images where the food fills most of the
    frame. A fixed ROI on a real table still contains a lot of non-food context (the dark scale
    glass, wood grain, a hand), and that clutter dominates the prediction, producing confident
    nonsense. We isolate the food as the region whose colour differs from the ROI border (which is
    almost always background), take the largest such blob, and crop to it. If the mask looks
    implausible (too small or nearly the whole frame), we return None and keep the full ROI rather
    than guess. No training and no plate assumption, so it is robust to lighting and background.
    """
    h, w = roi.shape[:2]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    bw = max(4, int(min(h, w) * 0.08))
    border = np.concatenate([lab[:bw].reshape(-1, 3), lab[-bw:].reshape(-1, 3),
                             lab[:, :bw].reshape(-1, 3), lab[:, -bw:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    mask = (dist > 22).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 0.05 * h * w or area > 0.95 * h * w:   # implausible mask -> keep full ROI
        return None
    x, y, bw2, bh2 = cv2.boundingRect(c)
    pad = int(0.10 * max(bw2, bh2))
    return (max(0, x - pad), max(0, y - pad), min(w, x + bw2 + pad), min(h, y + bh2 + pad))


def tighten_to_food(roi: np.ndarray) -> np.ndarray:
    bb = food_bbox(roi)
    if bb is None:
        return roi
    x0, y0, x1, y1 = bb
    crop = roi[y0:y1, x0:x1]
    return crop if crop.size else roi


def clahe_luma(bgr: np.ndarray) -> np.ndarray:
    """Equalise local contrast on the luminance (L) channel only.

    Why this and NOT gray-world white balance: gray-world assumes the average of the scene is
    neutral gray, which is false when one coloured object fills the ROI (a red pepper, a tomato).
    There gray-world desaturates the object and changes its hue, which can flip the prediction
    (a red pepper read as a beige pastry). CLAHE works on L in LAB space and leaves the colour
    channels (a, b) untouched, so it fixes contrast under dim or harsh light WITHOUT shifting hue.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# ==========================================================================================
# 3 · INFERENCE  (features computed identically to generate_arbiter_dataset.py)
# ==========================================================================================
def expert_features(model, bgr: np.ndarray, imgsz: int) -> dict:
    """Top-5 conf, entropy, margin from one expert.

    Two robustness measures:
      - predict() letterboxes internally, so the result is resolution independent.
      - test-time averaging: we average the softmax of the raw ROI and a CLAHE-enhanced copy.
        Averaging two hue-preserving views is more stable than betting on a single transform,
        and guarantees a bad enhancement can never dominate the decision.
    """
    names = None
    acc = None
    for view in (bgr, clahe_luma(bgr)):
        r = model.predict(view, imgsz=imgsz, verbose=False)[0]
        names = r.names
        p = r.probs.data.detach().cpu().numpy().astype(np.float64)
        acc = p if acc is None else acc + p
    full = acc / 2.0
    full = full / (full.sum() + 1e-12)
    order = np.argsort(-full)[:5]
    conf = [float(full[i]) for i in order]
    while len(conf) < 5:
        conf.append(0.0)
    fc = np.clip(full, 1e-12, 1.0)
    entropy = float(-(fc * np.log(fc)).sum())
    margin = float(conf[0] - conf[1])
    label = names[int(order[0])]
    return {"label": label, "conf": conf, "entropy": entropy, "margin": margin,
            "top5": [(names[int(i)], float(full[i])) for i in order]}


def build_feature_row(g: dict, i: dict) -> pd.DataFrame:
    row = {}
    for k in range(5):
        row[f"g_conf{k+1}"] = g["conf"][k]
        row[f"i_conf{k+1}"] = i["conf"][k]
    row["g_entropy"], row["g_margin"] = g["entropy"], g["margin"]
    row["i_entropy"], row["i_margin"] = i["entropy"], i["margin"]
    row["conf_gap"] = g["conf"][0] - i["conf"][0]
    row["conf_ratio"] = g["conf"][0] / (i["conf"][0] + 1e-6)
    row["entropy_gap"] = i["entropy"] - g["entropy"]
    row["margin_gap"] = g["margin"] - i["margin"]
    row["both_unsure"] = int((g["conf"][0] < 0.5) and (i["conf"][0] < 0.5))
    return pd.DataFrame([row], columns=FEATS)


def predict(bgr: np.ndarray) -> dict:
    """Full edge decision. Returns everything the UI needs to explain itself."""
    roi = center_roi(bgr)
    food = tighten_to_food(roi)          # crop to the food object, drop scale/table/hand clutter
    g = expert_features(model_g, food, GLOBAL_IMGSZ)
    i = expert_features(model_i, food, ISRAELI_IMGSZ)

    X = build_feature_row(g, i)
    p_israeli = float(arbiter.predict_proba(X)[0, 1])
    route_israeli = p_israeli >= 0.5

    chosen = i if route_israeli else g
    gate = GATE_ISRAELI if route_israeli else GATE_GLOBAL
    both_unsure = (g["conf"][0] < BOTH_UNSURE) and (i["conf"][0] < BOTH_UNSURE)
    confident = (chosen["conf"][0] >= gate) and not both_unsure

    return {"global": g, "israeli": i, "p_israeli": p_israeli, "route_israeli": route_israeli,
            "chosen": chosen, "label": chosen["label"], "conf": chosen["conf"][0],
            "confident": confident, "both_unsure": both_unsure, "roi": roi, "food": food}


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
def nutrition(label: str, grams: int) -> dict:
    key = label.lower().replace(" ", "_")
    ratio = max(grams, 0) / 100.0
    if key in LOCAL_DB:
        desc, cal, pro, carb, fat = LOCAL_DB[key]
        src = "local DB"
    else:
        cal = pro = carb = fat = 0
        desc = label.replace("_", " ").title()
        src = "USDA"
        if requests and USDA_API_KEY:
            try:
                r = requests.get("https://api.nal.usda.gov/fdc/v1/foods/search",
                                 params={"api_key": USDA_API_KEY, "query": desc, "pageSize": 5,
                                         "dataType": ["Foundation", "SR Legacy"]}, timeout=8)
                foods = r.json().get("foods", [])
                if foods:
                    desc = foods[0]["description"]
                    ids = {1008: "cal", 2047: "cal", 1003: "pro", 1005: "carb", 1004: "fat"}
                    got = {}
                    for n in foods[0]["foodNutrients"]:
                        nm = ids.get(n.get("nutrientId"))
                        if nm and nm not in got:
                            got[nm] = n.get("value", 0)
                    cal, pro, carb, fat = got.get("cal", 0), got.get("pro", 0), got.get("carb", 0), got.get("fat", 0)
            except Exception as e:
                print(f"[WARN] USDA failed: {e}")
    return {"name": label.replace("_", " ").title(), "desc": desc, "src": src, "weight": grams,
            "cal": int(cal * ratio), "pro": round(pro * ratio, 1),
            "carb": round(carb * ratio, 1), "fat": round(fat * ratio, 1)}


# ==========================================================================================
# 5 · BLE WEIGHT  (protocol recovered by reverse engineering; see scripts/ble/scale_reader.py)
# ==========================================================================================
weight_buffer = deque([0] * MAX_HISTORY, maxlen=MAX_HISTORY)
current_weight = 0
manual_weight = 0          # used when the BLE scale is not connected
sticky_high = 0
ble_status = "no BLE" if not _BLE_OK else "scanning"
stable = False


def get_weight() -> int:
    """Live BLE weight when connected, otherwise the value typed in the UI."""
    return int(current_weight) if ble_status == "connected" else int(manual_weight)


def on_notify(_sender, data):
    global current_weight, sticky_high, ble_status, stable
    b = list(data)
    if len(b) < 8:
        return
    low = b[4]
    high = b[5] | (b[3] & 0xFE)
    if high > 0:
        sticky_high = high
    if low < 5 and high == 0:
        sticky_high = 0
    w = low + max(high, sticky_high) * 256
    current_weight = w
    weight_buffer.append(w)
    s = list(weight_buffer)[-10:]
    stable = (max(s) - min(s)) <= 2 and len(s) == 10
    ble_status = "connected"


async def _ble_loop():
    global ble_status
    while True:
        try:
            dev = await BleakScanner.find_device_by_name("SWAN")
            if dev:
                async with BleakClient(dev) as c:
                    await c.start_notify(NOTIFY_UUID, on_notify)
                    while c.is_connected:
                        await asyncio.sleep(1)
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
class CalEyeZDemo:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.running = True
        self.frame_bgr = None
        self.last = None
        root.title("CalEyeZ  ·  Live Demo")
        root.geometry("1240x820")
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self._init_log()
        self._style()
        self._build_left(root)
        self._build_right(root)

        self.cap = cv2.VideoCapture(0)
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
        self.route_lbl.pack(anchor="w", padx=12, pady=(2, 10))

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

        self.btn_save = tk.Button(right, text="SAVE TO LOG", bg=CARD, fg=GREEN,
                                  font=("Segoe UI", 10, "bold"), bd=0, state="disabled", command=self.save)
        self.btn_save.pack(fill="x", padx=24, pady=10)

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
            h, w = frame.shape[:2]
            s = int(min(h, w) * ROI_FRAC)
            y0, x0 = (h - s) // 2, (w - s) // 2
            cv2.rectangle(frame, (x0, y0), (x0 + s, y0 + s), (204, 255, 0), 2)
            cv2.putText(frame, "PLACE FOOD HERE", (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (204, 255, 0), 2)
            # live "detected food" box: shows exactly what will be classified (throttled)
            self._fc = getattr(self, "_fc", 0) + 1
            if self._fc % 3 == 0:
                self._food_box = food_bbox(self.frame_bgr[y0:y0 + s, x0:x0 + s])
            bb = getattr(self, "_food_box", None)
            if bb is not None:
                fx0, fy0, fx1, fy1 = bb
                cv2.rectangle(frame, (x0 + fx0, y0 + fy0), (x0 + fx1, y0 + fy1), (255, 200, 0), 2)
                cv2.putText(frame, "food", (x0 + fx0, y0 + fy0 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((680, 510), Image.Resampling.LANCZOS)
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

    # ---------- analyze ----------
    def analyze(self):
        if self.frame_bgr is None or model_g is None:
            messagebox.showwarning("Not ready", "Camera or models not ready.")
            return
        self.btn.config(state="disabled", text="THINKING...", bg=MUT)
        self.result.config(text="Running edge inference...")
        threading.Thread(target=self._analyze_worker,
                         args=(self.frame_bgr.copy(), get_weight(), self.prep.get()),
                         daemon=True).start()

    def _analyze_worker(self, bgr, grams, prep):
        try:
            d = predict(bgr)
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
            self.src_chip.config(text="source: Gemini fallback (experts unsure)", fg=PURPLE)
        elif d["both_unsure"]:
            self.src_chip.config(text="experts unsure", fg=AMBER)
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
