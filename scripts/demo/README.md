# CalEyeZ Live Demo

End-to-end application: webcam frame to food class to BLE weight to full nutrition breakdown.
Runs the current Global + Israeli models behind the XGBoost arbiter, with a Gemini vision
fallback for inputs the local experts cannot win. See the **Live Demo** tab in `index.html`
for the design rationale.

## What it does

1. Captures a webcam frame at any resolution and crops a fixed center ROI ("place food here").
2. Normalizes lighting (gray-world white balance + CLAHE) so the result is stable under any bulb.
3. Runs the Global model (132 classes, imgsz 320) and Israeli model (13 classes, imgsz 224)
   through Ultralytics `predict()` (letterboxed, so it is resolution independent).
4. Builds the 19-feature vector and asks the arbiter `predict_proba` which expert to trust.
5. If the chosen expert is weak, or both experts are unsure, falls back to the Gemini vision API.
6. Reads the weight from the BLE scale and scales nutrition (local DB first, else USDA).

## Requirements

- A working webcam (OpenCV device 0).
- The SWAN BLE scale powered on and nearby (optional: the app runs without it, weight stays 0).
- The trained weights and arbiter present in the repo:
  - `runs/general_model_flattened/weights/best.pt`
  - `runs/israeli_food_yolo11l/weights/best.pt`
  - `scripts/arbiter/arbiter_xgb.json`

## Install

```bash
pip install ultralytics torch opencv-python pillow xgboost requests google-generativeai bleak matplotlib numpy pandas
```

## API keys (read from the environment, never hard-coded)

Get a free USDA key at https://fdc.nal.usda.gov/api-key-signup.html and a Gemini key at
https://aistudio.google.com/app/apikey.

Windows (PowerShell):

```powershell
$env:USDA_API_KEY   = "your_usda_key"
$env:GEMINI_API_KEY = "your_gemini_key"
```

macOS / Linux (bash):

```bash
export USDA_API_KEY="your_usda_key"
export GEMINI_API_KEY="your_gemini_key"
```

Both keys are optional. Without USDA, only the local nutrition DB is used; without Gemini, the
low-confidence fallback is skipped and the local prediction is kept.

## Run

From the repo root, so the model paths resolve:

```bash
python scripts/demo/caleyez_demo.py
```

## Using it

- Wait for the console to print `models + arbiter ready` and the BLE chip to read `connected`.
- Place the food inside the green box. Capture when the weight readout turns teal (stable).
- Press **Analyze**. The decision panel shows both experts' top guess and the live P(israeli),
  so you can see the arbiter route. A chip states the source (Global / Israeli expert, or Gemini
  fallback).
- Press **Save to log** to add the meal. The day's table and macro totals update in place;
  rows are colour-coded by which expert produced them (blue global, teal Israeli).

## If the scale shows "BLE off" or stays at 0 g

The weight comes from `bleak`. If the chip reads `BLE off (pip install bleak)`, then `bleak` is not
installed in the **same** Python environment you launched the demo from (it can be present in another
venv and still be missing here). Fix:

```bash
pip install bleak
```

The demo never blocks on this: when the scale is not connected it automatically uses the **manual g**
field next to the weight readout, so you can type the grams and continue. The chip always states which
source is live (BLE or manual).

## Camera angle matters (important for the demo)

The models were trained almost entirely on **side / eye-level** photos of food. A steep **top-down**
view is out of distribution: a round red bell pepper seen from directly above (a red disc with a green
stem) does not look like the side-view peppers in the training set, so it can be misread. This is not a
crop bug; the model scores about 99.97% on the same pepper from the side, on any background.

For the demo:

- Angle the camera closer to **45 degrees or eye level**, not straight down, so the food matches the
  training distribution.
- Set `GEMINI_API_KEY`. When the local experts are not confident (the app shows a yellow
  "LOW CONFIDENCE" chip), the ROI is sent to Gemini, which handles the unusual views the local models
  were never trained on. This is the intended path for out-of-distribution inputs.

## Edge build — run on a basic laptop (no GPU), torch-free via ONNX

The same app runs on a low-end laptop (8 GB RAM, basic CPU, no GPU) using **ONNX Runtime** instead of
PyTorch. Both YOLO11-cls experts are exported to ONNX and served by a torch-free backend
(`scripts/demo/onnx_backend.py`) that feeds the arbiter the **identical** features, so routing is unchanged
(verified: 0% top-1 mismatch vs the `.pt` on 200 test images/model; full-pipeline routing agrees on every
image). Measured ~0.35 s per analysis on CPU.

- Backend selection: set `CALEYEZ_ONNX=1` to force the ONNX backend (the app also auto-falls back to ONNX
  if PyTorch isn't installed). Otherwise it uses PyTorch/ultralytics (GPU dev).
- Build a standalone exe: see `build_edge_onnx/` (`build_onnx.ps1`, `caleyez_onnx.spec`, `README.md`).
- Export the ONNX models first: `py -3 scripts/edge/onnx_export_and_check.py`.

## Choosing the camera (integrated vs external USB)

The app scans camera indices 0–3 and uses the first that delivers frames (usually the integrated camera).
To use an external **USB webcam**:

- Click **`⟳ Switch camera`** under the video (or press **`c`**) to cycle to the next webcam; the label shows
  the active `Camera #n`.
- Or force an index: `CALEYEZ_CAM=1` (try `2`), **or** a `caleyez_camera.txt` next to the exe with just the
  number.
- Or disable the integrated camera in Windows Device Manager → Cameras.

## Nutrition values are empty?

That means no API key is set (the packaged exe ships no secrets). Set `USDA_API_KEY` (and optionally
`GEMINI_API_KEY`) in the environment, **or** drop a `caleyez_keys.txt` next to the exe:

```
USDA_API_KEY=your_usda_key
GEMINI_API_KEY=your_gemini_key
```

Restart afterwards (keys are read once at startup). Without a key, only the small built-in food table
returns values; the laptop also needs internet for the lookups.

## Notes

- Inference runs on a worker thread, so the preview and weight never freeze.
- Lighting handling is CLAHE (contrast only) plus test-time averaging of the raw and CLAHE views.
  We deliberately do NOT use gray-world white balance, which distorts the hue of dominant-colour foods.
- `nutrition_log.csv` and `temp.jpg` are written at the repo root and are gitignored.
- First run downloads nothing extra if the weights are present; model load takes a few seconds.
