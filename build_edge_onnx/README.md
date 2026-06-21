# CalEyeZ — ONNX edge build (torch-free, CPU-only)

A standalone Windows app that runs the full CalEyeZ pipeline on a **basic laptop (8 GB RAM, any CPU,
no GPU)** using **ONNX Runtime** instead of PyTorch. The two YOLO11-cls vision experts are exported to
ONNX; the XGBoost arbiter and the rest of the app are unchanged. The result is a much smaller, faster,
dependency-light executable than the PyTorch edge build.

## Why ONNX for the edge

| | PyTorch edge build (`build_edge/`) | **ONNX edge build (this folder)** |
|---|---|---|
| Runtime | PyTorch (CPU) + ultralytics | onnxruntime only (no torch) |
| Accuracy | baseline | **identical** — verified 0% top-1 mismatch vs the `.pt` on 200 test images; full-pipeline routing agrees on every image (max ΔP(israeli) = 0.013) |
| Latency / Analyze | ~1–3 s on CPU | **~0.35 s** measured here; ~0.7–1.4 s on a basic AMD CPU |
| Bundle size | very large — multiple GB (CUDA/MKL DLLs) | **~0.8 GB** folder (onnxruntime + the two ~50 MB ONNX models + deps) |
| RAM | higher | lower — comfortable in 8 GB |

The classify head is exported **with its softmax**, so the ONNX output is class probabilities. Preprocessing
is replicated torch-free in `scripts/demo/onnx_backend.py` exactly as ultralytics does it:
`BGR → RGB → Resize(shortest edge → imgsz, PIL bilinear) → CenterCrop(imgsz) → /255`.

## How it works

- `scripts/demo/onnx_backend.py` — `OnnxExpert`, a drop-in replacement for an ultralytics YOLO classifier
  that returns the **same feature dict** (top-5 confidences, entropy, margin, `i_p_background`) the arbiter
  consumes. So routing and the whole decision pipeline are untouched.
- `scripts/demo/caleyez_demo.py` — selects the backend at startup:
  - `CALEYEZ_ONNX=1` (or no PyTorch installed) → ONNX backend.
  - otherwise → the PyTorch/ultralytics backend (GPU dev).
  The packaged exe sets `CALEYEZ_ONNX=1` automatically (runtime hook).
- Data tagging still saves the labelled image + verification row; the 512-D "DNA" embedding is unavailable
  in ONNX (the exported graph only exposes the classifier output) and is stored as zeros.

## Build it

1. **Generate the ONNX models** (once), with a parity check vs the `.pt`:
   ```powershell
   py -3 scripts\edge\onnx_export_and_check.py
   ```
   → `build_edge_onnx\models\{global,israeli}.onnx` + `*_names.json`.

2. **Build the exe**:
   ```powershell
   cd build_edge_onnx
   .\build_onnx.ps1
   ```
   → `build_edge_onnx\dist\CalEyeZ\CalEyeZ.exe`. Zip the whole `CalEyeZ` folder to distribute.

## Run on the laptop

Copy the **whole `dist\CalEyeZ` folder** (exe + `_internal` + the bundled `.onnx` models) and double-click
`CalEyeZ.exe`. You want to see `[SYSTEM] loading models (ONNX backend)...` then
`[SYSTEM] models + arbiter ready.` in the console.

- Live camera preview is smooth; ANALYZE is ~0.5–1.5 s on a CPU laptop.
- Weight via the BLE scale (SWAN) if Bluetooth is available, else type grams manually.

### API keys (USDA + Gemini) are NOT embedded
The packaged exe ships **no secrets** (frozen build → the dev keys are blanked). If the nutrition values
come up **empty**, it's because no key is set: set env vars `USDA_API_KEY` / `GEMINI_API_KEY`, or drop a
`caleyez_keys.txt` **in the same folder as `CalEyeZ.exe`**:
```
USDA_API_KEY=your_usda_key
GEMINI_API_KEY=your_gemini_key
```
Then restart the exe (keys are read once at startup). Without keys the small built-in table still gives
macros for a few common foods, but anything else shows blank. The laptop needs **internet** for the lookups.
(Watch out for Notepad saving `caleyez_keys.txt.txt` — enable "File name extensions" in Explorer to verify.)

### Choosing the camera (integrated vs external USB)
The app scans camera indices 0–3 and uses the first that delivers frames — usually the **integrated**
camera (index 0). To use an external **USB webcam**:
- **In the app:** click **`⟳ Switch camera`** under the video (or press **`c`**) to cycle to the next webcam.
  The label shows the active `Camera #n`.
- **Force an index:** set `CALEYEZ_CAM=1` (try `2` if needed) before launching, **or** drop a file
  `caleyez_camera.txt` next to `CalEyeZ.exe` containing just the number (e.g. `1`).
- **Last resort (no app change):** disable the integrated camera in Windows Device Manager → Cameras.

## Notes / trade-offs

- The models are the **Large** `yolo11l-cls` variant. If your specific CPU is too slow even on ONNX:
  - **INT8 dynamic quantization** of the ONNX models (2–4× smaller, often faster, ~0.5–2% accuracy cost,
    no retrain), or
  - retrain on a smaller backbone (`yolo11s/n-cls`) for a larger speed-up at a few points of accuracy.
- Keep `console=True` in `caleyez_onnx.spec` for the first runs (shows logs); flip to `False` once verified.
