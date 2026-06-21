"""
Export the two YOLO11-cls models to ONNX and verify the ONNX outputs match the
PyTorch (.pt) outputs, using a torch-free preprocessing that replicates exactly what
ultralytics' classify_transforms does:  Resize(shortest edge -> size, bilinear) ->
CenterCrop(size) -> /255 (mean 0, std 1).  The exported ONNX classify head already
applies softmax, so its output is class probabilities.

Run:  py -3 scripts/edge/onnx_export_and_check.py
Outputs: build_edge_onnx/models/{global,israeli}.onnx  + *_names.json
"""
import os, json, glob, shutil
import numpy as np, cv2
from ultralytics import YOLO
import onnxruntime as ort

MODELS = {
    "global":  ("runs/general_model_flattened/weights/best.pt", 320),
    "israeli": ("runs/israeli_food_yolo11l_v2/weights/best.pt", 224),
}
OUT = "build_edge_onnx/models"
os.makedirs(OUT, exist_ok=True)


def preprocess(bgr, size):
    """torch-free replica of ultralytics classify inference preprocessing."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if h <= w:
        nh, nw = size, max(size, round(w * size / h))
    else:
        nw, nh = size, max(size, round(h * size / w))
    interp = cv2.INTER_AREA if (nh < h or nw < w) else cv2.INTER_LINEAR
    r = cv2.resize(rgb, (nw, nh), interpolation=interp)
    y0, x0 = (nh - size) // 2, (nw - size) // 2
    c = r[y0:y0 + size, x0:x0 + size]
    x = c.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    return np.ascontiguousarray(x)


imgs = sorted(glob.glob("assets/field/**/*.jpg", recursive=True))[:25]
print(f"parity images: {len(imgs)}")

summary = {}
for name, (pt, size) in MODELS.items():
    print(f"\n===== {name}  (imgsz {size}) =====")
    m = YOLO(pt)
    m.export(format="onnx", imgsz=size, opset=12, simplify=True, dynamic=False)
    exported = pt.replace(".pt", ".onnx")
    onnx_path = os.path.join(OUT, f"{name}.onnx")
    shutil.move(exported, onnx_path)
    json.dump({int(k): v for k, v in m.names.items()}, open(os.path.join(OUT, f"{name}_names.json"), "w"))
    print(f"  saved {onnx_path}  ({os.path.getsize(onnx_path)/1e6:.1f} MB)")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    maxdiff, mismatch, n = 0.0, 0, 0
    for ip in imgs:
        bgr = cv2.imread(ip)
        if bgr is None:
            continue
        pt_probs = m.predict(bgr, imgsz=size, verbose=False)[0].probs.data.cpu().numpy()
        out = sess.run(None, {inp: preprocess(bgr, size)})[0][0]
        if abs(float(out.sum()) - 1.0) > 0.01:        # apply softmax if the head didn't
            e = np.exp(out - out.max()); out = e / e.sum()
        maxdiff = max(maxdiff, float(np.abs(pt_probs - out).max()))
        mismatch += int(pt_probs.argmax() != out.argmax())
        n += 1
    summary[name] = (n, mismatch, maxdiff)
    print(f"  PARITY: n={n}  top-1 mismatches={mismatch}  max|Δprob|={maxdiff:.5f}")

print("\n===== SUMMARY =====")
ok = True
for name, (n, mm, md) in summary.items():
    verdict = "OK" if (mm == 0 and md < 0.02) else "CHECK"
    if verdict != "OK": ok = False
    print(f"  {name:8s}: {mm} mismatches / {n},  max|Δprob|={md:.5f}  -> {verdict}")
print("\nALL GOOD - ONNX matches PyTorch." if ok else "\nMismatch - preprocessing needs a tweak.")
