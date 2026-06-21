"""
Torch-free ONNX inference backend for the CalEyeZ edge build.

Replaces the ultralytics/torch YOLO classifier with onnxruntime so the packaged exe needs
NO PyTorch (far smaller + faster on a CPU laptop). Preprocessing replicates ultralytics'
classify_transforms EXACTLY (verified to 0% top-1 mismatch vs the .pt on 200 test images):
    BGR -> RGB -> Resize(shortest edge -> imgsz, PIL BILINEAR) -> CenterCrop(imgsz) -> /255.
The exported classify head already applies softmax, so the ONNX output is class probabilities.

`OnnxExpert.features(bgr, imgsz)` returns the SAME dict shape as the demo's expert_features(),
so the XGBoost arbiter and the rest of the pipeline are untouched.
"""
import json
import numpy as np
from PIL import Image
import cv2
import onnxruntime as ort


def preprocess(bgr, size):
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    w, h = im.size
    if w <= h:
        nw, nh = size, int(size * h / w)        # floor, exactly like torchvision Resize(scalar)
    else:
        nh, nw = size, int(size * w / h)
    im = im.resize((nw, nh), Image.BILINEAR)
    left, top = (nw - size) // 2, (nh - size) // 2
    im = im.crop((left, top, left + size, top + size))
    x = np.asarray(im, dtype=np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None])   # (1,3,size,size)


class OnnxExpert:
    """One classification expert backed by an ONNX model. Drop-in for an ultralytics YOLO model."""

    def __init__(self, onnx_path, names_path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 0          # 0 = let ORT use all CPU cores
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        with open(names_path, encoding="utf-8") as f:
            self.names = {int(k): v for k, v in json.load(f).items()}
        self._bg_idx = next((k for k, v in self.names.items() if v == "background"), None)

    def features(self, bgr, imgsz):
        out = self.sess.run(None, {self.inp: preprocess(bgr, imgsz)})[0][0]
        p = out.astype(np.float64)
        if abs(float(p.sum()) - 1.0) > 1e-3:                     # safety: softmax if the head didn't
            e = np.exp(p - p.max()); p = e / e.sum()
        order = np.argsort(-p)[:5]
        conf = [float(p[i]) for i in order]
        while len(conf) < 5:
            conf.append(0.0)
        full = np.clip(p, 1e-12, 1.0)
        entropy = float(-(full * np.log(full)).sum())
        margin = float(conf[0] - conf[1])
        p_background = float(full[self._bg_idx]) if self._bg_idx is not None else 0.0
        return {"label": self.names[int(order[0])], "conf": conf, "entropy": entropy,
                "margin": margin, "p_background": p_background,
                "top5": [(self.names[int(i)], float(c)) for i, c in zip(order, conf)]}
