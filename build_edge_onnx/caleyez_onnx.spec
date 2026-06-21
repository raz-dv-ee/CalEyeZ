# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the CalEyeZ ONNX edge build (CPU, torch-free, onedir).
#   Build:   cd build_edge_onnx ; .\build_onnx.ps1
#   Output:  build_edge_onnx/dist/CalEyeZ/CalEyeZ.exe   (zip the CalEyeZ folder to distribute)
#
# Why this is far smaller/faster than the PyTorch build: it bundles onnxruntime + the two ONNX
# classifiers instead of PyTorch + ultralytics (which add several GB of CUDA/MKL DLLs). The runtime
# hook sets CALEYEZ_ONNX=1 so caleyez_demo.py uses scripts/demo/onnx_backend.py.
import os
from PyInstaller.utils.hooks import collect_all

REPO = os.path.abspath(os.path.join(SPECPATH, ".."))
datas, binaries, hiddenimports = [], [], []
for pkg in ("onnxruntime", "cv2", "PIL", "xgboost", "sklearn", "pandas", "numpy",
            "bleak", "google", "requests", "matplotlib"):
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
        print(f"[spec] collected {pkg}")
    except Exception as e:
        print(f"[spec] collect_all({pkg}) skipped: {e}")

# Bundle the ONNX models + class names + the arbiter at the paths caleyez_demo.py expects under ROOT.
datas += [
    (os.path.join(REPO, "build_edge_onnx", "models", "global.onnx"),        os.path.join("build_edge_onnx", "models")),
    (os.path.join(REPO, "build_edge_onnx", "models", "global_names.json"),   os.path.join("build_edge_onnx", "models")),
    (os.path.join(REPO, "build_edge_onnx", "models", "israeli.onnx"),        os.path.join("build_edge_onnx", "models")),
    (os.path.join(REPO, "build_edge_onnx", "models", "israeli_names.json"),  os.path.join("build_edge_onnx", "models")),
    (os.path.join(REPO, "scripts", "arbiter", "arbiter_xgb.json"),           os.path.join("scripts", "arbiter")),
]

a = Analysis(
    [os.path.join(REPO, "scripts", "demo", "caleyez_demo.py")],
    pathex=[REPO, os.path.join(REPO, "scripts", "demo")],   # so `from onnx_backend import ...` resolves
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + ["onnxruntime", "sklearn", "xgboost", "onnx_backend", "PIL._tkinter_finder"],
    runtime_hooks=[os.path.join(SPECPATH, "rthook_onnx.py")],
    excludes=["torch", "torchvision", "ultralytics"],       # keep the bundle torch-free + small
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="CalEyeZ",
          debug=False, strip=False, upx=False, console=True)   # console=True surfaces logs on first runs
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="CalEyeZ")
