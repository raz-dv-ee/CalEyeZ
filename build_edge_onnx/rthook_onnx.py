# PyInstaller runtime hook: force the torch-free ONNX backend in the packaged edge exe.
import os
os.environ.setdefault("CALEYEZ_ONNX", "1")
