"""
Export a feature-visualisation ONNX that exposes the output of every top-level block of the
Global YOLO11l-cls model, so the report's CNN Visualizer can show the REAL activations of all
11 blocks on a user-uploaded photo (run in-browser via onnxruntime-web).

The normal export only returns the final probabilities. Here we wrap the backbone so it returns
every block's feature map (blk0..blk9) plus the final 132-class softmax (probs). Weights are
converted to fp16 (fully, incl. IO) so the file is ~26 MB and the browser feeds a float16 input.

Run:  py -3 scripts/edge/export_features_onnx.py
Out:  webmodels/global_features_fp16.onnx   (served at CalEyeZ/webmodels/ via GitHub Pages)
"""
import os, warnings
import torch, torch.nn as nn
import onnx
from onnxconverter_common import float16
from ultralytics import YOLO
warnings.filterwarnings("ignore")

PT = "runs/general_model_flattened/weights/best.pt"
OUT = "webmodels/global_features_fp16.onnx"


class Feat(nn.Module):
    """Runs the backbone and returns every block output (blk0..blk9) + final probs."""
    def __init__(self, seq):
        super().__init__()
        self.seq = seq

    def forward(self, x):
        outs = []
        for b in self.seq:
            x = b(x)
            outs.append(x)
        return tuple(outs[0:10]) + (outs[10],)


def main():
    seq = YOLO(PT).model.model.eval()
    names = [f"blk{i}" for i in range(10)] + ["probs"]
    tmp = "webmodels/global_features.onnx"
    with torch.no_grad():
        torch.onnx.export(Feat(seq).eval(), torch.zeros(1, 3, 320, 320), tmp,
                          opset_version=12, input_names=["img"], output_names=names)
    m16 = float16.convert_float_to_float16(onnx.load(tmp), keep_io_types=False)
    onnx.save(m16, OUT)
    os.remove(tmp)
    print(f"wrote {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
