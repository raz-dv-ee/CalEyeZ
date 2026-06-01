"""Test-set confusion analysis for the flattened Global model.
Prints overall acc, worst classes by recall, and the top confused (true->pred) pairs.
"""
from pathlib import Path
from collections import defaultdict
import numpy as np
from ultralytics import YOLO

BEST = r"runs\general_model_flattened\weights\best.pt"
TEST = Path(r"datasets\general_model_flattened\test")

m = YOLO(BEST)
names = m.names  # idx -> class name (training order)

imgs, truths = [], []
for cls_dir in sorted(p for p in TEST.iterdir() if p.is_dir()):
    for img in cls_dir.iterdir():
        imgs.append(str(img)); truths.append(cls_dir.name)

print(f"Predicting {len(imgs)} test images...")
conf = defaultdict(int)                 # (true, pred) -> count
total = defaultdict(int); correct = defaultdict(int)

B = 512
for i in range(0, len(imgs), B):
    batch = imgs[i:i+B]
    res = m.predict(batch, imgsz=320, verbose=False, device=0)
    for r, t in zip(res, truths[i:i+B]):
        p = names[int(r.probs.top1)]
        total[t] += 1
        conf[(t, p)] += 1
        if p == t:
            correct[t] += 1

n_correct = sum(correct.values()); n_total = sum(total.values())
print(f"\n=== OVERALL test top-1: {n_correct/n_total:.4f}  ({n_correct}/{n_total}) ===")

recall = {c: correct[c]/total[c] for c in total}
print("\n=== 15 WORST classes by recall ===")
print(f"{'class':<24}{'recall':>8}{'n':>6}  top confusion")
for c in sorted(recall, key=recall.get)[:15]:
    # most common wrong prediction for this true class
    wrong = [((t,p),n) for (t,p),n in conf.items() if t==c and p!=c]
    top = max(wrong, key=lambda x: x[1]) if wrong else None
    tip = f"-> {top[0][1]} ({top[1]})" if top else ""
    print(f"{c:<24}{recall[c]*100:>7.1f}%{total[c]:>6}  {tip}")

print("\n=== Top 20 confused (true -> pred) pairs ===")
pairs = [((t,p),n) for (t,p),n in conf.items() if t!=p]
for (t,p),n in sorted(pairs, key=lambda x:-x[1])[:20]:
    print(f"  {t:<22} -> {p:<22} {n:>4}  ({n/total[t]*100:.0f}% of {t})")
