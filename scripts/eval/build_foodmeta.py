#!/usr/bin/env python3
"""Generate foodmeta.json for the web app: per-class cooking-method applicability + curated sub-foods.

Design rule ("make sense"): a cooking-method dropdown appears ONLY for foods genuinely eaten in
multiple preparations - cookable single-ingredient vegetables and raw-or-cooked proteins. The vast
majority of the 145 classes are already-prepared DISHES (pizza is baked, falafel is deep-fried,
sushi is sushi), so they get prep: [] and no dropdown.

Sub-foods carry their own per-100g macros and REPLACE the base when picked (steak -> filet mignon).
Curated only where sub-types materially change nutrition (meats / fish / rice).

Reads the class lists from webmodels/, writes foodmeta.json (copied into caleyez-web).
"""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
g = json.load(open(ROOT / "webmodels/global_names.json"))
i = json.load(open(ROOT / "webmodels/israeli_names.json"))
CLASSES = sorted({*g.values(), *i.values()} - {"background"})

# --- prep applicability -------------------------------------------------------------------------
# Cookable single-ingredient vegetables: raw or boiled/grilled/fried.
VEG = {"beetroot", "cabbage", "carrot", "cauliflower", "corn", "eggplant", "onion", "peas",
       "spinach", "turnip", "bell_pepper"}
VEG_PREP = ["raw", "boiled", "grilled", "fried"]
# Single-ingredient proteins eaten raw (tartare/carpaccio) through cooked.
PROTEIN = {"steak", "scallops"}
PROTEIN_PREP = ["raw", "grilled", "fried", "deep_fried"]

def prep_for(cls):
    if cls in VEG:
        return VEG_PREP
    if cls in PROTEIN:
        return PROTEIN_PREP
    return []                      # prepared dishes / fruits / nuts: no method dropdown

# --- curated sub-foods (per-100g: kcal, pro, carb, fat) ------------------------------------------
SUBS = {
    "steak": {
        "filet mignon": {"kcal": 227, "pro": 24, "carb": 0, "fat": 15},
        "ribeye":       {"kcal": 291, "pro": 24, "carb": 0, "fat": 22},
        "sirloin":      {"kcal": 244, "pro": 26, "carb": 0, "fat": 15},
        "t-bone":       {"kcal": 247, "pro": 24, "carb": 0, "fat": 17},
        "flank":        {"kcal": 192, "pro": 28, "carb": 0, "fat": 8},
    },
    "sashimi": {
        "salmon":     {"kcal": 208, "pro": 20, "carb": 0, "fat": 13},
        "tuna":       {"kcal": 144, "pro": 23, "carb": 0, "fat": 5},
        "yellowtail": {"kcal": 146, "pro": 23, "carb": 0, "fat": 5},
        "sea bream":  {"kcal": 133, "pro": 20, "carb": 0, "fat": 5},
    },
    "sushi": {
        "salmon nigiri":   {"kcal": 150, "pro": 6, "carb": 30, "fat": 1},
        "tuna roll":       {"kcal": 130, "pro": 6, "carb": 28, "fat": 0.5},
        "california roll": {"kcal": 92,  "pro": 3, "carb": 18, "fat": 1.5},
        "eel (unagi)":     {"kcal": 184, "pro": 9, "carb": 26, "fat": 5},
    },
    "white_rice": {
        "white":   {"kcal": 130, "pro": 2.7, "carb": 28, "fat": 0.3},
        "brown":   {"kcal": 123, "pro": 2.7, "carb": 26, "fat": 1.0},
        "basmati": {"kcal": 121, "pro": 3.5, "carb": 25, "fat": 0.4},
        "jasmine": {"kcal": 129, "pro": 2.9, "carb": 28, "fat": 0.3},
    },
}

meta = {}
for cls in CLASSES:
    prep = prep_for(cls)
    subs = SUBS.get(cls, {})
    # Only emit entries that actually carry a control, plus a category hint for the UI.
    if prep or subs:
        meta[cls] = {"prep": prep, "subs": subs}

out = {"_meta": {"classes": len(CLASSES),
                 "with_prep": sum(1 for c in CLASSES if prep_for(c)),
                 "with_subs": len(SUBS),
                 "note": "classes not listed have no prep/sub controls (already-prepared dishes, "
                         "fruits, nuts)"},
       **meta}

dest = ROOT / "webmodels_meta" / "foodmeta.json"  # staging; copied into caleyez-web by the caller
dest.parent.mkdir(exist_ok=True)
dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"{len(CLASSES)} classes -> {out['_meta']['with_prep']} with prep, "
      f"{out['_meta']['with_subs']} with subs")
print("prep classes:", [c for c in CLASSES if prep_for(c)])
print("sub  classes:", list(SUBS))
print("wrote", dest)
