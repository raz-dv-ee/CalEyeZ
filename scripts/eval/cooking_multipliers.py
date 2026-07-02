#!/usr/bin/env python3
"""Derive cooking-method calorie multipliers EMPIRICALLY from USDA FoodData Central.

Motivation
----------
The web app lets the user tag how a food was prepared (grilled / pan-fried / deep-fried / boiled /
baked). Rather than invent "fried adds 50%" out of thin air, we measure it: for foods that USDA
carries in BOTH a raw and a cooked state, the per-100g kcal ratio cooked/raw is the multiplier.

Why per-100g cooked/raw is the correct statistic: frying drives off water (raising calorie density)
and absorbs oil. Because the CalEyeZ scale weighs the COOKED, plated food, the factor we need is
exactly cooked_kcal_per_100g / raw_kcal_per_100g. So the multiplier is self-consistent with how the
product measures weight.

Output
------
cooking_multipliers.json:
    { "<method>": {"factor": mean, "added_fat_g": mean, "n": k, "ci95": [lo, hi], "sd": s},
      "_per_food": [ {food, method, raw_kcal, cooked_kcal, ratio, ...}, ... ],
      "_meta": {...} }

Run
---
    USDA_API_KEY=... python scripts/eval/cooking_multipliers.py
(or drop the key in build_edge_onnx/dist/CalEyeZ/caleyez_keys.txt as USDA_API_KEY=...)
"""
import os, sys, json, time, statistics as st, pathlib
import requests

# ---- key (env first, then the local git-ignored keyfile) ----------------------------------------
def _load_key():
    k = os.environ.get("USDA_API_KEY")
    if k:
        return k
    for p in [pathlib.Path("build_edge_onnx/dist/CalEyeZ/caleyez_keys.txt"),
              pathlib.Path("caleyez_keys.txt")]:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.startswith("USDA_API_KEY="):
                    return line.split("=", 1)[1].strip()
    sys.exit("No USDA_API_KEY (env or caleyez_keys.txt).")

KEY = _load_key()
SEARCH = "https://api.nal.usda.gov/fdc/v1/foods/search"
ENERGY = {1008, 2047, 2048}

# Curated raw<->cooked pairs. Each entry names the food and a "must" keyword the matched USDA
# description MUST contain (USDA search is fuzzy: "shrimp ... moist heat" otherwise returns crab).
# Tiers:  grilled = dry heat, no added oil (roasted/broiled/baked);  fried = shallow oil, no batter;
#         deep_fried = submerged / battered;  boiled = moist heat (also steamed).
FOODS = [
    {"food": "potato", "must": "potato", "raw": "Potatoes, flesh and skin, raw",
     "grilled": "Potatoes, baked, flesh and skin",
     "boiled": "Potatoes, boiled, cooked in skin, flesh",
     "deep_fried": "Potatoes, french fried, all types, salt not added"},
    {"food": "chicken breast", "must": "chicken", "raw": "Chicken, broilers or fryers, breast, meat only, raw",
     "grilled": "Chicken, broilers or fryers, breast, meat only, roasted",
     "fried": "Chicken, broilers or fryers, breast, meat only, cooked, fried",
     "deep_fried": "Chicken, broilers or fryers, breast, meat and skin, batter, fried"},
    {"food": "beef", "must": "beef", "raw": "Beef, loin, top sirloin steak, boneless, separable lean only, raw",
     "grilled": "Beef, loin, top sirloin cap steak, boneless, separable lean only, grilled"},
    {"food": "egg", "must": "egg", "raw": "Egg, whole, raw, fresh",
     "fried": "Egg, whole, cooked, fried"},
    {"food": "salmon", "must": "salmon", "raw": "Fish, salmon, Atlantic, farmed, raw",
     "grilled": "Fish, salmon, Atlantic, farmed, cooked, dry heat"},
    {"food": "catfish", "must": "catfish", "raw": "Fish, catfish, channel, farmed, raw",
     "deep_fried": "Fish, catfish, channel, cooked, breaded and fried"},
    {"food": "plantain", "must": "plantain", "raw": "Plantains, yellow, raw",
     "deep_fried": "Plantains, yellow, fried"},
    {"food": "shrimp", "must": "shrimp", "raw": "Crustaceans, shrimp, raw",
     "boiled": "Crustaceans, shrimp, cooked, moist heat"},
    {"food": "carrot", "must": "carrot", "raw": "Carrots, raw",
     "boiled": "Carrots, cooked, boiled, drained, without salt"},
    {"food": "broccoli", "must": "broccoli", "raw": "Broccoli, raw",
     "boiled": "Broccoli, cooked, boiled, drained, without salt"},
    {"food": "zucchini", "must": "zucchini", "raw": "Squash, summer, zucchini, includes skin, raw",
     "grilled": "Squash, summer, green, cooked, boiled, drained, without salt"},
]

def lookup(query, must=""):
    """Return (desc, kcal, fat) per 100 g for the first Foundation/SR-Legacy hit whose description
    contains `must`, or None (guards against USDA's fuzzy search returning the wrong food)."""
    r = requests.get(SEARCH, params={"api_key": KEY, "query": query, "pageSize": 5,
                                     "dataType": ["Foundation", "SR Legacy"]}, timeout=15)
    r.raise_for_status()
    foods = [x for x in r.json().get("foods", []) if must.lower() in x.get("description", "").lower()]
    if not foods:
        return None
    f = foods[0]
    kcal = fat = None
    for n in f.get("foodNutrients", []):
        nid = n.get("nutrientId")
        if nid in ENERGY and kcal is None:
            kcal = n.get("value")
        if nid == 1004 and fat is None:
            fat = n.get("value")
    if kcal is None:
        return None
    return f.get("description", query), kcal, (fat or 0.0)

def ci95(vals):
    if len(vals) < 2:
        return [round(vals[0], 3), round(vals[0], 3)] if vals else [None, None]
    m, sd = st.mean(vals), st.stdev(vals)
    half = 1.96 * sd / (len(vals) ** 0.5)
    return [round(m - half, 3), round(m + half, 3)]

def main():
    per_food, by_method = [], {}
    for spec in FOODS:
        raw = lookup(spec["raw"], spec["must"]); time.sleep(0.2)
        if not raw:
            print(f"[skip] no raw match for {spec['food']!r} ({spec['raw']})")
            continue
        raw_desc, raw_kcal, raw_fat = raw
        for method, q in spec.items():
            if method in ("food", "raw", "must"):
                continue
            hit = lookup(q, spec["must"]); time.sleep(0.2)
            if not hit:
                print(f"[skip] no {method} match for {spec['food']!r} ({q})")
                continue
            c_desc, c_kcal, c_fat = hit
            ratio = c_kcal / raw_kcal
            row = {"food": spec["food"], "method": method,
                   "raw_desc": raw_desc, "raw_kcal": raw_kcal,
                   "cooked_desc": c_desc, "cooked_kcal": c_kcal,
                   "ratio": round(ratio, 3), "added_fat_g": round(c_fat - raw_fat, 2)}
            per_food.append(row)
            by_method.setdefault(method, {"ratios": [], "fats": []})
            by_method[method]["ratios"].append(ratio)
            by_method[method]["fats"].append(c_fat - raw_fat)
            print(f"  {spec['food']:<14} {method:<11} raw {raw_kcal:>4.0f} -> "
                  f"cooked {c_kcal:>4.0f}  x{ratio:4.2f}  (+{c_fat - raw_fat:5.1f} g fat)  [{c_desc[:42]}]")

    out = {"_meta": {"source": "USDA FoodData Central (Foundation + SR Legacy)",
                     "basis": "per-100g cooked/raw kcal ratio; scale weighs cooked food",
                     "pairs": len(per_food)}}
    for method, d in sorted(by_method.items()):
        ratios = d["ratios"]
        out[method] = {"factor": round(st.mean(ratios), 3),
                       "added_fat_g": round(st.mean(d["fats"]), 2),
                       "n": len(ratios), "sd": round(st.stdev(ratios), 3) if len(ratios) > 1 else 0.0,
                       "ci95": ci95(ratios)}
    out["raw"] = {"factor": 1.0, "added_fat_g": 0.0, "n": len(FOODS), "sd": 0.0, "ci95": [1.0, 1.0]}
    out["_per_food"] = per_food

    dest = pathlib.Path("scripts/eval/cooking_multipliers.json")
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nMULTIPLIERS")
    for m in ("raw", "boiled", "grilled", "fried", "deep_fried"):
        if m in out:
            e = out[m]
            print(f"  {m:<11} x{e['factor']:<5} +{e['added_fat_g']:>4} g fat   "
                  f"n={e['n']}  CI95 {e['ci95']}")
    print(f"\nwrote {dest}")

if __name__ == "__main__":
    main()
