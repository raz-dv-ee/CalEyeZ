"""
Export the trained XGBoost arbiter to the flat JSON the browser/edge walks.

Source : scripts/arbiter/arbiter_xgb.json   (XGBoost native, from train_arbiter_xgb.py)
Output : webmodels/arbiter_trees.json        (compact {base, trees:[{id:[f,thr,yes,no] | [leaf]}]})

The web app has no ONNX TreeEnsemble operator, so it evaluates the arbiter as a JS tree-walk.
XGBoost is an ADDITIVE model, so the walk is exact, not an approximation:

    margin(x) = base + sum_k f_k(x)          # base = logit(base_score); f_k = leaf reached in tree k
    P(israeli) = 1 / (1 + e^-margin)

Node format matches the JS in the web app:
    internal node -> [feature_index, threshold, yes_id, no_id]   ("go to yes if x[feature] < threshold")
    leaf node     -> [leaf_value]

Run:    py -3 scripts/arbiter/export_arbiter_trees.py
Verify: reproduces the committed webmodels/arbiter_trees.json AND matches clf.predict_proba to <1e-6.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

import sys
ROOT      = Path(__file__).resolve().parent.parent.parent
MODEL_IN  = ROOT / "scripts" / "arbiter" / "arbiter_xgb.json"
TREES_OUT = ROOT / "webmodels" / "arbiter_trees.json"
DATASET   = ROOT / "datasets" / "arbiter_dataset.csv"
# non-destructive by default: writes *.regen.json and verifies it reproduces the committed file.
# pass --write to overwrite the real webmodels/arbiter_trees.json.
WRITE_REAL = "--write" in sys.argv
DEST = TREES_OUT if WRITE_REAL else TREES_OUT.with_suffix(".regen.json")

# same 20 features, same order, as train_arbiter_xgb.py
FEATS = ([f"g_conf{k}" for k in range(1, 6)] + ["g_entropy", "g_margin"]
         + [f"i_conf{k}" for k in range(1, 6)] + ["i_entropy", "i_margin", "i_p_background"]
         + ["conf_gap", "conf_ratio", "entropy_gap", "margin_gap", "both_unsure"])
FIDX = {name: i for i, name in enumerate(FEATS)}


def flatten_tree(node: dict, out: dict) -> None:
    """Recursively turn one XGBoost get_dump JSON tree into {id: [f,thr,yes,no] | [leaf]}."""
    nid = str(node["nodeid"])
    if "leaf" in node:
        out[nid] = [float(node["leaf"])]
        return
    # split feature can be a name ("i_p_background") or an index ("f14")
    sp = node["split"]
    fidx = FIDX[sp] if sp in FIDX else int(str(sp).lstrip("f"))
    # store the threshold at float32 precision (XGBoost's internal precision)
    thr = float(np.float32(node["split_condition"]))
    out[nid] = [fidx, thr, int(node["yes"]), int(node["no"])]
    for child in node["children"]:
        flatten_tree(child, out)


def build_flat(model_path: Path) -> dict:
    clf = xgb.XGBClassifier()
    clf.load_model(str(model_path))
    booster = clf.get_booster()

    # base margin = logit(base_score)   (binary:logistic stores base_score in probability space)
    base_score = float(json.load(open(model_path))["learner"]["learner_model_param"]["base_score"])
    base = math.log(base_score / (1.0 - base_score))

    trees = []
    for tree_json in booster.get_dump(dump_format="json"):
        flat: dict = {}
        flatten_tree(json.loads(tree_json), flat)
        trees.append(flat)
    return {"base": base, "trees": trees}, clf


def walk(flat: dict, x) -> float:
    """Pure-python mirror of the web app's arbiterP(). Compares in float32, exactly as
    XGBoost does internally, so it matches predict_proba to ~1e-7 (float64 boundary-flips a few)."""
    m = flat["base"]
    for tree in flat["trees"]:
        n = tree["0"]
        while len(n) > 1:                              # internal: [f, thr, yes, no]
            n = tree[str(n[2] if np.float32(x[n[0]]) < n[1] else n[3])]
        m += n[0]                                      # leaf: [value]
    return 1.0 / (1.0 + math.exp(-m))


def trees_match(a: dict, b: dict, tol: float = 1e-6) -> bool:
    """Structural + numeric equality of two {base, trees} models (float-format tolerant)."""
    if abs(a["base"] - b["base"]) > 1e-12 or len(a["trees"]) != len(b["trees"]):
        return False
    for ta, tb in zip(a["trees"], b["trees"]):
        if set(ta) != set(tb):
            return False
        for k in ta:
            va, vb = ta[k], tb[k]
            if len(va) != len(vb) or any(abs(x - y) > tol for x, y in zip(va, vb)):
                return False
    return True


def main() -> None:
    flat, clf = build_flat(MODEL_IN)
    print(f"base = {flat['base']!r}   trees = {len(flat['trees'])}")

    # ---- verify 1: reproduce the committed file (structural, float-tolerant) ----
    if TREES_OUT.exists():
        committed = json.load(open(TREES_OUT))
        print(f"reproduces committed webmodels/arbiter_trees.json : "
              f"{'OK (identical to 1e-6)' if trees_match(committed, flat) else 'DIFF'}")

    # ---- verify 2: tree-walk == XGBoost predict_proba to <1e-6 ----
    df = pd.read_csv(DATASET)
    if "i_p_background" not in df.columns:
        df["i_p_background"] = 0.0
    df["conf_gap"]    = df["g_conf1"] - df["i_conf1"]
    df["conf_ratio"]  = df["g_conf1"] / (df["i_conf1"] + 1e-6)
    df["entropy_gap"] = df["i_entropy"] - df["g_entropy"]
    df["margin_gap"]  = df["g_margin"] - df["i_margin"]
    df["both_unsure"] = ((df["g_conf1"] < 0.5) & (df["i_conf1"] < 0.5)).astype(int)
    X = df[FEATS].to_numpy(dtype=np.float64)
    sample = X                       # score the entire feature table, not a subsample
    p_xgb = clf.predict_proba(sample)[:, 1]
    p_js  = np.array([walk(flat, row) for row in sample])
    maxerr = float(np.abs(p_xgb - p_js).max())
    print(f"max |predict_proba - JS tree-walk| over {len(sample)} rows = {maxerr:.2e}"
          f"   {'PASS (<1e-6)' if maxerr < 1e-6 else 'FAIL'}")

    # ---- write ----
    DEST.write_text(json.dumps(flat))
    print(f"wrote -> {DEST}" + ("" if WRITE_REAL else "   (--write to overwrite the real file)"))


if __name__ == "__main__":
    main()
