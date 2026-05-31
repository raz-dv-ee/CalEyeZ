# Caleyez — XGBoost Arbiter (Re-Ranker)

The Arbiter re-ranks the combined Top-5 candidates of the Global (142-class) and
Israeli (13-class) YOLO11l-cls models to produce a single final prediction.

## Folder Map

Docs live at the **project root**: this file (`ARBITER_README.md`),
`SYSTEM_STATE.md` (the living architecture & v1→v5 history tracker), and
`THE_CEILING.html` (why overall caps at ~88.4%).

```
arbiter/
├── data/                dataset preparation
│   ├── generate_reranker_dataset.py
│   └── reranker_vector_mechanics.html   ← explains vector alignment + the 15 features
├── train/               training + evaluation
│   ├── train_reranker_xgb.py
│   ├── evaluate_system.py
│   ├── evaluation_mechanics_and_hierarchy.html   ← explains training + hierarchy
│   └── reranker_xgb.ubj                  ← trained model (generated)
└── old_versions/        superseded prototypes (router design, early arbiter)
```

## Run Order

Run from the project root (`E:\final project - models retrain`). Scripts use
absolute paths, so the working directory does not matter.

```powershell
# 1. Build dataset CSV (val + test) — produces the 9 base features
#    (the 6 derived features are computed at train time)
python arbiter\data\generate_reranker_dataset.py     # → arbiter\data\reranker_dataset.csv

# 2. Train the v5 Arbiter (optional --alpha 0..1 to shift Global/Israeli balance)
python arbiter\train\train_reranker_xgb.py           # → arbiter\train\reranker_xgb.ubj

# 3. Evaluate the full pipeline on the test split
python arbiter\train\evaluate_system.py              # → arbiter\train\system_evaluation_results.csv
```

## Current Version: v5 — Tunable Domain Dial + Interaction Features

**Result:** overall **84.99%** Re-Rank Accuracy@1 (Global 86.18% / Israeli 62.84%).

15 features = 9 base (`global_prob`, `israeli_prob`, `is_global_top1`,
`is_israeli_top1`, `global_entropy`, `local_entropy`, `global_top1_vs_top2`,
`local_top1_vs_top2`, `arbiter_dominance_score`) + 6 derived (`conf_global`,
`conf_israeli`, `israeli_top1_p`, `global_top1_p`, `prob_ratio`, `both_agree`).
The feature list and column order **must stay identical** between
`train_reranker_xgb.py` and `evaluate_system.py`.

`DOMAIN_WEIGHT_ALPHA` dials the Global↔Israeli trade-off (0=max overall, 0.3=shipped,
1.0=v4 parity).

> ⚠️ **Hard ceiling:** overall caps at ~88.4% with these two models — 90%+ requires
> improving the Global YOLO, not the arbiter. See `THE_CEILING.html` and
> `SYSTEM_STATE.md` for the full v1→v5 history and the oracle analysis.
