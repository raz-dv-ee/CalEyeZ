# Caleyez — XGBoost Arbiter (Re-Ranker)

The Arbiter re-ranks the combined Top-5 candidates of the Global (142-class) and
Israeli (13-class) YOLO11l-cls models to produce a single final prediction.

## Folder Map

```
arbiter/
├── README.md            this file
├── SYSTEM_STATE.md      living architecture & state tracker (v1→v4 history)
├── data/                dataset preparation
│   ├── generate_reranker_dataset.py
│   └── reranker_vector_mechanics.html   ← explains vector alignment + the 9 features
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
# 1. Build dataset CSV with the 9 features (val + test)
python arbiter\data\generate_reranker_dataset.py     # → arbiter\data\reranker_dataset.csv

# 2. Train the v4 Arbiter
python arbiter\train\train_reranker_xgb.py           # → arbiter\train\reranker_xgb.ubj

# 3. Evaluate the full pipeline on the test split
python arbiter\train\evaluate_system.py              # → arbiter\train\system_evaluation_results.csv
```

## Current Version: v4 — Evidence-Based Routing

9 features (4 raw + 5 engineered): `global_prob`, `israeli_prob`, `is_global_top1`,
`is_israeli_top1`, `global_entropy`, `local_entropy`, `global_top1_vs_top2`,
`local_top1_vs_top2`, `arbiter_dominance_score`.

The feature list and its column order **must stay identical** across all three
scripts. See `SYSTEM_STATE.md` for the full v1→v4 history.
