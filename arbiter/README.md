# Caleyez — XGBoost Arbiter

The Arbiter re-ranks the combined Top-5 candidates of the Global (142-class) and
Israeli (13-class) YOLO11l-cls models to produce a single final prediction.

## Folder Map

| Folder | Contents |
|--------|----------|
| `data_gen/` | `generate_reranker_dataset.py` — builds the Pointwise training CSV.<br>`reranker_vector_mechanics.html` — explains vector alignment + the 9-feature space. |
| `train/` | `train_reranker_xgb.py` — trains the XGBoost Re-Ranker (v4, evidence-based routing). |
| `evaluate/` | `evaluate_system.py` — full end-to-end pipeline evaluation on the test split.<br>`evaluation_mechanics_and_hierarchy.html` — version history + hierarchy analysis. |
| `docs/` | `SYSTEM_STATE.md` — the living architecture & state tracker. **Start here.** |
| `artifacts/` | Generated data + model: `reranker_dataset.csv`, `reranker_xgb.ubj`, `system_evaluation_results.csv`. |
| `_archive/` | Superseded earlier files, kept for reference. |

## Run Order

Run from the project root (`E:\final project - models retrain`). The scripts use
absolute paths, so the working directory does not matter.

```powershell
# 1. Build dataset CSV with the 9 features (val + test)
python arbiter\data_gen\generate_reranker_dataset.py

# 2. Train the v4 Arbiter
python arbiter\train\train_reranker_xgb.py

# 3. Evaluate the full pipeline on the test split
python arbiter\evaluate\evaluate_system.py
```

## Current Version: v4 — Evidence-Based Routing

9 features (4 raw + 5 engineered): `global_prob`, `israeli_prob`, `is_global_top1`,
`is_israeli_top1`, `global_entropy`, `local_entropy`, `global_top1_vs_top2`,
`local_top1_vs_top2`, `arbiter_dominance_score`.

The feature list and its column order **must stay identical** across all three
scripts. See `docs/SYSTEM_STATE.md` Section 3 for the full v1→v4 history.
