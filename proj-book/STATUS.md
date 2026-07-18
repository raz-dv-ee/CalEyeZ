# CalEyeZ Project Book — Build Status Tracker

> This file tracks **exactly where we are** in writing the capstone book.
> Update it every time a section is drafted, reviewed, or revised.
> The book itself is written in **Typst markup** (paste into https://typst.app).

---

## Files in `proj-book/`
| File | Purpose |
| :--- | :--- |
| `capstone_project_template.md` | Official Shenkar book template (chapter list + weights). Source of structure. |
| `CalEyeZ_FRS.md` | Functional Requirement Spec — source data (problem, requirements, use cases). |
| `CalEyeZ_SDD.md` | System Design Doc — source data (architecture, components, BOM, schedule). |
| `caleyez_book.typ` | **THE BOOK** — Typst markup, copy-paste into typst.app. Built section by section. |
| `STATUS.md` | This tracker. |

---

## Ground-truth numbers (USE THESE, not the older draft figures)
*Reconciled from the live build log / memory on 2026-06-21.*

- **Global model:** YOLO11l-cls, **132 classes**, imgsz 320, best.pt = epoch 73.
  Test top-1 **88.18%**, top-5 96.94%. val≈test (leakage removed).
- **Israeli model (V2):** YOLO11l-cls, **14 classes** (13 food + 1 open-set "background"),
  imgsz 224, best ep112. ~92% on its own domain; overconfident out-of-domain (the routing problem).
- **XGBoost Router (V2):** **20 features** (top-5 confs, entropy, margins, interactions,
  + `i_p_background` = #1 by gain). Routing acc **96.1%**, **ROC-AUC 0.973**, Israeli recall **79.2%**.
- **Full system top-1 (held-out test): 86.2%** (V2). Always-global baseline 78.0%.
  **Oracle / ceiling ≈ 88.4%** → bottleneck is the MODELS, not the router (~70% of errors = both-models-wrong).
- **Weight:** pivoted **OCR → BLE** (Swan scale, GATT serial, HEX packets). OCR kept as documented R&D.
- **Nutrition:** USDA FoodData Central API + heuristic filter + local JSON fallback.
- **Edge build:** both models exported to **ONNX** (torch-free CPU), parity 0% top-1 mismatch,
  ~352 ms/Analyze on CPU. PyInstaller standalone exe.
- **Real-image field test:** 30/40 = **75%**, Wilson 95% CI [59.8, 85.8] (later updated set 85.1% on 47 photos).
- **GUI:** Python + Tkinter/ttk, single-window dashboard.
- **Authors:** Raz Dvora & Roi Tzur. Supervisors: Dr. Gabriela Dorfman Furman, Dr. Zeev Weissman. Shenkar.

---

## Writing conventions (apply to every new chapter)
- **No em-dashes** (`—`). Use a spaced hyphen ` - `, comma, colon, or split the sentence. (AI-tell.)
- **No `§` sign.** Write "Section 6", not "§6".
- **Live-site links:** use the `#web("anchor")[text]` helper. Base = `https://raz-dv-ee.github.io/CalEyeZ/#`
  (CONFIRM this Pages URL with Raz). Anchors = tab names: `trainlab`, `arbiter`, `dataset`, `ble`,
  `demo`, `cnn`, `voice`, `rejected`, `riv`, `global`, `israeli`, `israeliv2`, `analysisv2`.
  Already linked: XGBoost Router→arbiter, learning rate & softmax→trainlab, Bluetooth protocol→ble,
  dataset→dataset, Tried&Rejected→rejected.
- Note: `lr0` = initial learning rate; final LR = `lr0 × lrf` (now stated inline in §3.3).

## Chapter-by-chapter progress

| # | Chapter (weight) | Status | Notes |
| :--- | :--- | :--- | :--- |
| — | Preamble / title / styling | ✅ DRAFTED | Typst setup, title page, acknowledgements |
| 1 | Introduction (10%) | ✅ DRAFTED | 1.1–1.4 written with real numbers |
| 2 | Functional Requirements (10%) | ✅ DRAFTED | Unit 1/2 + UI reqs + use-case scenario |
| 3 | Design (15%) | ✅ DRAFTED | Architecture, HW, SW. §3.3 now opens with the **voice-fingerprint analogy** intuition + 2 Typst-drawn diagrams (pipeline + router split), then the math (softmax, CE+label-smooth, cosine LR, entropy, margin, XGBoost objective, scale_pos_weight, SHAP, fusion formula) + real code |
| 4 | Testing & Validation Plan (10%) | ✅ DRAFTED | Objectives, unit/integration/system + edge-parity + field test; top-k & Wilson CI formulas; plan only (no results) |
| 5 | Implementation (10%) | ✅ DRAFTED | BLE reverse-eng (real decode + carry-byte bug), dataset/train/router/ONNX stages, fusion integration, PyInstaller deploy, BOM table — real code snippets — **awaiting your review** |
| 6 | Results & Evaluation (25%) | ✅ DRAFTED | Model results, router V1-vs-V2 table, system vs baseline/oracle, error decomp, ONNX parity, field test (Wilson CI + per-class table), **Tried & Rejected** (lossy-chain diagram + MiDaS/shadow/OCR + comparison table), spec validation, ≥6 limitations, known bugs, conclusion — **awaiting your review** |
| 7 | Conclusion (5%) | ✅ DRAFTED | Interpretation, challenges/lessons, achievements, future work |
| 8 | References (5%) | ✅ DRAFTED | YOLO/XGBoost/SHAP/USDA/Wilson/MiDaS/ONNX/Bleak + project site |
| 9 | Appendices (10%) | ✅ DRAFTED | Acronyms table, script-roles table, quick-start manual, pointer to Ch6 test tables |

Legend: ⬜ TODO · 🚧 IN PROGRESS · ✅ DRAFTED · ✔️ REVIEWED-BY-RAZ

---

## Training theory added to §3.3 (2026-06-22)
Five new subsections, all with math + tied to our real numbers:
- 3.3.3 Gradient descent & backpropagation (θ←θ-η∇L, the layer-wise δ recursion, AdamW update).
- 3.3.4 Batches/epochs (mini-batch gradient, Var(ĝ)∝σ²/B, N/B steps = 4,343/epoch at N=69,491 B=16).
- 3.3.5 Train vs val loss + generalisation gap; REAL run: val-loss min @ep50 (0.558 vs train 0.346,
  gap 0.21) widening to 0.58 @ep116; best.pt @ep73 (88.16%); early stopping patience 30 (150→116).
- 3.3.6 Hardware budget: VRAM M_act∝B·H·W·C, 224²→320² = ×2.04 offset by batch 32→16 (~2.2GB/8GB),
  why YOLO11l not 11x; CPU t≈FLOPs/throughput → 0.35s ONNX vs 1-3s torch-CPU, embeddings=0 on edge.
- Typst note: Hadamard product uses `circle.small` (NOT `dot.circle`, which errors). Page count now 38.

## Figures added (2026-06-22)
- Real images live in `proj-book/figures/` (self-contained relative paths, so it works on typst.app
  if you upload the `figures/` folder alongside the .typ). 10 figures, all `#figure(...)` auto-numbered:
  - §6.1: global_curves, global_confusion (132-class, full-res in repo), global_val_preds,
    israeli_curves, israeli_confusion (V2, 14-class).
  - §6.5 field test: field_card_1 / field_card_2 (side-by-side evidence cards).
  - §6.7 Tried & Rejected: fail_apple_midas (MiDaS 139.3 cm³), fail_can_shadow (519.67 cm³),
    fail_archimedes (gold-standard validation).
- DROPPED `sys_5_separability.png` (ROC-AUC 0.93) and `sys_1_outcome.png` (83.8%): those are **V1**
  figures and would contradict the **V2** headline numbers (0.973 / 86.2%). If a V2 separability/ROC
  figure is wanted, regenerate from the current arbiter via scripts/arbiter/system_analysis.py.
- Page count now **36** (was 29) - still well under the 80 limit. Compiles clean.

## Layout fix (2026-06-22)
- Paragraphs no longer split across pages: `#show par: set block(breakable: false)`.
- Code snippets kept on one page: `breakable: false` in the raw-block show rule.
- Headings stick to following content: `#show heading: set block(sticky: true)`.
- Still 29 pages after the change; tradeoff is some bottom-of-page whitespace (expected/correct).

## Compile check (2026-06-22) — PASSED
- Compiled with `typst` (python pkg 0.15.0): **clean, no errors. 29 pages** (limit 80).
  `caleyez_book.pdf` in this folder is the build artifact (regenerate: `python -c "import typst;
  open('proj-book/caleyez_book.pdf','wb').write(typst.compile('proj-book/caleyez_book.typ'))"`).
- Visually verified bottom-up: title/TOC, both §3.3 diagrams (pipeline + router split), lossy-chain
  diagram, all Ch6 tables, BOM, acronym/script tables, and complex math (fusion fraction, Wilson CI,
  XGBoost objective, softmax) all render. `#web(...)` links show blue+underlined and resolve.
- FIXED 3 accidental bullets caused by the em-dash→hyphen swap (an em-dash that wrapped to the start
  of a source line became a `- ` Typst list item): §1.1 innovation, §3.4 result cards, §6.2 router.
  If editing later, watch for any new line that *starts* with `- ` mid-sentence.

## Next step — FULL FIRST DRAFT COMPLETE (all 9 chapters)
The whole book is drafted end-to-end. Remaining work is review/polish, not new chapters:
- Paste the full `caleyez_book.typ` into typst.app and read it through; fix any compile errors.
- Decide on real figures/screenshots to drop in (confusion matrices, GUI shots, field-test cards)
  via `image("...")` — currently the book uses Typst-drawn diagrams + tables only.
- Verify all `#web(...)` links resolve on the live site.
- Optional: tighten any spots where the em-dash→hyphen swap reads awkwardly.
