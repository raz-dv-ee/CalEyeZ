#!/usr/bin/env python3
"""
Label-Space Flattening + Dataset Restructure for the Caleyez Global YOLOv11 model.

What it does
------------
1. Scans datasets/general_model_data_set/{train,val,test}.
2. MERGES sub-classes into parent classes (logical remap, applied while pooling):
       filet_mignon, prime_rib   -> steak
       frozen_yogurt             -> ice_cream
       dumplings                 -> gyoza
       beef_tartare, tuna_tartare-> tartare
3. DELETES (drops) noisy classes:
       pork_chop, pulled_pork_sandwich
4. Pools every surviving image per FINAL label, then resplits cleanly
   70% train / 20% val / 10% test into datasets/general_model_flattened.
5. (Optional, gated) Removes deprecated Arbiter files.

Design notes
------------
- NON-DESTRUCTIVE by default: reads the original dataset read-only and COPIES into
  the new tree, so the original 6.5 GB dataset survives as a backup. Use --move to
  reclaim disk (destructive to source).
- Deterministic: fixed --seed so the split is reproducible.
- Idempotent target: the destination is wiped (with confirmation) before building.
- Filename collisions across source splits are resolved by prefixing.

Usage
-----
    python scripts/flatten_general_dataset.py                 # build flattened dataset (copy)
    python scripts/flatten_general_dataset.py --move          # move instead of copy
    python scripts/flatten_general_dataset.py --dry-run       # plan only, touch nothing
    python scripts/flatten_general_dataset.py --clean-arbiter --yes   # also wipe arbiter files
"""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import sys
import csv
from collections import defaultdict
from pathlib import Path

try:
    import warnings
    import numpy as np
    from PIL import Image
    warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
    _PERCEPTUAL_OK = True
except Exception:  # pragma: no cover - perceptual dedup just disabled
    _PERCEPTUAL_OK = False

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = PROJECT_ROOT / "datasets" / "general_model_data_set"
DEFAULT_DST = PROJECT_ROOT / "datasets" / "general_model_flattened"

SPLITS = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}

# source_class -> final_class
CLASS_MERGE_MAP = {
    "filet_mignon": "steak",
    "prime_rib": "steak",
    "frozen_yogurt": "ice_cream",
    "dumplings": "gyoza",
    "beef_tartare": "tartare",
    "tuna_tartare": "tartare",
}

# Label hygiene: normalize broken folder names (lowercase, underscores, spelling).
# Lossless 1:1 renames — distinct from merges. None of these collide with a merge
# target or an existing class, so they create cleanly-named standalone classes.
RENAME_MAP = {
    "Baked_Potato": "baked_potato",   # only capitalized class
    "chilli pepper": "chili_pepper",  # only class containing a space + misspelling
    "jalepeno": "jalapeno",           # misspelling
    "raddish": "radish",              # misspelling
}

# classes dropped entirely
#   pork_chop, pulled_pork_sandwich, baby_back_ribs : out-of-scope (pork)
#   paprika : mixed/noisy folder (spice powder + bell peppers + chillies) -> unusable
#   soy_beans : redundant with edamame (same food); the small/noisy folder is dropped
DELETE_CLASSES = {"pork_chop", "pulled_pork_sandwich", "baby_back_ribs", "paprika",
                  "soy_beans"}

# classes worth a human eyeball (printed, NOT auto-deleted)
AMBIGUOUS_WARN: set[str] = set()

# Deprecated arbiter files (relative to PROJECT_ROOT). Only the real project root,
# never the gitignored nested CalEyeZ\ clone.
DEPRECATED_ARBITER = [
    "arbiter/train/train_reranker_xgb.py",
    "arbiter/data/generate_reranker_dataset.py",
    "arbiter/data/reranker_dataset.csv",
    "arbiter/data/reranker_vector_mechanics.html",
    "arbiter/train/evaluation_mechanics_and_hierarchy.html",
    "arbiter/train/reranker_xgb.ubj",
    "datasets/predict_system.py",
    "SYSTEM_STATE.md",
    "REPO_STATE.md",
    "ARBITER_README.md",
    "THE_CEILING.html",
]

BAR = "=" * 72


def log(msg: str = "") -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Step 1-3: scan + pool with merge/delete applied
# --------------------------------------------------------------------------- #

def list_images(folder: Path) -> list[Path]:
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def build_pool(src: Path) -> dict[str, list[tuple[Path, str, str]]]:
    """Return {final_label: [(image_path, src_split, src_class), ...]}."""
    log(BAR)
    log("STEP 1-3  Scanning source, applying merges + deletions")
    log(BAR)

    missing = [s for s in SPLITS if not (src / s).is_dir()]
    if missing:
        sys.exit(f"ERROR: missing split folder(s) under {src}: {missing}")

    pool: dict[str, list[tuple[Path, str, str]]] = defaultdict(list)
    deleted_imgs = 0
    seen_classes: set[str] = set()

    for split in SPLITS:
        split_dir = src / split
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            cls = class_dir.name
            seen_classes.add(cls)
            imgs = list_images(class_dir)

            if cls in DELETE_CLASSES:
                deleted_imgs += len(imgs)
                log(f"  [DELETE] {split}/{cls:<28} dropping {len(imgs):>5} images")
                continue

            if cls in CLASS_MERGE_MAP:
                final = CLASS_MERGE_MAP[cls]
                log(f"  [MERGE ] {split}/{cls:<28} -> {final:<18} ({len(imgs):>5} images)")
            elif cls in RENAME_MAP:
                final = RENAME_MAP[cls]
                log(f"  [RENAME] {split}/{cls:<28} -> {final:<18} ({len(imgs):>5} images)")
            else:
                final = cls
            for img in imgs:
                pool[final].append((img, split, cls))

    # warn about ambiguous classes we deliberately did NOT touch
    for cls in sorted(AMBIGUOUS_WARN & seen_classes):
        log(f"  [WARN  ] '{cls}' is pork-adjacent but was NOT in the delete list "
            f"-> KEPT. Remove manually if undesired.")

    log()
    log(f"  Final label count : {len(pool)}")
    log(f"  Images pooled     : {sum(len(v) for v in pool.values()):,}")
    log(f"  Images deleted    : {deleted_imgs:,}")
    return pool


# --------------------------------------------------------------------------- #
# Step 3.5: content-based dedup — guarantee splits never share an image
# --------------------------------------------------------------------------- #
#
# Two layers:
#   (a) EXACT: SHA-1 of file bytes. Byte-identical copies are collapsed to ONE
#       representative; the redundant copies are dropped entirely.
#   (b) NEAR (perceptual): 64-bit dHash. Images whose hashes differ by <= HAMMING
#       are grouped into a cluster. We DON'T drop near-dups (they add real visual
#       variance) but we keep each cluster together so it lands in a single split.
#
# The split unit is therefore the cluster, never the raw image -> a photo and its
# resized/recompressed/near-identical twins can never appear in two splits.

DHASH_HAMMING = 5  # threshold for "near duplicate"


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dhash64(path: Path) -> int | None:
    """9x8 grayscale difference hash -> 64-bit int. None if unreadable."""
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((9, 8), Image.BILINEAR)
            a = np.asarray(im, dtype=np.int16)
        bits = (a[:, 1:] > a[:, :-1]).flatten()
        out = 0
        for b in bits:
            out = (out << 1) | int(b)
        return out
    except Exception:
        return None


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def dedup_label(items: list[tuple], perceptual: bool) -> tuple[list[list[tuple]], dict]:
    """Collapse exact dups, cluster near-dups. Return (clusters, stats)."""
    # (a) exact dedup
    by_sha: dict[str, tuple] = {}
    dropped_exact = 0
    for it in items:
        digest = _sha1(it[0])
        if digest in by_sha:
            dropped_exact += 1
        else:
            by_sha[digest] = it
    reps = list(by_sha.values())

    stats = {"in": len(items), "dropped_exact": dropped_exact,
             "unique": len(reps), "near_clusters_merged": 0, "unreadable": 0}

    if not perceptual or not _PERCEPTUAL_OK or len(reps) < 2:
        return [[r] for r in reps], stats

    # (b) near-dup clustering via dHash + vectorized pairwise hamming
    hashes, valid = [], []
    for r in reps:
        h = _dhash64(r[0])
        if h is None:
            stats["unreadable"] += 1
        else:
            hashes.append(h)
            valid.append(r)
    if len(valid) < 2:
        return [[r] for r in reps], stats

    arr = np.array(hashes, dtype=np.uint64)
    uf = _UF(len(valid))
    # pairwise XOR then popcount; chunked to bound memory on big classes
    for i in range(len(valid)):
        xor = np.bitwise_xor(arr[i], arr[i + 1:])
        if xor.size:
            # popcount on uint64
            ham = np.zeros(xor.shape, dtype=np.uint16)
            v = xor.copy()
            while v.any():
                ham += (v & np.uint64(1)).astype(np.uint16)
                v >>= np.uint64(1)
            for off in np.nonzero(ham <= DHASH_HAMMING)[0]:
                uf.union(i, i + 1 + int(off))

    groups: dict[int, list[tuple]] = defaultdict(list)
    for idx, r in enumerate(valid):
        groups[uf.find(idx)].append(r)
    stats["near_clusters_merged"] = len(valid) - len(groups)
    return list(groups.values()), stats


# --------------------------------------------------------------------------- #
# Step 4: resplit 70/20/10 and write
# --------------------------------------------------------------------------- #

def split_clusters(clusters: list[list[tuple]], rng: random.Random) -> dict[str, list[tuple]]:
    """Assign whole clusters to splits, hitting 70/20/10 by image count."""
    rng.shuffle(clusters)
    total = sum(len(c) for c in clusters)
    want_train = int(total * SPLIT_RATIOS["train"])
    want_val = int(total * SPLIT_RATIOS["val"])
    out = {"train": [], "val": [], "test": []}
    n_train = n_val = 0
    for c in clusters:
        if n_train < want_train:
            out["train"].extend(c); n_train += len(c)
        elif n_val < want_val:
            out["val"].extend(c); n_val += len(c)
        else:
            out["test"].extend(c)
    return out


def safe_name(src_split: str, src_class: str, img: Path, used: set[str]) -> str:
    """Build a collision-free destination filename."""
    name = img.name
    if name not in used:
        used.add(name)
        return name
    stem, ext = img.stem, img.suffix
    candidate = f"{src_split}_{src_class}_{stem}{ext}"
    i = 1
    while candidate in used:
        candidate = f"{src_split}_{src_class}_{stem}_{i}{ext}"
        i += 1
    used.add(candidate)
    return candidate


def build_dataset(pool, dst: Path, seed: int, move: bool, dry_run: bool,
                  perceptual: bool) -> None:
    log()
    log(BAR)
    log(f"STEP 3.5/4  Dedup + resplit 70/20/10 -> {dst}")
    log(f"        mode={'MOVE' if move else 'COPY'}  seed={seed}  dry_run={dry_run}")
    log(f"        exact-dedup=ON  perceptual-dedup="
        f"{'ON (h<=%d)' % DHASH_HAMMING if (perceptual and _PERCEPTUAL_OK) else 'OFF'}")
    log(BAR)
    if perceptual and not _PERCEPTUAL_OK:
        log("  [WARN] Pillow/numpy unavailable -> perceptual dedup skipped.")

    if dst.exists() and not dry_run:
        log(f"  Destination exists, clearing: {dst}")
        shutil.rmtree(dst)

    rng = random.Random(seed)
    op = shutil.move if move else shutil.copy2
    totals = {s: 0 for s in SPLITS}
    manifest_rows: list[tuple] = []
    tot_exact = tot_near = 0

    for label in sorted(pool):
        clusters, st = dedup_label(pool[label], perceptual)
        tot_exact += st["dropped_exact"]
        tot_near += st["near_clusters_merged"]
        assigned = split_clusters(clusters, rng)
        per = {s: len(assigned[s]) for s in SPLITS}
        dd = ""
        if st["dropped_exact"] or st["near_clusters_merged"]:
            dd = f"  [dedup -{st['dropped_exact']} exact, {st['near_clusters_merged']} near-merged]"
        log(f"  {label:<22} n={st['unique']:>6}  "
            f"train={per['train']:>5} val={per['val']:>5} test={per['test']:>5}{dd}")

        for split in SPLITS:
            used: set[str] = set()
            out_dir = dst / split / label
            if not dry_run and assigned[split]:
                out_dir.mkdir(parents=True, exist_ok=True)
            for img, src_split, src_class in assigned[split]:
                fname = safe_name(src_split, src_class, img, used)
                if not dry_run:
                    op(str(img), str(out_dir / fname))
                totals[split] += 1
                manifest_rows.append((split, label, src_split, src_class, img.name, fname))

    log()
    log(f"  Exact duplicates dropped : {tot_exact:,}")
    log(f"  Near-dup pairs co-located: {tot_near:,}")
    grand = sum(totals.values())
    log("  RESPLIT TOTALS")
    for s in SPLITS:
        pct = (totals[s] / grand * 100) if grand else 0
        log(f"    {s:<5} {totals[s]:>7,}  ({pct:4.1f}%)")
    log(f"    total {grand:>7,}")

    if not dry_run:
        manifest = dst / "split_manifest.csv"
        with open(manifest, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["dst_split", "label", "src_split", "src_class",
                        "orig_filename", "dst_filename"])
            w.writerows(manifest_rows)
        log(f"  Manifest written: {manifest}")


# --------------------------------------------------------------------------- #
# Step 5: deprecated arbiter cleanup (gated)
# --------------------------------------------------------------------------- #

def clean_arbiter(confirm: bool) -> None:
    log()
    log(BAR)
    log("STEP 5  Deprecated Arbiter cleanup")
    log(BAR)
    found = [(rel, PROJECT_ROOT / rel) for rel in DEPRECATED_ARBITER
             if (PROJECT_ROOT / rel).exists()]
    if not found:
        log("  Nothing to delete (already clean).")
        return
    for rel, _ in found:
        log(f"  {'DELETE' if confirm else 'WOULD DELETE'}: {rel}")
    if not confirm:
        log("\n  Dry run. Re-run with --clean-arbiter --yes to actually delete.")
        return
    for _, path in found:
        path.unlink()
    log(f"\n  Deleted {len(found)} deprecated file(s).")
    log("  (Note: also superseded -> arbiter/old_versions/ and the nested CalEyeZ\\ "
        "clone were left untouched.)")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Flatten + restructure the Global dataset.")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--move", action="store_true",
                    help="MOVE files (destructive to source) instead of copying.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan and report only; write nothing.")
    ap.add_argument("--clean-arbiter", action="store_true",
                    help="Also remove deprecated arbiter files.")
    ap.add_argument("--yes", action="store_true",
                    help="Confirm destructive arbiter deletion (otherwise dry-run).")
    ap.add_argument("--no-perceptual", action="store_true",
                    help="Disable near-duplicate (dHash) clustering; exact dedup still runs.")
    args = ap.parse_args()

    log(BAR)
    log("CALEYEZ  Label-Space Flattening pipeline")
    log(f"  src = {args.src}")
    log(f"  dst = {args.dst}")
    log(BAR)

    if args.move and not args.dry_run:
        log("  !! --move is DESTRUCTIVE to the source dataset. Ctrl-C now to abort. !!")

    pool = build_pool(args.src)
    build_dataset(pool, args.dst, args.seed, args.move, args.dry_run,
                  perceptual=not args.no_perceptual)

    if args.clean_arbiter:
        clean_arbiter(confirm=args.yes and not args.dry_run)

    log()
    log(BAR)
    log("DONE." + ("  (dry run — nothing was written)" if args.dry_run else ""))
    log(BAR)


if __name__ == "__main__":
    main()
