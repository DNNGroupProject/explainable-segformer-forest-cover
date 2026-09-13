"""Pair listing + split for full-scale DeepLabV3+ training.

Mirrors `dataset.py` / `attention_consistency/data.py`: sorted mask
filenames, stdlib `random.shuffle` under seed 42, front-slice
val / test / train -> 3576 / 766 / 766 on the 5,108-pair dataset.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = REPO_ROOT / "data" / "images"
MASK_DIR = REPO_ROOT / "data" / "masks"

# Canonical full-scale sizes (5108 * 0.15 = 766)
N_TRAIN, N_VAL, N_TEST = 3576, 766, 766


def _image_path_for(mask_filename: str, img_dir: Path = IMG_DIR) -> Optional[Path]:
    stem, ext = os.path.splitext(mask_filename)
    candidate_stem = stem.replace("_mask", "_sat")
    for e in (ext, ".jpg", ".jpeg", ".png"):
        p = img_dir / f"{candidate_stem}{e}"
        if p.exists():
            return p
    return None


def list_mask_files(masks_dir: Path = MASK_DIR) -> List[str]:
    return sorted(
        f
        for f in os.listdir(masks_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def make_splits_files(
    mask_files: Optional[List[str]] = None,
    *,
    n_train: int = N_TRAIN,
    n_val: int = N_VAL,
    n_test: int = N_TEST,
    seed: int = 42,
) -> dict:
    """Return train/val/test *mask filename* lists matching dataset.py's make_splits.

    Uses val_split=0.15, test_split=0.15 on the full 5108 list
    (766/766/3576). We take the same front slices after the same shuffle.
    """
    files = list(mask_files) if mask_files is not None else list_mask_files()
    files = sorted(files)
    random.seed(seed)
    random.shuffle(files)

    total = n_train + n_val + n_test
    if len(files) < total:
        raise ValueError(
            f"Need at least {total} masks, found {len(files)} under {MASK_DIR}"
        )
    files = files[:total]
    val_files = files[:n_val]
    test_files = files[n_val : n_val + n_test]
    train_files = files[n_val + n_test :]
    assert len(train_files) == n_train, (len(train_files), n_train)
    assert len(val_files) == n_val
    assert len(test_files) == n_test
    return {"train": train_files, "val": val_files, "test": test_files}


def load_split_arrays(
    split: str = "test",
    *,
    seed: int = 42,
    img_size: int = 256,
    img_dir: Path = IMG_DIR,
    mask_dir: Path = MASK_DIR,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load one split as (N,H,W,3) float01 images and (N,H,W) float masks."""
    splits = make_splits_files(seed=seed)
    names = splits[split]
    images, masks, kept = [], [], []
    for mask_name in names:
        img_path = _image_path_for(mask_name, img_dir)
        mask_path = mask_dir / mask_name
        if img_path is None or not mask_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size)).astype(np.float32) / 255.0

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)

        images.append(img)
        masks.append(mask)
        kept.append(mask_name)

    if not images:
        raise FileNotFoundError(f"No pairs loaded for split={split}")
    return np.stack(images), np.stack(masks), kept
