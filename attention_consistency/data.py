"""Dataset loading for the SegFormer + Attention Consistency Loss pipeline.

Defaults to `data/{images,masks}` at the repo root (not included in the
repo — see README). The shuffle/split mirrors `dataset.py`'s `make_splits`
bit-for-bit (sorted mask filenames, stdlib `random.shuffle` under the shared
seed 42, front-slice val/test/train) so every baseline in this repo is
trained/scored on the exact same held-out split (3576/766/766 off the
5,108-pair dataset).
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = REPO_ROOT / "data" / "images"
MASK_DIR = REPO_ROOT / "data" / "masks"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _image_path_for(mask_filename: str) -> Optional[Path]:
    """Mask -> image filename, matching dataset.py's
    `mask_name_to_image_name` exactly ('_mask' -> '_sat' substring replace,
    not '_mask_' -> '_sat_')."""
    stem, ext = os.path.splitext(mask_filename)
    candidate_stem = stem.replace("_mask", "_sat")
    for e in (ext, ".jpg", ".jpeg", ".png"):
        p = IMG_DIR / f"{candidate_stem}{e}"
        if p.exists():
            return p
    return None


def list_pairs(max_samples: Optional[int] = None, seed: int = 42) -> List[Tuple[Path, Path]]:
    """Sort *mask* filenames (dataset.py's source-of-truth list, not
    the image list), shuffle with stdlib `random` under `seed`, then take
    the first `max_samples`. Bit-identical pool selection to dataset.py's
    make_splits given the same seed and total count, so splits computed
    from this are comparable to the U-Net baseline's."""
    files = sorted(f for f in os.listdir(MASK_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    random.seed(seed)
    random.shuffle(files)
    pairs = []
    for f in files:
        ip = _image_path_for(f)
        if ip is not None:
            pairs.append((ip, MASK_DIR / f))
        if max_samples is not None and len(pairs) >= max_samples:
            break
    if not pairs:
        raise FileNotFoundError(f"No image/mask pairs found under {IMG_DIR} / {MASK_DIR}")
    return pairs


def load_pairs(
    pairs: List[Tuple[Path, Path]], img_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    images, masks = [], []
    for img_path, mask_path in pairs:
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
    return np.stack(images), np.stack(masks)


def make_splits(
    n_train: int, n_val: int, n_test: int, seed: int = 42
) -> dict:
    """Disjoint train/val/test split, same seed convention (42) the rest of
    the team uses -- and, as of this fix, the same *order* dataset.py uses (val = front slice, test = next slice, train =
    remainder of the shuffled pool), not a separate sklearn
    `train_test_split` call. At n_train=3576/n_val=766/n_test=766 (the real
    audited 5,108-pair dataset's 70/15/15 split), this selects the exact
    same held-out images as the U-Net baseline and augmentation ablation,
    not just a same-sized split."""
    total = n_train + n_val + n_test
    pairs = list_pairs(max_samples=total, seed=seed)
    val_pairs = pairs[:n_val]
    test_pairs = pairs[n_val : n_val + n_test]
    train_pairs = pairs[n_val + n_test :]
    return {"train": train_pairs, "val": val_pairs, "test": test_pairs}


def to_model_input(images_hw3_01: np.ndarray) -> torch.Tensor:
    """(N,H,W,3) in [0,1] -> (N,3,H,W) ImageNet-normalized tensor, matching
    the pretrained MiT-B0 encoder's expected input distribution."""
    x = (images_hw3_01 - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x.astype(np.float32)).permute(0, 3, 1, 2).contiguous()
    return x
