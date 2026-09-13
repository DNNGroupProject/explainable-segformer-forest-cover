"""Dataset helpers shared by adapters (mirrors the evaluation-harness split convention)."""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import config


def find_mask_path(mask_folder: str, filename: str) -> Optional[str]:
    stem, ext = os.path.splitext(filename)
    common_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    candidate_stems = [stem]
    if "_sat_" in stem:
        candidate_stems.append(stem.replace("_sat_", "_mask_"))
    ext_order = [ext] + [e for e in common_exts if e != ext]
    for candidate_stem in candidate_stems:
        for candidate_ext in ext_order:
            candidate = os.path.join(mask_folder, candidate_stem + candidate_ext)
            if os.path.exists(candidate):
                return candidate
    return None


def load_pairs(
    img_dir=None,
    mask_dir=None,
    img_size: int = None,
    mask_threshold: int = None,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    img_dir = img_dir or str(config.IMG_DIR)
    mask_dir = mask_dir or str(config.MASK_DIR)
    img_size = img_size or config.IMG_SIZE
    mask_threshold = mask_threshold if mask_threshold is not None else config.MASK_THRESHOLD

    image_files = sorted(
        f
        for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    )
    if max_samples is not None:
        image_files = image_files[:max_samples]

    images, masks = [], []
    for file in image_files:
        mask_path = find_mask_path(mask_dir, file)
        if mask_path is None:
            continue
        img = cv2.imread(os.path.join(img_dir, file))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype(np.float32) / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > mask_threshold).astype(np.float32)

        images.append(img)
        masks.append(mask)

    if not images:
        raise FileNotFoundError(
            f"No image/mask pairs found under\n  {img_dir}\n  {mask_dir}"
        )
    return np.stack(images), np.stack(masks)


def split_dataset(
    images: np.ndarray,
    masks: np.ndarray,
    seed: int = None,
    train_ratio: float = None,
    val_ratio: float = None,
) -> dict:
    seed = config.SEED if seed is None else seed
    train_ratio = config.TRAIN_RATIO if train_ratio is None else train_ratio
    val_ratio = config.VAL_RATIO if val_ratio is None else val_ratio

    X_temp, X_test, y_temp, y_test = train_test_split(
        images, masks, test_size=0.10, random_state=seed
    )
    val_fraction_of_temp = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction_of_temp, random_state=seed
    )
    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
