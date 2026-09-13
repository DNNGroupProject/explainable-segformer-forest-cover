"""Shared train-time augmentation for image/mask pairs.

By default every pipeline in this repo trains without augmentation -- the
SegFormer, U-Net, and attention-consistency training scripts all feed raw
pairs straight to the model.

This is a torch-friendly augmentation module, kept in `shared/` so every
baseline imports the *same* transforms -- otherwise "with augmentation" would mean
something slightly different in each person's results.

Usage (inside a Dataset.__getitem__, after /255.0 and before the ImageNet
normalize -- brightness/contrast only make sense while pixels are in [0,1]):

    from shared.augmentation import AugmentConfig, augment_pair

    rng = np.random.default_rng(42)          # once, in __init__
    image, mask = augment_pair(image, mask, rng)

numpy only -- no torch, no cv2, no albumentations. That keeps it importable
from the tf code as well as the torch code, and installable nowhere (same
reasoning as `scripts/mask_audit.py`, which sticks to
numpy + Pillow because cv2 isn't in everyone's environment).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AugmentConfig:
    """Knobs for one augmentation pass.

    Values mirror the project's only previous augmentation attempt, so
    "with augmentation" keeps meaning roughly what it meant
    in the pre-Phase 1 runs.
    """

    flip_horizontal_p: float = 0.5
    flip_vertical_p: float = 0.5
    rot90: bool = True             # picks k in {0,1,2,3} uniformly
    brightness_delta: float = 0.1  # image += U(-delta, +delta)
    contrast_range: tuple[float, float] = (0.9, 1.1)


DEFAULT_CONFIG = AugmentConfig()

# Everything off -- augment_pair becomes an identity function. Handy for
# tests and for anyone who wants the call site to stay in place while turning
# augmentation off from config.
NO_AUGMENT = AugmentConfig(
    flip_horizontal_p=0.0,
    flip_vertical_p=0.0,
    rot90=False,
    brightness_delta=0.0,
    contrast_range=(1.0, 1.0),
)


# --- geometric: applied to the image AND the mask -------------------------
#
# Only flips and 90-degree rotations. Arbitrary-angle rotation, scaling and
# elastic warps all need interpolation, and interpolating a label map is how
# you get the gray boundary pixels that the dataset audit
# (`scripts/mask_audit.py`) measured as completely absent from this dataset (0 pixels in
# 32-223). Flips and rot90 just move pixels around, so the mask comes out
# exactly as binary as it went in and the audit's conclusion still holds.

def _apply_geometry(
    image: np.ndarray, mask: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < cfg.flip_horizontal_p:
        image = np.fliplr(image)
        mask = np.fliplr(mask)

    if rng.random() < cfg.flip_vertical_p:
        image = np.flipud(image)
        mask = np.flipud(mask)

    if cfg.rot90:
        k = int(rng.integers(0, 4))
        # axes=(0,1) is (H,W) for both -- the image's trailing channel axis is
        # left alone, so the same rotation lands on both arrays.
        image = np.rot90(image, k, axes=(0, 1))
        mask = np.rot90(mask, k, axes=(0, 1))

    # np.fliplr/flipud/rot90 return views, so copy once at the end to hand
    # back a normal contiguous array (torch.from_numpy rejects negative strides).
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


# --- photometric: image only ----------------------------------------------
#
# The mask is a label map; brightening it would silently move pixels across
# the 0/1 boundary. Satellite tiles do vary in exposure and haze, which is
# what these two are standing in for.

def _apply_photometry(
    image: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator
) -> np.ndarray:
    if cfg.brightness_delta > 0:
        image = image + rng.uniform(-cfg.brightness_delta, cfg.brightness_delta)

    lo, hi = cfg.contrast_range
    if lo != 1.0 or hi != 1.0:
        # Scale around the image's own mean so contrast changes spread, not
        # overall brightness -- that's what the brightness term is for.
        pivot = float(image.mean())
        image = (image - pivot) * rng.uniform(lo, hi) + pivot

    return np.clip(image, 0.0, 1.0)


def augment_pair(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    cfg: AugmentConfig = DEFAULT_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment one (image, mask) pair.

    image: (H, W, 3) float in [0,1]   -- NOT yet ImageNet-normalized
    mask:  (H, W) with values {0, 1}
    rng:   a seeded np.random.default_rng; pass the same one every call so a
           run is reproducible from a single seed

    Returns new arrays; the inputs are left untouched.
    """
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"image and mask must cover the same grid, got {image.shape[:2]} vs {mask.shape[:2]}"
        )

    image, mask = _apply_geometry(image, mask, cfg, rng)
    image = _apply_photometry(image, cfg, rng)
    # rng.uniform hands back float64, which would silently double the memory
    # of every batch, so pin the dtype back down on the way out.
    return image.astype(np.float32), mask
