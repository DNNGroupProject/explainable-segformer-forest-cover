"""
AAMO — Average Attention-Mask Overlap

Proposal (§4.4): normalized overlap between the thresholded attention map
and the ground-truth forest mask. Quantifies explanation faithfulness.

Primary definition used here (hard AAMO / recall of forest under attention):

    AAMO = | threshold(A) ∩ Y | / | Y |

Also report Dice between thresholded attention and Y (AAMO_dice) as a
symmetric overlap check.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    while x.ndim > 2 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim != 2:
        raise ValueError(f"Expected 2D attention/mask, got shape {x.shape}")
    return x


def normalize_attention(attention: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    a = _as_2d(attention)
    amin, amax = float(a.min()), float(a.max())
    if amax - amin < eps:
        return np.zeros_like(a)
    return (a - amin) / (amax - amin)


def compute_aamo(
    attention: np.ndarray,
    mask: np.ndarray,
    thr: float = 0.5,
    normalize: bool = True,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Parameters
    ----------
    attention : float map, any scale (will be normalized if normalize=True)
    mask      : binary forest GT {0,1}
    thr       : threshold on (normalized) attention

    Returns
    -------
    dict with:
      aamo       — |A_thr ∩ Y| / |Y|   (proposal-style overlap / forest coverage)
      aamo_dice  — Dice(A_thr, Y)
      aamo_iou   — IoU(A_thr, Y)
      precision_att, recall_att
    """
    a = normalize_attention(attention) if normalize else _as_2d(attention)
    y = (_as_2d(mask) >= 0.5).astype(np.float64)
    # Inclusive threshold so thr=0.5 keeps pixels with attention exactly 0.5
    a_thr = (a >= thr).astype(np.float64)

    inter = float(np.sum(a_thr * y))
    y_sum = float(np.sum(y))
    a_sum = float(np.sum(a_thr))

    aamo = inter / (y_sum + eps)
    precision = inter / (a_sum + eps)
    recall = inter / (y_sum + eps)
    dice = (2 * inter) / (a_sum + y_sum + eps)
    iou = inter / (a_sum + y_sum - inter + eps)

    return {
        "aamo": float(aamo),
        "aamo_dice": float(dice),
        "aamo_iou": float(iou),
        "precision_att": float(precision),
        "recall_att": float(recall),
    }


def compute_aamo_soft(
    attention: np.ndarray,
    mask: np.ndarray,
    normalize: bool = True,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Soft variant: mean attention mass on forest pixels (no threshold)."""
    a = normalize_attention(attention) if normalize else _as_2d(attention)
    y = (_as_2d(mask) >= 0.5).astype(np.float64)
    forest_mass = float(np.sum(a * y))
    y_sum = float(np.sum(y))
    total_mass = float(np.sum(a))
    return {
        "aamo_soft": forest_mass / (y_sum + eps),
        "att_mass_on_forest_frac": forest_mass / (total_mass + eps),
    }


def mean_aamo(
    attentions: list,
    masks: list,
    thr: float = 0.5,
) -> Dict[str, float]:
    """Average AAMO over a list of (attention, mask) pairs."""
    rows = [compute_aamo(a, m, thr=thr) for a, m in zip(attentions, masks)]
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


# ── Numeric unit example (also runnable as script) ───────────────────────────

def _unit_example() -> None:
    """
    4x4 hand calculation:

    Y (forest)          A (attention, already 0..1)
    1 1 0 0             0.9 0.8 0.1 0.0
    1 1 0 0             0.7 0.6 0.2 0.0
    1 1 0 0             0.5 0.4 0.1 0.0
    1 1 0 0             0.9 0.2 0.0 0.0

    thr=0.5 → A_thr:
    1 1 0 0
    1 1 0 0
    1 0 0 0
    1 0 0 0

    |A_thr ∩ Y| = 6   |Y| = 8
    AAMO = 6/8 = 0.75
    """
    Y = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.float64,
    )
    A = np.array(
        [
            [0.9, 0.8, 0.1, 0.0],
            [0.7, 0.6, 0.2, 0.0],
            [0.5, 0.4, 0.1, 0.0],
            [0.9, 0.2, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    out = compute_aamo(A, Y, thr=0.5, normalize=False)
    expected = 0.75
    ok = abs(out["aamo"] - expected) < 1e-6
    print("AAMO unit example")
    print("  computed aamo =", round(out["aamo"], 4))
    print("  expected      =", expected)
    print("  aamo_dice     =", round(out["aamo_dice"], 4))
    print("  PASS" if ok else "  FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    _unit_example()
