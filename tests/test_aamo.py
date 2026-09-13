"""
Unit tests for the AAMO metric, against dummy attention/mask arrays only —
no trained model or checkpoint needed.

Run:
    python -m pytest tests/test_aamo.py -v
or:
    python tests/test_aamo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))

from aamo import compute_aamo, compute_aamo_soft, mean_aamo, normalize_attention


def _quadrant_mask(size: int = 8) -> np.ndarray:
    """Top-left quadrant is forest (1), rest is not (0)."""
    y = np.zeros((size, size), dtype=np.float64)
    h, w = size // 2, size // 2
    y[:h, :w] = 1.0
    return y


def test_perfect_overlap_gives_aamo_one():
    y = _quadrant_mask()
    a = y.copy()  # attention exactly matches the mask
    # compute_aamo divides by (|Y| + eps), so a perfect match approaches but
    # never exactly equals 1.0 — tolerance is loosened to account for eps.
    out = compute_aamo(a, y, thr=0.5, normalize=False)
    assert abs(out["aamo"] - 1.0) < 1e-4
    assert abs(out["aamo_dice"] - 1.0) < 1e-4
    assert abs(out["aamo_iou"] - 1.0) < 1e-4


def test_zero_overlap_gives_aamo_zero():
    y = _quadrant_mask()
    a = 1.0 - y  # attention exactly on the complement
    out = compute_aamo(a, y, thr=0.5, normalize=False)
    assert out["aamo"] == 0.0
    assert out["aamo_dice"] == 0.0
    assert out["aamo_iou"] == 0.0


def test_known_4x4_hand_calculation_matches_module_docstring():
    # Same numbers as aamo.py's own _unit_example, kept independent here
    # so a future edit to _unit_example can't silently mask a regression.
    y = np.array(
        [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]], dtype=np.float64
    )
    a = np.array(
        [
            [0.9, 0.8, 0.1, 0.0],
            [0.7, 0.6, 0.2, 0.0],
            [0.5, 0.4, 0.1, 0.0],
            [0.9, 0.2, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    out = compute_aamo(a, y, thr=0.5, normalize=False)
    assert abs(out["aamo"] - 0.75) < 1e-6


def test_normalize_attention_rescales_to_unit_range():
    a = np.array([[2.0, 4.0], [6.0, 8.0]])
    out = normalize_attention(a)
    assert out.min() == 0.0
    assert out.max() == 1.0


def test_normalize_attention_constant_map_returns_zeros_not_nan():
    a = np.full((4, 4), 3.0)
    out = normalize_attention(a)
    assert np.all(out == 0.0)
    assert not np.any(np.isnan(out))


def test_compute_aamo_normalizes_by_default():
    # Raw attention on an arbitrary scale (e.g. un-normalized logits/relevance)
    # should give the same result as pre-normalized attention, since
    # normalize=True is the default.
    y = _quadrant_mask()
    a_raw = y * 100.0 + 5.0  # arbitrary positive scale, same relative ranking
    a_raw[y == 0] = 5.0
    out_raw = compute_aamo(a_raw, y, thr=0.5, normalize=True)
    out_prenormalized = compute_aamo(y.copy(), y, thr=0.5, normalize=False)
    assert abs(out_raw["aamo"] - out_prenormalized["aamo"]) < 1e-9


def test_soft_variant_matches_hard_variant_on_binary_attention():
    y = _quadrant_mask()
    a = y.copy()
    soft = compute_aamo_soft(a, y, normalize=False)
    hard = compute_aamo(a, y, thr=0.5, normalize=False)
    assert abs(soft["aamo_soft"] - hard["aamo"]) < 1e-9


def test_mean_aamo_averages_across_samples():
    y = _quadrant_mask()
    a_perfect = y.copy()
    a_zero = 1.0 - y
    result = mean_aamo([a_perfect, a_zero], [y, y], thr=0.5)
    assert abs(result["aamo"] - 0.5) < 1e-4


def test_shapes_with_batch_and_channel_dims_are_squeezed():
    # rollout.py / model outputs may come through as (1, H, W) or (1, 1, H, W)
    # rather than bare (H, W) — compute_aamo should handle both.
    y = _quadrant_mask()
    a = y.copy()
    out_2d = compute_aamo(a, y, thr=0.5, normalize=False)
    out_3d = compute_aamo(a[np.newaxis, ...], y[np.newaxis, ...], thr=0.5, normalize=False)
    out_4d = compute_aamo(
        a[np.newaxis, np.newaxis, ...], y[np.newaxis, np.newaxis, ...], thr=0.5, normalize=False
    )
    assert out_2d["aamo"] == out_3d["aamo"] == out_4d["aamo"]


def test_precision_recall_are_consistent_with_intersection():
    y = _quadrant_mask()
    a = np.zeros_like(y)
    a[:2, :2] = 1.0  # only top-left 2x2 of the 4x4 forest quadrant attended to
    out = compute_aamo(a, y, thr=0.5, normalize=False)
    # a_thr is fully inside y here, so precision should be 1.0
    assert abs(out["precision_att"] - 1.0) < 1e-6
    # recall = intersection / |Y| = 4 / 16
    assert abs(out["recall_att"] - 4 / 16) < 1e-6


ALL_TESTS = [
    test_perfect_overlap_gives_aamo_one,
    test_zero_overlap_gives_aamo_zero,
    test_known_4x4_hand_calculation_matches_module_docstring,
    test_normalize_attention_rescales_to_unit_range,
    test_normalize_attention_constant_map_returns_zeros_not_nan,
    test_compute_aamo_normalizes_by_default,
    test_soft_variant_matches_hard_variant_on_binary_attention,
    test_mean_aamo_averages_across_samples,
    test_shapes_with_batch_and_channel_dims_are_squeezed,
    test_precision_recall_are_consistent_with_intersection,
]

if __name__ == "__main__":
    failed = 0
    for t in ALL_TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - failed}/{len(ALL_TESTS)} passed")
    if failed:
        raise SystemExit(1)
