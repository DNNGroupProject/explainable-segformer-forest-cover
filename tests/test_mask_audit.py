"""Unit tests for mask_audit.py, on synthetic arrays only -- no dataset,
no GPU, no internet.

The important ones are the binarisation-delta tests: they check the metric
reports ~0 on data that is already binary AND a large value on genuinely
gray data. Without the second case a near-zero reading on the real dataset
would be meaningless -- it could just mean the metric never fires.

Run:
    python -m pytest tests/test_mask_audit.py -v
or:
    python tests/test_mask_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from mask_audit import (
    DELTA_TOL,
    MASK_THRESHOLD,
    MIDGRAY_HI,
    MIDGRAY_LO,
    binarisation_delta,
    build_rows,
    find_mask_path,
    mask_name_for,
    summarise_histogram,
)


def _hist(values: dict[int, int]) -> np.ndarray:
    """Build a fake 256-bin histogram from {pixel value: how many}."""
    h = np.zeros(256, dtype=np.int64)
    for value, count in values.items():
        h[value] = count
    return h


# ---- naming / pairing -------------------------------------------------
# The audit is only correct if it finds the right mask for each image, so
# these cover the _sat_ -> _mask_ rename and the extension fallback.

def test_mask_name_follows_sat_to_mask_convention():
    assert mask_name_for("10452_sat_08.jpg") == "10452_mask_08.jpg"


def test_mask_name_left_alone_when_no_sat_marker():
    assert mask_name_for("something_else.png") == "something_else.png"


def test_find_mask_path_falls_back_to_other_extension():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        mask_dir = Path(d)
        (mask_dir / "77_mask_01.png").write_bytes(b"")
        # Image is .jpg but the mask on disk is .png -- must still be found.
        found = find_mask_path(mask_dir, "77_sat_01.jpg")
        assert found is not None and found.name == "77_mask_01.png"


def test_find_mask_path_returns_none_when_absent():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        assert find_mask_path(Path(d), "77_sat_01.jpg") is None


# ---- histogram summary ------------------------------------------------
# Feed in histograms whose answers are obvious by hand, so the percentages
# reported for the real dataset can be trusted.

def test_summarise_histogram_splits_pure_black_and_white():
    s = summarise_histogram(_hist({0: 300, 255: 700}))
    assert s["pixels_total"] == 1000
    assert abs(s["pct_exactly_0"] - 30.0) < 1e-9
    assert abs(s["pct_exactly_255"] - 70.0) < 1e-9
    assert s["pct_midgray"] == 0.0
    assert s["distinct_values"] == 2
    assert s["max_drift_toward_middle"] == 0


def test_drift_measures_distance_from_nearest_extreme():
    # 9 and 246 are both 9 levels from an extreme; 246 must not be read as 246.
    s = summarise_histogram(_hist({0: 10, 9: 10, 246: 10, 255: 10}))
    assert s["max_drift_toward_middle"] == 9
    assert s["distinct_values"] == 4


def test_drift_peaks_at_the_threshold_itself():
    s = summarise_histogram(_hist({127: 5, 128: 5}))
    assert s["max_drift_toward_middle"] == 127


def test_summarise_histogram_counts_midgray_band():
    mid = (MIDGRAY_LO + MIDGRAY_HI) // 2
    s = summarise_histogram(_hist({0: 900, mid: 100}))
    assert abs(s["pct_midgray"] - 10.0) < 1e-9


def test_summarise_histogram_excludes_near_extremes_from_midgray():
    # Values just off 0/255 are JPEG rounding, not ambiguity -- they belong in
    # the off-black / off-white buckets but must stay out of the mid-gray band.
    s = summarise_histogram(_hist({3: 500, 252: 500}))
    assert s["pct_midgray"] == 0.0
    assert abs(s["pct_1_to_threshold"] - 50.0) < 1e-9
    assert abs(s["pct_threshold_to_254"] - 50.0) < 1e-9


def test_threshold_value_counts_as_background_not_foreground():
    # 127 itself is the trap: it looks mid-range but `mask > 127` sends it to
    # background, so it must land in the lower bucket. 128 is the first
    # foreground value.
    s = summarise_histogram(_hist({MASK_THRESHOLD: 100}))
    assert s["pct_1_to_threshold"] == 100.0
    assert s["pct_threshold_to_254"] == 0.0

    s = summarise_histogram(_hist({MASK_THRESHOLD + 1: 100}))
    assert s["pct_1_to_threshold"] == 0.0
    assert s["pct_threshold_to_254"] == 100.0


def test_buckets_account_for_every_pixel():
    # The four buckets must partition 0..255 exactly -- no pixel counted twice,
    # none dropped. This is what broke when the split was off by one.
    s = summarise_histogram(_hist({0: 10, 5: 10, MASK_THRESHOLD: 10, 200: 10, 250: 10, 255: 10}))
    covered = (
        s["pct_exactly_0"]
        + s["pct_exactly_255"]
        + s["pct_1_to_threshold"]
        + s["pct_threshold_to_254"]
    )
    assert abs(covered - 100.0) < 1e-9


def test_summarise_histogram_rejects_empty():
    try:
        summarise_histogram(np.zeros(256, dtype=np.int64))
        raise AssertionError("expected ValueError for an empty histogram")
    except ValueError:
        pass


# ---- binarisation delta ----------------------------------------------
# The three cases below go from "already binary" to "genuinely gray". The
# last one is the important one: it proves a near-zero reading on the real
# masks means the data is clean, not that the metric is broken.

def test_delta_is_zero_for_a_perfectly_binary_mask():
    mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    mean_delta, frac_over = binarisation_delta(mask)
    assert mean_delta == 0.0
    assert frac_over == 0.0


def test_delta_is_small_for_jpeg_style_near_extremes():
    # The shape the real dataset takes: everything within a few levels of 0/255.
    mask = np.array([[1, 254], [3, 252]], dtype=np.uint8)
    mean_delta, frac_over = binarisation_delta(mask)
    assert mean_delta < 0.02
    assert frac_over == 0.0


def test_delta_is_large_for_genuinely_gray_mask():
    # Proves the metric fires when it should: a 128-valued mask thresholds to 1
    # but scales to ~0.502, so every pixel disagrees by ~0.498.
    mask = np.full((8, 8), 128, dtype=np.uint8)
    mean_delta, frac_over = binarisation_delta(mask)
    assert mean_delta > DELTA_TOL
    assert frac_over == 1.0


def test_threshold_boundary_is_exclusive():
    # 127 is background (> 127 is the repo-wide rule), 128 is forest.
    assert binarisation_delta(np.array([[MASK_THRESHOLD]], dtype=np.uint8))[0] > 0.49
    below = (np.array([[MASK_THRESHOLD]], dtype=np.uint8) > MASK_THRESHOLD).item()
    above = (np.array([[MASK_THRESHOLD + 1]], dtype=np.uint8) > MASK_THRESHOLD).item()
    assert below is False and above is True


# ---- report shape -----------------------------------------------------
# Catches a renamed dict key breaking the .csv/.md writers at run time.

def test_build_rows_is_three_columns_and_non_empty():
    result = {
        "images_found": 2, "masks_found": 2, "masks_sampled": 2,
        "orphan_images": [], "orphan_masks": [], "unreadable": [],
        "mask_sizes": [(256, 256)], "pixels_total": 100,
        "pct_exactly_0": 40.0, "pct_exactly_255": 60.0,
        "pct_1_to_threshold": 0.0, "pct_threshold_to_254": 0.0, "pct_midgray": 0.0,
        "distinct_values": 2, "max_drift_toward_middle": 0,
        "soft_vs_hard_mean_delta": 0.0, "soft_vs_hard_pct_over_tol": 0.0,
        "forest_ratio_mean": 0.6, "forest_ratio_std": 0.1,
    }
    rows = build_rows(result)
    assert rows and all(len(r) == 3 for r in rows)
    assert any("256x256" in value for _, value, _ in rows)


ALL_TESTS = [
    test_mask_name_follows_sat_to_mask_convention,
    test_mask_name_left_alone_when_no_sat_marker,
    test_find_mask_path_falls_back_to_other_extension,
    test_find_mask_path_returns_none_when_absent,
    test_summarise_histogram_splits_pure_black_and_white,
    test_drift_measures_distance_from_nearest_extreme,
    test_drift_peaks_at_the_threshold_itself,
    test_summarise_histogram_counts_midgray_band,
    test_summarise_histogram_excludes_near_extremes_from_midgray,
    test_threshold_value_counts_as_background_not_foreground,
    test_buckets_account_for_every_pixel,
    test_summarise_histogram_rejects_empty,
    test_delta_is_zero_for_a_perfectly_binary_mask,
    test_delta_is_small_for_jpeg_style_near_extremes,
    test_delta_is_large_for_genuinely_gray_mask,
    test_threshold_boundary_is_exclusive,
    test_build_rows_is_three_columns_and_non_empty,
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
