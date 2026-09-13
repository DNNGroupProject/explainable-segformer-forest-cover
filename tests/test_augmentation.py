"""Unit tests for shared/augmentation.py, on synthetic arrays only -- no
dataset, no GPU, no internet (same constraint as Person 3's and Person 5's
tests, and as tests/test_mask_audit.py in Phase 2).

The ones that matter most are the pairing tests: a segmentation augmentation
that moves the image but not the mask still trains, still produces a loss
curve, and quietly teaches the model nothing. Nothing downstream would catch
it -- so it gets caught here.

Run:
    python -m pytest tests/test_augmentation.py -v
or:
    python tests/test_augmentation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# shared/ is at the repo root, one level above this file's folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.augmentation import (  # noqa: E402
    DEFAULT_CONFIG,
    NO_AUGMENT,
    AugmentConfig,
    augment_pair,
)

# Geometry only -- no brightness/contrast, so the image survives a flip
# unchanged and can be compared to the mask pixel for pixel.
GEOMETRY_ONLY = AugmentConfig(brightness_delta=0.0, contrast_range=(1.0, 1.0))

# One op at a time, for the tests that need to know which one fired.
HFLIP_ONLY = AugmentConfig(
    flip_horizontal_p=1.0, flip_vertical_p=0.0, rot90=False,
    brightness_delta=0.0, contrast_range=(1.0, 1.0),
)
PHOTOMETRY_ONLY = AugmentConfig(flip_horizontal_p=0.0, flip_vertical_p=0.0, rot90=False)


def _pair(size: int = 8):
    """An image whose red channel *is* the mask.

    That coupling is the whole trick: any geometric op that lands on one array
    and not the other shows up as image[..., 0] != mask.
    """
    mask = (np.arange(size * size).reshape(size, size) % 3 == 0).astype(np.int64)
    image = np.zeros((size, size, 3), dtype=np.float32)
    image[..., 0] = mask.astype(np.float32)
    image[..., 1] = 0.25
    image[..., 2] = 0.75
    return image, mask


# ---- image and mask must move together --------------------------------

def test_geometry_keeps_image_and_mask_aligned():
    image, mask = _pair()
    # 200 draws so every flip/rot90 combination gets exercised.
    rng = np.random.default_rng(0)
    for _ in range(200):
        out_img, out_mask = augment_pair(image, mask, rng, GEOMETRY_ONLY)
        assert np.array_equal(out_img[..., 0], out_mask.astype(np.float32))


def test_geometry_actually_moves_something():
    # Guards the test above: if augment_pair were a no-op it would also pass.
    image, mask = _pair()
    rng = np.random.default_rng(0)
    moved = sum(
        not np.array_equal(augment_pair(image, mask, rng, GEOMETRY_ONLY)[1], mask)
        for _ in range(50)
    )
    assert moved > 0


def test_horizontal_flip_is_a_left_right_mirror():
    image, mask = _pair()
    out_img, out_mask = augment_pair(image, mask, np.random.default_rng(0), HFLIP_ONLY)
    assert np.array_equal(out_mask, np.fliplr(mask))
    assert np.array_equal(out_img, np.fliplr(image))


def test_all_four_rotations_occur():
    # A hardcoded k would still pass every alignment test above, so check the
    # rotation is actually sampled: over many draws all 4 orientations appear.
    image, mask = _pair()
    rot_only = AugmentConfig(
        flip_horizontal_p=0.0, flip_vertical_p=0.0, rot90=True,
        brightness_delta=0.0, contrast_range=(1.0, 1.0),
    )
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(100):
        _, out_mask = augment_pair(image, mask, rng, rot_only)
        for k in range(4):
            if np.array_equal(out_mask, np.rot90(mask, k)):
                seen.add(k)
    assert seen == {0, 1, 2, 3}, f"only saw rotations {sorted(seen)}"


def test_flips_fire_roughly_half_the_time():
    # p=0.5 should mean p=0.5. A wrong comparison direction (<= vs <) or a
    # swapped probability would show up as a badly skewed count.
    image, mask = _pair()
    hflip = AugmentConfig(
        flip_horizontal_p=0.5, flip_vertical_p=0.0, rot90=False,
        brightness_delta=0.0, contrast_range=(1.0, 1.0),
    )
    rng = np.random.default_rng(2)
    n = 1000
    # _pair()'s mask isn't left-right symmetric, so "equals fliplr(mask)"
    # identifies a flip unambiguously.
    flipped = sum(
        np.array_equal(augment_pair(image, mask, rng, hflip)[1], np.fliplr(mask))
        for _ in range(n)
    )
    assert 0.4 * n < flipped < 0.6 * n, f"{flipped}/{n} flipped"


# ---- the mask stays a label map ---------------------------------------

def test_mask_stays_exactly_binary():
    # The reason the op list is flips + rot90 only: no interpolation, so no
    # new values can appear. See the note in shared/augmentation.py.
    image, mask = _pair()
    rng = np.random.default_rng(3)
    for _ in range(100):
        _, out_mask = augment_pair(image, mask, rng, DEFAULT_CONFIG)
        assert set(np.unique(out_mask).tolist()) <= {0, 1}


def test_mask_dtype_and_shape_survive():
    image, mask = _pair()
    out_img, out_mask = augment_pair(image, mask, np.random.default_rng(4), DEFAULT_CONFIG)
    assert out_mask.dtype == mask.dtype
    assert out_mask.shape == mask.shape
    assert out_img.shape == image.shape
    assert out_img.dtype == np.float32


def test_photometry_never_touches_the_mask():
    image, mask = _pair()
    rng = np.random.default_rng(5)
    for _ in range(20):
        out_img, out_mask = augment_pair(image, mask, rng, PHOTOMETRY_ONLY)
        assert np.array_equal(out_mask, mask)
        assert not np.array_equal(out_img, image)  # the image did change


def test_image_stays_in_unit_range():
    # Brightness and contrast both push pixels out of [0,1]; the clip has to
    # bring them back, or the ImageNet normalize downstream sees garbage.
    image, mask = _pair()
    image[..., 0] = 0.99  # start near the ceiling so a shift would overflow
    strong = AugmentConfig(
        flip_horizontal_p=0.0, flip_vertical_p=0.0, rot90=False,
        brightness_delta=0.5, contrast_range=(0.5, 2.0),
    )
    rng = np.random.default_rng(6)
    for _ in range(50):
        out_img, _ = augment_pair(image, mask, rng, strong)
        assert out_img.min() >= 0.0 and out_img.max() <= 1.0


# ---- reproducibility ---------------------------------------------------

def test_same_seed_gives_the_same_result():
    image, mask = _pair()
    a = augment_pair(image, mask, np.random.default_rng(42), DEFAULT_CONFIG)
    b = augment_pair(image, mask, np.random.default_rng(42), DEFAULT_CONFIG)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_different_seeds_diverge():
    image, mask = _pair()
    rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(2)
    outs_a = [augment_pair(image, mask, rng_a, DEFAULT_CONFIG)[0] for _ in range(10)]
    outs_b = [augment_pair(image, mask, rng_b, DEFAULT_CONFIG)[0] for _ in range(10)]
    assert any(not np.array_equal(x, y) for x, y in zip(outs_a, outs_b))


def test_no_augment_config_is_the_identity():
    # The "without augmentation" arm of the ablation depends on this being
    # exact -- not "close enough".
    image, mask = _pair()
    rng = np.random.default_rng(7)
    for _ in range(20):
        out_img, out_mask = augment_pair(image, mask, rng, NO_AUGMENT)
        assert np.array_equal(out_img, image)
        assert np.array_equal(out_mask, mask)


def test_inputs_are_not_modified_in_place():
    # The DataLoader hands out the same arrays every epoch; mutating them
    # would compound augmentations run after run.
    image, mask = _pair()
    image_before, mask_before = image.copy(), mask.copy()
    augment_pair(image, mask, np.random.default_rng(8), DEFAULT_CONFIG)
    assert np.array_equal(image, image_before)
    assert np.array_equal(mask, mask_before)


def test_mismatched_shapes_are_rejected():
    image, mask = _pair(8)
    try:
        augment_pair(image, mask[:4], np.random.default_rng(9), DEFAULT_CONFIG)
        raise AssertionError("expected ValueError for mismatched image/mask")
    except ValueError:
        pass


def test_dataloader_workers_get_independent_streams():
    # Forked workers inherit a copy of the dataset's generator, so without
    # per-worker seeding they all replay the same augmentations -- the run
    # trains fine and logs a normal curve while seeing a fraction of the
    # variety it claims. This is the guard against that.
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from dataset import worker_rng

    streams = [worker_rng(42, wid).random(20) for wid in range(4)]
    for a in range(len(streams)):
        for b in range(a + 1, len(streams)):
            assert not np.array_equal(streams[a], streams[b]), f"workers {a},{b} match"

    # ...but each worker is still reproducible from the same (seed, id).
    assert np.array_equal(worker_rng(42, 1).random(20), worker_rng(42, 1).random(20))


def test_output_is_contiguous():
    # np.fliplr/rot90 return negative-stride views and torch.from_numpy
    # refuses those -- this is the test that catches a missing copy.
    import torch

    image, mask = _pair()
    rng = np.random.default_rng(10)
    for _ in range(20):
        out_img, out_mask = augment_pair(image, mask, rng, DEFAULT_CONFIG)
        torch.from_numpy(out_img)
        torch.from_numpy(out_mask)


ALL_TESTS = [
    test_geometry_keeps_image_and_mask_aligned,
    test_geometry_actually_moves_something,
    test_horizontal_flip_is_a_left_right_mirror,
    test_all_four_rotations_occur,
    test_flips_fire_roughly_half_the_time,
    test_mask_stays_exactly_binary,
    test_mask_dtype_and_shape_survive,
    test_photometry_never_touches_the_mask,
    test_image_stays_in_unit_range,
    test_same_seed_gives_the_same_result,
    test_different_seeds_diverge,
    test_no_augment_config_is_the_identity,
    test_inputs_are_not_modified_in_place,
    test_mismatched_shapes_are_rejected,
    test_dataloader_workers_get_independent_streams,
    test_output_is_contiguous,
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
