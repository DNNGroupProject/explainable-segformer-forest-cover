"""Adapter tests for unet_torch.py against the tiny fixture checkpoint.

Does not load unet_baseline_best.pt (59 MB). Run:

    python tests/test_unet_torch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
sys.path.insert(0, str(REPO_ROOT))

from adapters.unet_torch import UnetTorchAdapter  # noqa: E402
from unet_model import features_from_state_dict  # noqa: E402

FIXTURE = HERE / "fixtures" / "unet_fixture_random.pt"
REAL_FEATURES = (64, 128, 256, 512)


def test_fixture_loads_without_error():
    ad = UnetTorchAdapter()
    ad.load(str(FIXTURE))
    assert ad.model is not None
    assert not ad.model.training


def test_fixture_output_shape():
    ad = UnetTorchAdapter()
    ad.load(str(FIXTURE))
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = ad.model(x)
    assert out.shape == (2, 2, 256, 256)


def test_fixture_is_not_the_baseline_width():
    sd = torch.load(FIXTURE, map_location="cpu")
    assert features_from_state_dict(sd) != REAL_FEATURES


def test_imagenet_normalize_is_applied():
    """U-Net was trained with ImageNet mean/std; adapter must not use plain /255."""
    ad = UnetTorchAdapter()
    rgb = np.ones((1, 256, 256, 3), dtype=np.float32)
    t = ad._to_tensor(rgb)
    assert t.shape == (1, 3, 256, 256)
    assert abs(float(t[0, 0, 0, 0]) - (1.0 - 0.485) / 0.229) < 1e-5


ALL = [
    test_fixture_loads_without_error,
    test_fixture_output_shape,
    test_fixture_is_not_the_baseline_width,
    test_imagenet_normalize_is_applied,
]


if __name__ == "__main__":
    failed = 0
    for t in ALL:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(ALL) - failed}/{len(ALL)} passed")
    if failed:
        raise SystemExit(1)
