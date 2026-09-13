"""Tests for unet_model.py's loader helpers and the committed fixture.

No GPU, no dataset, no internet -- everything here is either a freshly built
model or the 0.5 MB fixture checkpoint in this folder.

The load-time tests matter because Person 4's adapter will be written against
these helpers, and the failure they guard against is silent: a loader that
quietly builds the wrong architecture, or that reads the fixture as if it were
the trained baseline, produces plausible-looking numbers that mean nothing.

Run:
    python -m pytest tests/test_unet_model.py -v
or:
    python tests/test_unet_model.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from unet_model import (  # noqa: E402
    UNet,
    features_from_state_dict,
    load_unet,
)

FIXTURE = HERE / "fixtures" / "unet_fixture_random.pt"

# The committed baseline's shape. Both numbers appear in test_metrics.txt and
# in the paper draft, so a change here means one of those is now wrong.
REAL_FEATURES = (64, 128, 256, 512)
REAL_PARAM_COUNT = 31_037_698


# ---- the model still matches the notebook -----------------------------

def test_default_unet_matches_the_committed_baseline():
    assert sum(p.numel() for p in UNet().parameters()) == REAL_PARAM_COUNT


# ---- width inference ---------------------------------------------------

def test_features_are_recovered_from_a_state_dict():
    for features in [(4, 8, 16, 32), (16, 32, 64, 128), REAL_FEATURES]:
        sd = UNet(features=features).state_dict()
        assert features_from_state_dict(sd) == features


def test_features_from_a_non_unet_dict_is_an_error():
    try:
        features_from_state_dict({"something.else": torch.zeros(1)})
        raise AssertionError("expected ValueError for a non-U-Net state_dict")
    except ValueError:
        pass


# ---- loading -----------------------------------------------------------

def test_load_unet_round_trips_weights_exactly():
    model = UNet(features=(4, 8, 16, 32))
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ckpt.pt"
        torch.save(model.state_dict(), path)
        loaded = load_unet(path)
    for k, v in model.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k]), k


def test_load_unet_returns_an_eval_mode_model():
    # BatchNorm behaves differently in train mode, so a loader that forgets
    # .eval() gives different predictions on every call.
    assert not load_unet(FIXTURE).training


def test_load_unet_accepts_half_precision():
    # However the real checkpoint ends up being shipped -- fp32, fp16, or a
    # partial cast -- the adapter shouldn't have to change.
    model = UNet(features=(4, 8, 16, 32))
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "half.pt"
        torch.save({k: (v.half() if v.is_floating_point() else v)
                    for k, v in model.state_dict().items()}, path)
        loaded = load_unet(path)
    assert loaded.final_conv.weight.dtype == torch.float32


# ---- the fixture -------------------------------------------------------

def test_fixture_exists_and_is_small():
    assert FIXTURE.exists(), "run make_fixture_checkpoint.py"
    assert FIXTURE.stat().st_size < 2_000_000, "fixture should stay under 2 MB"


def test_fixture_loads_and_runs():
    model = load_unet(FIXTURE)
    with torch.no_grad():
        out = model(torch.randn(1, 3, 256, 256))
    assert out.shape == (1, 2, 256, 256)


def test_fixture_cannot_be_loaded_as_the_real_baseline():
    # This is the safeguard, not a quirk: the fixture is deliberately a
    # different width, so anything that assumes the trained architecture
    # fails loudly here instead of reporting noise as a baseline result.
    sd = torch.load(FIXTURE, map_location="cpu")
    assert features_from_state_dict(sd) != REAL_FEATURES
    try:
        UNet(features=REAL_FEATURES).load_state_dict(sd)
        raise AssertionError("fixture loaded into the real architecture -- the guard is gone")
    except RuntimeError:
        pass


ALL_TESTS = [
    test_default_unet_matches_the_committed_baseline,
    test_features_are_recovered_from_a_state_dict,
    test_features_from_a_non_unet_dict_is_an_error,
    test_load_unet_round_trips_weights_exactly,
    test_load_unet_returns_an_eval_mode_model,
    test_load_unet_accepts_half_precision,
    test_fixture_exists_and_is_small,
    test_fixture_loads_and_runs,
    test_fixture_cannot_be_loaded_as_the_real_baseline,
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
