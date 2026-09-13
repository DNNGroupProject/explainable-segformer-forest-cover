"""
Unit tests for the Boundary Refinement Module (Person 5), against dummy
tensors only -- no trained model needed (proposal Section 3.4 is a Phase 2
stretch goal; per Person 3's Phase 1 pattern, this is developed and
unit-tested standalone before it's wired into the real training loop).

Run:
    python -m pytest tests/test_boundary_refinement.py -v
or:
    python tests/test_boundary_refinement.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boundary_refinement.boundary_ops import morphological_gradient_boundary
from boundary_refinement.loss import BoundaryDiceLoss, total_objective_with_boundary


def _quadrant_mask(size: int = 16) -> torch.Tensor:
    """Top-left quadrant is forest (1), rest is not (0). (1, H, W)."""
    y = torch.zeros(1, size, size)
    h, w = size // 2, size // 2
    y[:, :h, :w] = 1.0
    return y


def test_kernel_size_must_be_odd():
    y = _quadrant_mask()
    try:
        morphological_gradient_boundary(y, kernel_size=4)
        raise AssertionError("expected ValueError for even kernel_size")
    except ValueError:
        pass


def test_uniform_mask_has_no_boundary():
    zeros = torch.zeros(1, 16, 16)
    ones = torch.ones(1, 16, 16)
    b0 = morphological_gradient_boundary(zeros, kernel_size=3)
    b1 = morphological_gradient_boundary(ones, kernel_size=3)
    assert torch.all(b0 == 0)
    assert torch.all(b1 == 0)


def test_boundary_highlights_quadrant_edge_not_interior():
    y = _quadrant_mask(size=16)  # forest quadrant is rows/cols [0:8)
    b = morphological_gradient_boundary(y, kernel_size=3)
    # Deep interior of the forest quadrant, away from any edge -> no boundary.
    assert b[0, 3, 3].item() == 0.0
    # Deep interior of the non-forest region, away from any edge -> no boundary.
    assert b[0, 12, 12].item() == 0.0
    # Right at the quadrant's edge -> boundary present.
    assert b[0, 7, 3].item() > 0.0
    assert b[0, 3, 7].item() > 0.0


def test_boundary_values_stay_in_unit_range():
    torch.manual_seed(0)
    soft = torch.rand(2, 1, 16, 16)  # arbitrary soft "prediction" in [0,1)
    b = morphological_gradient_boundary(soft, kernel_size=3)
    assert torch.all(b >= 0) and torch.all(b <= 1.0 + 1e-6)


def test_boundary_dice_near_zero_when_boundaries_match():
    y = _quadrant_mask()
    loss_fn = BoundaryDiceLoss(kernel_size=3)
    loss = loss_fn(y.clone(), y)
    assert loss.item() < 1e-4


def test_boundary_dice_positive_when_boundary_missing():
    y = _quadrant_mask()
    flat_pred = torch.zeros_like(y)  # no edges anywhere -> boundary(pred) is all 0
    loss_fn = BoundaryDiceLoss(kernel_size=3)
    loss = loss_fn(flat_pred, y)
    assert loss.item() > 0.9  # near-total mismatch -> loss near 1


def test_boundary_dice_worse_for_shifted_mask_than_matched_mask():
    y = _quadrant_mask(size=16)
    shifted = torch.roll(y, shifts=4, dims=2)  # same-shaped quadrant, moved
    loss_fn = BoundaryDiceLoss(kernel_size=3)
    loss_matched = loss_fn(y.clone(), y)
    loss_shifted = loss_fn(shifted, y)
    assert loss_shifted.item() > loss_matched.item()


def test_gradients_flow_to_prediction():
    y = _quadrant_mask()
    pred = _quadrant_mask().clone().requires_grad_(True)
    loss_fn = BoundaryDiceLoss(kernel_size=3)
    loss = loss_fn(pred, y)
    loss.backward()
    assert pred.grad is not None
    assert torch.any(pred.grad != 0)


def test_handles_2d_and_4d_input_consistently():
    y = _quadrant_mask()  # (1, H, W)
    loss_fn = BoundaryDiceLoss(kernel_size=3)
    loss_3d = loss_fn(y.clone(), y)
    loss_4d = loss_fn(y.clone().unsqueeze(1), y.unsqueeze(1))  # (1, 1, H, W)
    assert abs(loss_3d.item() - loss_4d.item()) < 1e-6


def test_total_objective_with_boundary_matches_formula():
    l_dice = torch.tensor(0.4)
    l_bce = torch.tensor(0.2)
    l_att = torch.tensor(0.1)
    l_boundary = torch.tensor(0.3)
    total = total_objective_with_boundary(
        l_dice, l_bce, l_att, l_boundary, lambda1=1.0, lambda2=0.3, lambda3=0.2
    )
    expected = 0.4 + 1.0 * 0.2 + 0.3 * 0.1 + 0.2 * 0.3
    assert abs(total.item() - expected) < 1e-6


ALL_TESTS = [
    test_kernel_size_must_be_odd,
    test_uniform_mask_has_no_boundary,
    test_boundary_highlights_quadrant_edge_not_interior,
    test_boundary_values_stay_in_unit_range,
    test_boundary_dice_near_zero_when_boundaries_match,
    test_boundary_dice_positive_when_boundary_missing,
    test_boundary_dice_worse_for_shifted_mask_than_matched_mask,
    test_gradients_flow_to_prediction,
    test_handles_2d_and_4d_input_consistently,
    test_total_objective_with_boundary_matches_formula,
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
