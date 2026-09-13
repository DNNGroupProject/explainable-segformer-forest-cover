"""
Unit tests for the Attention Consistency Loss (Person 3), against dummy
attention maps only — no trained SegFormer needed (per Phase 1 plan,
Person 3 weeks 1-2: "no trained model to attach a loss to yet").

Run:
    python -m pytest tests/test_attention_consistency_loss.py -v
or:
    python tests/test_attention_consistency_loss.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attention_consistency.loss import (
    AttentionConsistencyLoss,
    gaussian_soft_target,
    total_objective,
)


def _quadrant_mask(size: int = 64) -> torch.Tensor:
    """Top-left quadrant is forest (1), rest is not (0)."""
    y = torch.zeros(1, size, size)
    h, w = size // 2, size // 2
    y[:, :h, :w] = 1.0
    return y


def test_gaussian_soft_target_range_and_spread():
    y = _quadrant_mask()
    a_star = gaussian_soft_target(y, sigma=4.0)
    assert a_star.shape == y.shape
    assert torch.all(a_star >= 0) and torch.all(a_star <= 1.0 + 1e-6)
    # Peak should sit inside the forest quadrant (smoothing shouldn't move
    # the maximum outside the original foreground).
    peak_idx = torch.argmax(a_star[0])
    peak_row, peak_col = divmod(int(peak_idx), a_star.shape[-1])
    assert peak_row < a_star.shape[1] // 2
    assert peak_col < a_star.shape[2] // 2
    # Some blur actually happened: a hard boundary pixel just outside the
    # mask should have picked up nonzero soft mass.
    assert a_star[0, a_star.shape[1] // 2, 0] > 0


def test_gaussian_soft_target_all_zero_mask_stays_zero():
    y = torch.zeros(1, 32, 32)
    a_star = gaussian_soft_target(y, sigma=4.0)
    assert torch.allclose(a_star, torch.zeros_like(a_star))


def test_loss_near_zero_when_attention_matches_target():
    y = _quadrant_mask()
    a_star = gaussian_soft_target(y, sigma=4.0)
    for mode in ("mse", "kl"):
        loss_fn = AttentionConsistencyLoss(mode=mode, sigma=4.0)
        loss = loss_fn(a_star, y)
        assert loss.item() < 1e-5, f"{mode}: expected ~0, got {loss.item()}"


def test_loss_penalizes_off_target_attention_more_than_on_target():
    """
    The whole point of L_att: attention placed on roads/shadows (off the
    forest mask) must score worse than attention placed on canopy (on the
    forest mask), for both MSE and KL.
    """
    size = 64
    y = _quadrant_mask(size)  # forest in top-left quadrant

    on_target = torch.zeros(1, size, size)
    on_target[:, : size // 2, : size // 2] = 1.0  # attends to forest

    off_target = torch.zeros(1, size, size)
    off_target[:, size // 2 :, size // 2 :] = 1.0  # attends to bottom-right, no forest there

    for mode in ("mse", "kl"):
        loss_fn = AttentionConsistencyLoss(mode=mode, sigma=4.0)
        l_on = loss_fn(on_target, y).item()
        l_off = loss_fn(off_target, y).item()
        assert l_off > l_on, (
            f"{mode}: off-target loss ({l_off}) should exceed on-target loss ({l_on})"
        )


def test_gradients_flow_to_attention():
    y = _quadrant_mask()
    a = torch.rand(1, 64, 64, requires_grad=True)
    loss_fn = AttentionConsistencyLoss(mode="mse", sigma=4.0)
    loss = loss_fn(a, y)
    loss.backward()
    assert a.grad is not None
    assert torch.any(a.grad != 0)


def test_kl_mode_rejects_bad_string():
    try:
        AttentionConsistencyLoss(mode="not-a-mode")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_total_objective_matches_proposal_formula():
    l_dice = torch.tensor(0.4)
    l_bce = torch.tensor(0.2)
    l_att = torch.tensor(0.1)
    total = total_objective(l_dice, l_bce, l_att, lambda1=1.0, lambda2=0.3)
    expected = 0.4 + 1.0 * 0.2 + 0.3 * 0.1
    assert abs(total.item() - expected) < 1e-6


def test_handles_2d_input_without_batch_dim():
    y = _quadrant_mask()[0]  # (H, W), no batch dim
    a = _quadrant_mask()[0]
    loss_fn = AttentionConsistencyLoss(mode="mse", sigma=4.0)
    loss = loss_fn(a, y)
    assert loss.item() >= 0


ALL_TESTS = [
    test_gaussian_soft_target_range_and_spread,
    test_gaussian_soft_target_all_zero_mask_stays_zero,
    test_loss_near_zero_when_attention_matches_target,
    test_loss_penalizes_off_target_attention_more_than_on_target,
    test_gradients_flow_to_attention,
    test_kl_mode_rejects_bad_string,
    test_total_objective_matches_proposal_formula,
    test_handles_2d_input_without_batch_dim,
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
