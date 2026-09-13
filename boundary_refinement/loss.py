"""
Boundary Dice Loss (Person 5 -- Phase 2 stretch goal, proposal Section 3.4).

    L_boundary = 1 - Dice(boundary(P), boundary(Y))
    L = L_dice + lambda1*L_bce + lambda2*L_att + lambda3*L_boundary

Most forest-segmentation errors occur near canopy edges; L_dice/L_bce
weight interior and boundary pixels equally, so this loss supervises the
boundary band directly. Matches the pattern of AttentionConsistencyLoss/total_objective in
attention_consistency/loss.py -- total_objective_with_boundary here is
meant to *replace* that module's total_objective(...) call once the
Boundary Refinement Module is wired into the training loop.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .boundary_ops import morphological_gradient_boundary


class BoundaryDiceLoss(nn.Module):
    """L_boundary = 1 - Dice(boundary(P), boundary(Y)), per proposal Section 3.4."""

    def __init__(self, kernel_size: int = 3, eps: float = 1e-6):
        super().__init__()
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        pred: (B, H, W) or (B, 1, H, W), predicted forest probability P in [0,1].
        mask: (B, H, W) or (B, 1, H, W), binary ground-truth mask Y.
        """
        b_pred = morphological_gradient_boundary(pred, self.kernel_size)
        b_mask = morphological_gradient_boundary(mask, self.kernel_size)

        b_pred_flat = b_pred.reshape(b_pred.shape[0], -1)
        b_mask_flat = b_mask.reshape(b_mask.shape[0], -1)

        inter = (b_pred_flat * b_mask_flat).sum(dim=1)
        dice = (2 * inter + self.eps) / (
            b_pred_flat.sum(dim=1) + b_mask_flat.sum(dim=1) + self.eps
        )
        return (1 - dice).mean()


def total_objective_with_boundary(
    l_dice: torch.Tensor,
    l_bce: torch.Tensor,
    l_att: torch.Tensor,
    l_boundary: torch.Tensor,
    lambda1: float = 1.0,
    lambda2: float = 0.3,
    lambda3: float = 0.2,
) -> torch.Tensor:
    """L = L_dice + lambda1*L_bce + lambda2*L_att + lambda3*L_boundary.

    lambda3=0.2 is an untuned initial value (proposal Section 3.4 gives no
    default) -- tune on validation alongside lambda1/lambda2 once this is
    integrated into the real training loop.
    """
    return l_dice + lambda1 * l_bce + lambda2 * l_att + lambda3 * l_boundary
