"""
Morphological-gradient boundary extraction (Person 5 -- Phase 2 stretch
goal, proposal Section 3.4: Boundary Refinement Module).

Implemented as pure tensor ops, needing no trained model to develop or
unit-test against -- the same pattern Person 3 used for the Attention
Consistency Loss in Phase 1 (dummy tensors only; see tests/).

Why max/min pooling instead of binary morphology
--------------------------------------------------------------------------
Classic morphological dilation/erosion operate on binary images and aren't
differentiable. L_boundary must backpropagate into the segmentation
decoder through the *predicted* (soft, [0,1]-valued) mask P, not just the
hard ground-truth mask Y, so this module implements dilation/erosion as
max/min pooling instead:

    dilate(x) = max_pool2d(x, k, stride=1, padding=k//2)
    erode(x)  = -max_pool2d(-x, k, stride=1, padding=k//2)   (= min-pool)
    boundary(x) = dilate(x) - erode(x)

On a binary input this reduces to the standard morphological gradient (a
k-pixel-wide ring of 1s around each region's edge); on a soft input it
stays differentiable via max_pool2d's subgradient.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _dilate(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    pad = kernel_size // 2
    return F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)


def _erode(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    pad = kernel_size // 2
    return -F.max_pool2d(-x, kernel_size=kernel_size, stride=1, padding=pad)


def morphological_gradient_boundary(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    boundary(x) = dilation(x) - erosion(x) -- proposal Section 3.4.

    mask: (B, H, W) or (B, 1, H, W), values in [0,1] (binary Y or soft P).
    Returns a boundary map of the same rank as the input, in [0,1].

    kernel_size must be odd (structuring-element size); default 3 gives a
    roughly 1-pixel-wide morphological gradient, matching typical
    canopy-edge scale relative to a 256x256 forest mask. Larger values
    (5, 7, ...) widen the boundary band the loss supervises.
    """
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    if mask.dim() not in (3, 4):
        raise ValueError(f"mask must be (B,H,W) or (B,1,H,W), got shape {tuple(mask.shape)}")

    squeeze_back = False
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # (B,1,H,W)
        squeeze_back = True

    mask = mask.float()
    dil = _dilate(mask, kernel_size)
    ero = _erode(mask, kernel_size)
    boundary = (dil - ero).clamp(0, 1)

    return boundary.squeeze(1) if squeeze_back else boundary
