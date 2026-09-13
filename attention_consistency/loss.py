"""
Attention Consistency Loss (Person 3 — Loss & Training; proposal §3.3).

    A* = Gaussian(Y)                          soft attention target
    L_att = MSE(A, A*)   or   KL(A || A*)     attention consistency loss
    L = L_dice + λ1 * L_bce + λ2 * L_att      total training objective
        (λ1 = 1, λ2 = 0.3 initial, tuned on validation)

A is the adapted Grad-Rollout attention map (rollout.py), Y is the binary
ground-truth forest mask. Gaussian-smoothing Y turns a hard 0/1 mask into a
soft target so the loss doesn't punish attention for being slightly inside
canopy edges — only for attending to clearly off-target regions (roads,
shadows, buildings).

This module needs no trained model to be developed or unit-tested against
per the Phase 1 plan (Person 3, weeks 1-2: "no trained model to attach a
loss to yet") — gaussian_soft_target and AttentionConsistencyLoss both
operate on plain tensors, exercised here with synthetic dummy attention
maps in tests/test_attention_consistency_loss.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel1d(sigma: float, radius: int, device=None, dtype=None) -> torch.Tensor:
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype or torch.float32)
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_soft_target(
    mask: torch.Tensor, sigma: float = 8.0, normalize: bool = True
) -> torch.Tensor:
    """
    A* = Gaussian(Y) — proposal §3.3.

    mask: (B, H, W) or (B, 1, H, W) binary {0,1} ground-truth forest mask.
    Returns a soft target of the same (B, H, W) shape. sigma=8 spreads mass
    roughly 1-2 canopy-edge pixels' worth at 256x256 (tune experimentally
    per proposal's "to be tuned via validation-set performance").
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # (B,1,H,W)
    mask = mask.float()

    radius = max(1, int(round(3 * sigma)))
    kernel1d = _gaussian_kernel1d(sigma, radius, device=mask.device, dtype=mask.dtype)
    kernel_x = kernel1d.view(1, 1, 1, -1)
    kernel_y = kernel1d.view(1, 1, -1, 1)

    smoothed = F.conv2d(mask, kernel_x, padding=(0, radius))
    smoothed = F.conv2d(smoothed, kernel_y, padding=(radius, 0))

    if normalize:
        b = smoothed.shape[0]
        flat = smoothed.view(b, -1)
        vmax = flat.max(dim=1, keepdim=True).values.clamp_min(1e-8)
        smoothed = (flat / vmax).view_as(smoothed)

    return smoothed.squeeze(1)  # (B, H, W)


class AttentionConsistencyLoss(nn.Module):
    """L_att = MSE(A, A*) or KL(A || A*), per proposal §3.3."""

    def __init__(self, mode: str = "mse", sigma: float = 8.0, eps: float = 1e-8):
        super().__init__()
        if mode not in ("mse", "kl"):
            raise ValueError(f"mode must be 'mse' or 'kl', got {mode!r}")
        self.mode = mode
        self.sigma = sigma
        self.eps = eps

    def forward(self, attention: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        attention: (B, H, W) or (H, W), values in [0, 1] (A from rollout.py).
        mask: (B, H, W) or (H, W) binary ground-truth forest mask (Y).
        """
        if attention.dim() == 2:
            attention = attention.unsqueeze(0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        target = gaussian_soft_target(mask, sigma=self.sigma).to(attention.dtype)

        if self.mode == "mse":
            return F.mse_loss(attention, target)

        # KL(A || A*): treat both as unnormalized densities over the image,
        # normalize each sample to sum to 1, then standard KL divergence.
        b = attention.shape[0]
        a_flat = attention.reshape(b, -1).clamp_min(self.eps)
        t_flat = target.reshape(b, -1).clamp_min(self.eps)
        a_prob = a_flat / a_flat.sum(dim=1, keepdim=True)
        t_prob = t_flat / t_flat.sum(dim=1, keepdim=True)
        kl = (a_prob * (a_prob.log() - t_prob.log())).sum(dim=1)
        return kl.mean()


def total_objective(
    l_dice: torch.Tensor,
    l_bce: torch.Tensor,
    l_att: torch.Tensor,
    lambda1: float = 1.0,
    lambda2: float = 0.3,
) -> torch.Tensor:
    """L = L_dice + λ1·L_bce + λ2·L_att — proposal §3.3 total objective."""
    return l_dice + lambda1 * l_bce + lambda2 * l_att
