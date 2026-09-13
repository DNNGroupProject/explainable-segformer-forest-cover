"""
Adapted Gradient-weighted Attention Rollout for SegFormer-B0
(Person 2 — Transformer Lead task; proposal §3.2).

--------------------------------------------------------------------------
Why plain Attention Rollout (Abnar & Zuidema, 2020) does not apply as-is
--------------------------------------------------------------------------
Standard rollout recursively multiplies row-normalized attention matrices
across layers:

    A_hat_l = 0.5*A_l + 0.5*I         (residual connection)
    A_hat_l = A_hat_l / rowsum(A_hat_l)
    R_l     = A_hat_l @ R_{l-1},  R_0 = I

This recursion is only well-defined when every layer's attention matrix is
square (N x N) over the *same* N tokens, as in vanilla ViT. SegFormer's
MiT-B0 encoder breaks that assumption twice over:

  1. Across stages, patch-merging changes the token grid entirely
     (64x64 -> 32x32 -> 16x16 -> 8x8 for a 256x256 input) — there is no
     single N shared across the whole encoder to roll out over.
  2. Within stages 1-3, "Efficient Self-Attention" spatially reduces the
     keys/values by sr_ratio (8, 4, 2), so even a single layer's attention
     matrix is rectangular (N_query x N_query/sr^2), not square — verified
     empirically for MiT-B0 on a 256x256 input:

         stage 1: (heads=1, 4096, 64)   sr=8
         stage 2: (heads=2, 1024, 64)   sr=4
         stage 3: (heads=5,  256, 64)   sr=2
         stage 4: (heads=8,   64, 64)   sr=1  <-- square

--------------------------------------------------------------------------
The adaptation (proposal §3.2)
--------------------------------------------------------------------------
Restrict rollout to encoder stage 4, the only stage where sr_ratio=1 makes
the attention matrix square and token-consistent across its blocks, so the
standard rollout recursion is mathematically valid without further
approximation. Within that stage, each block's attention is weighted by
the gradient of a target scalar w.r.t. that attention map before rolling
out — Chefer, Gur & Wolf's Gradient-weighted Rollout, "Transformer
Interpretability Beyond Attention Visualization" (arXiv:2101.03919, already
in this project's literature review). Their method reads relevance off the
CLS-token row of the final matrix; SegFormer has no CLS token, so this
adaptation reads relevance as the row-mean of the final matrix instead —
each token's average received relevance across all 64 stage-4 queries.

The result is an 8x8 relevance grid (stage-4 resolution for a 256x256
input), bilinearly upsampled to (256, 256) and min-max normalized to
[0, 1], matching A in proposal §3.3 (A ∈ [0,1]^(256×256)).
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

from .hooks import AttentionExtractor
from .segformer_model import forest_prob


def _default_target(outputs) -> torch.Tensor:
    """Sum of predicted forest-class probability mass — a natural default
    scalar to explain: 'what pixels drove the model to say forest here?'"""
    return forest_prob(outputs.logits).sum()


def grad_rollout_attention_map(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    extractor: Optional[AttentionExtractor] = None,
    target_fn: Callable[[object], torch.Tensor] = _default_target,
    out_size: Tuple[int, int] = (256, 256),
    create_graph: bool = False,
) -> Tuple[torch.Tensor, object]:
    """
    Returns (attention_map, outputs).
    attention_map: (out_size) tensor in [0, 1], batch size must be 1.
    outputs: the model's SemanticSegmenterOutput for this forward pass
             (re-usable by the caller — no need to re-run the model).

    Two modes, controlled by `create_graph`:

    create_graph=False (default — inference / qualitative figures):
        Cheap path. Gradients are computed with a plain `.backward()` into
        `.grad`, then detached. Calls `model.zero_grad()` internally, so
        don't use this mid-way through accumulating gradients for something
        else.

    create_graph=True (training with the Attention Consistency Loss):
        The whole point of L_att is "attention map as a first-class
        training target" (proposal §3.1) — but A is itself *defined* via a
        gradient (Grad-Rollout). Backpropagating L_att(A, A*) into the
        model therefore requires a SECOND derivative through that first
        gradient, i.e. double backprop. This path computes the inner
        gradient with `torch.autograd.grad(..., create_graph=True)` instead
        of `.backward()`, keeping it attached to the autograd graph instead
        of writing to `.grad`, so the returned `attn_map` is differentiable
        w.r.t. model parameters and can be plugged into a combined loss
        that the caller backpropagates ONCE at the end. Caller is
        responsible for that final `.backward()` (and for calling
        `model.zero_grad()` beforehand) — this function does not touch
        `.grad` or call `.backward()` in this mode.
    """
    if pixel_values.shape[0] != 1:
        raise ValueError("grad_rollout_attention_map expects batch size 1.")

    own_extractor = extractor is None
    if own_extractor:
        extractor = AttentionExtractor(model, stage_index=4)

    outputs, stage_attentions = extractor.forward_with_attention(
        pixel_values, retain_grad=not create_graph
    )

    target = target_fn(outputs)

    if create_graph:
        grads = torch.autograd.grad(
            target, stage_attentions, create_graph=True, retain_graph=True
        )
    else:
        model.zero_grad(set_to_none=True)
        target.backward(retain_graph=False)
        grads = [attn.grad for attn in stage_attentions]
        if any(g is None for g in grads):
            raise RuntimeError(
                "Stage-4 attention has no gradient after backward() — "
                "target_fn likely doesn't depend on this stage's attention."
            )

    n_tokens = stage_attentions[0].shape[-1]  # 64 for a 256x256 input
    device = stage_attentions[0].device
    R = torch.eye(n_tokens, device=device)

    for attn, grad in zip(stage_attentions, grads):
        # Chefer et al.: positive part of grad * attn, averaged over heads.
        weighted = (grad * attn).clamp(min=0).mean(dim=1)[0]  # (N, N), batch 0
        # Rollout residual + row-normalize, then recursively combine.
        a_hat = weighted + torch.eye(n_tokens, device=device)
        a_hat = a_hat / a_hat.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        R = a_hat @ R

    relevance = R.mean(dim=0).clamp(min=0)  # (N,) row-mean: no CLS token to read off
    rmax = relevance.max()
    relevance = relevance / rmax if rmax > 0 else relevance

    grid = int(round(n_tokens**0.5))
    attn_map = relevance.reshape(1, 1, grid, grid)
    attn_map = F.interpolate(attn_map, size=out_size, mode="bilinear", align_corners=False)
    attn_map = attn_map[0, 0]
    if not create_graph:
        attn_map = attn_map.detach()

    return attn_map, outputs
