"""
Attention-extraction hooks for SegFormer-B0 (Person 2 — Transformer Lead task).

SegFormer-B0's encoder (`nvidia/mit-b0`) has 4 stages with depths [2,2,2,2] —
8 transformer blocks total, each holding a `SegformerAttention` submodule at
`segformer.stages.{stage_idx}.blocks.{block_idx}.attention`. When the model
is run with `output_attentions=True` and `attn_implementation="eager"`, each
`SegformerAttention.forward` returns `(context, attention_probs)`.

Rather than relying on that tuple only reaching us via the top-level
`ModelOutput.attentions` (which forces recomputing/discarding the whole
16-tuple every call), this module registers `torch.nn.Module` forward hooks
directly on the stage-4 attention submodules (the ones with sr_ratio=1 —
see rollout.py for why stage 4 is the one that matters) and captures the
attention-probability tensors as they're produced, retaining their
gradients in place so Gradient-weighted Attention Rollout (rollout.py) can
read `.grad` after a backward pass.

Verified attention_probs shape for a 256x256 input, MiT-B0 stage 4:
(batch, num_heads=8, N=64, N=64) — square, since sr_ratio=1 there.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class AttentionExtractor:
    """Captures stage-4 (sr_ratio=1) SegFormer attention maps via forward hooks."""

    def __init__(self, model: nn.Module, stage_index: int = 4):
        """
        stage_index: 1-indexed SegFormer encoder stage (1..4). Defaults to
        the last stage, the only one with sr_ratio=1 (see rollout.py).
        """
        depths = model.config.depths
        if not (1 <= stage_index <= len(depths)):
            raise ValueError(f"stage_index must be in [1, {len(depths)}]")
        self.model = model
        self.stage_index = stage_index
        self.n_blocks = depths[stage_index - 1]

        self._modules: List[nn.Module] = []
        for block_idx in range(self.n_blocks):
            name = f"segformer.stages.{stage_index - 1}.blocks.{block_idx}.attention"
            module = dict(model.named_modules()).get(name)
            if module is None:
                raise RuntimeError(
                    f"Could not find attention submodule '{name}' — "
                    "SegFormer internals may have changed in this transformers version."
                )
            self._modules.append(module)

        self._captured: List[Optional[torch.Tensor]] = [None] * self.n_blocks
        self._handles = []

    def _make_hook(self, slot: int):
        def _hook(module, inputs, output):
            # output = (context, attention_probs) when output_attentions=True
            attn_probs = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            if attn_probs is None:
                raise RuntimeError(
                    "Attention hook fired without attention_probs — "
                    "call the model with output_attentions=True."
                )
            self._captured[slot] = attn_probs

        return _hook

    def __enter__(self) -> "AttentionExtractor":
        self._handles = [
            mod.register_forward_hook(self._make_hook(i))
            for i, mod in enumerate(self._modules)
        ]
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def forward_with_attention(self, pixel_values: torch.Tensor, retain_grad: bool = False):
        """
        Runs the model once and returns (outputs, stage_attentions), where
        stage_attentions is a list of length `n_blocks` of tensors shaped
        (B, heads, N, N) captured via the forward hooks.
        """
        self._captured = [None] * self.n_blocks
        with self:
            outputs = self.model(pixel_values, output_attentions=True)

        stage_attentions = self._captured
        if any(a is None for a in stage_attentions):
            raise RuntimeError("Not all stage attention hooks fired during forward().")
        if retain_grad:
            for a in stage_attentions:
                a.retain_grad()
        return outputs, stage_attentions
