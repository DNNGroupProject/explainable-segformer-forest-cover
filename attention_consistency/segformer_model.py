"""
SegFormer-B0 model builder.

Uses HuggingFace
`SegformerForSemanticSegmentation` with the pretrained "nvidia/mit-b0"
encoder (ImageNet-pretrained MiT-B0 backbone) and a freshly-initialized
2-class decode head (background / forest).

MiT-B0 stage config (SegFormer paper, Table 6):
  hidden_sizes      = [32, 64, 160, 256]
  depths            = [2, 2, 2, 2]        (blocks per stage)
  num_attention_heads = [1, 2, 5, 8]
  sr_ratios         = [8, 4, 2, 1]        (spatial-reduction ratio on K/V per stage)

Stage 4 has sr_ratio=1, i.e. no spatial reduction, ordinary square
self-attention. That property is what attention_consistency/rollout.py
relies on (see its docstring and proposal §3.2).
"""
from __future__ import annotations

from transformers import SegformerConfig, SegformerForSemanticSegmentation

PRETRAINED_ENCODER = "nvidia/mit-b0"

MIT_B0_STAGE_CONFIG = dict(
    num_channels=3,
    num_encoder_blocks=4,
    depths=[2, 2, 2, 2],
    sr_ratios=[8, 4, 2, 1],
    hidden_sizes=[32, 64, 160, 256],
    num_attention_heads=[1, 2, 5, 8],
    patch_sizes=[7, 3, 3, 3],
    strides=[4, 2, 2, 2],
    mlp_ratios=[4, 4, 4, 4],
    decoder_hidden_size=256,
)


def build_segformer(
    num_labels: int = 2,
    pretrained: bool = True,
    output_attentions: bool = True,
) -> SegformerForSemanticSegmentation:
    """
    Build SegFormer-B0 for binary forest/non-forest segmentation.

    pretrained=True downloads the ImageNet-pretrained MiT-B0 encoder from the
    HF Hub (requires internet, ~15MB) and randomly initializes the decode
    head. pretrained=False builds
    the same architecture fully offline with random weights everywhere
    (useful for fast shape/unit tests).
    """
    # "sdpa" (the default attn backend) does not expose attention_probs at
    # all, which both the extraction hooks and Grad-Rollout need ("eager"
    # is required, not optional, for this pipeline.
    if pretrained:
        model = SegformerForSemanticSegmentation.from_pretrained(
            PRETRAINED_ENCODER,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            attn_implementation="eager",
        )
    else:
        cfg = SegformerConfig(
            num_labels=num_labels, attn_implementation="eager", **MIT_B0_STAGE_CONFIG
        )
        model = SegformerForSemanticSegmentation(cfg)

    model.config.output_attentions = output_attentions
    return model


def forest_prob(logits) -> "torch.Tensor":  # noqa: F821 - torch imported lazily by caller
    """
    logits: (B, num_labels, H, W) raw decode-head output (pre-upsample or
    post-upsample, either works since softmax is pointwise).
    Returns the forest-class probability map (B, H, W) in [0, 1].
    """
    import torch.nn.functional as F

    probs = F.softmax(logits, dim=1)
    return probs[:, 1]
