"""
DeepLabV3+ (MobileNetV3-Large) — optional extra baseline for Person 4.

Proposal stretch: "Extra baseline (DeepLabV3+ or Swin-Unet-Tiny), if time allows."
We use torchvision DeepLabV3-MobileNetV3-Large (ASPP decoder ≈ DeepLabV3+ family)
because it is lightweight and available without custom Swin weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


def build_deeplabv3(num_classes: int = 2, pretrained_backbone: bool = True) -> nn.Module:
    """
    Binary forest/non-forest: num_classes=2 (background, forest).
    Forest probability = softmax(logits)[:, 1].
    """
    backbone_weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
    model = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=backbone_weights)
    # Replace classification head for binary segmentation
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        # FCN head last conv — keep defensive
        try:
            model.aux_classifier[4] = nn.Conv2d(
                model.aux_classifier[4].in_channels, num_classes, kernel_size=1
            )
        except Exception:
            model.aux_classifier = None
    return model


def forest_prob_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """logits (B,2,H,W) → forest prob (B,H,W)."""
    return torch.softmax(logits, dim=1)[:, 1]
