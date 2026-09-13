"""
DeepLabV3+ (MobileNetV3-Large) adapter for the extra baseline.

Usage:
    python evaluate.py --model deeplab
    python train_deeplab_extra.py          # creates checkpoints/deeplabv3_mobilenet_best.pt
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import config
from adapters.base import ModelAdapter
from adapters.data import load_pairs, split_dataset
from adapters.deeplab_model import build_deeplabv3, forest_prob_from_logits
from efficiency import count_torch_params, gflops_torch, measure_fps


class DeepLabV3Adapter(ModelAdapter):
    name = "DeepLabV3+ (MobileNetV3), extra baseline"

    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = str(config.DEEPLAB_CKPT)

    def load(self, checkpoint: Optional[str] = None) -> None:
        path = checkpoint or self.checkpoint
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"DeepLabV3+ checkpoint not found: {path}\n"
                "Train it first with scripts/train_deeplab_multiseed.py, or pass\n"
                "--checkpoint /path/to/deeplabv3_mobilenet_best.pt"
            )
        self.model = build_deeplabv3(num_classes=2, pretrained_backbone=False)
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.checkpoint = path

    def _to_tensor(self, images: np.ndarray) -> torch.Tensor:
        # images: (B,H,W,3) float 0..1 → (B,3,H,W) ImageNet-normalized
        x = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (x - mean) / std

    def predict_dataset(
        self,
        max_samples: Optional[int] = None,
        split: str = "test",
        batch_size: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if self.model is None:
            self.load()

        if max_samples is None:
            max_samples = int(os.environ.get("EVAL_MAX_SAMPLES", "1200"))
            if max_samples <= 0:
                max_samples = None

        images, masks = load_pairs(max_samples=max_samples)
        splits = split_dataset(images, masks)
        X, y = splits[split]

        probs_list = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i : i + batch_size]
                t = self._to_tensor(batch).to(self.device)
                out = self.model(t)["out"]
                # out may be lower-res; upsample to mask size
                if out.shape[-2:] != (config.IMG_SIZE, config.IMG_SIZE):
                    out = F.interpolate(
                        out,
                        size=(config.IMG_SIZE, config.IMG_SIZE),
                        mode="bilinear",
                        align_corners=False,
                    )
                probs_list.append(forest_prob_from_logits(out).cpu().numpy())

        probs = np.concatenate(probs_list, axis=0).astype(np.float32)
        return X, y, probs, None

    def count_params(self) -> Optional[int]:
        if self.model is None:
            self.load()
        return count_torch_params(self.model)

    def estimate_gflops(self) -> Optional[float]:
        if self.model is None:
            self.load()
        return gflops_torch(self.model, input_size=config.TORCH_FLOPS_INPUT, device=str(self.device))

    def measure_speed(self) -> Dict[str, float]:
        if self.model is None:
            self.load()
        dummy = torch.zeros(1, 3, config.IMG_SIZE, config.IMG_SIZE, device=self.device)

        def _run():
            with torch.no_grad():
                _ = self.model(dummy)

        _run()
        return measure_fps(_run, warmup=config.FPS_WARMUP, runs=config.FPS_RUNS)

    def supports_attention(self) -> bool:
        return False
