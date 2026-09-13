"""
PyTorch U-Net adapter: official CNN baseline.

Loads `unet_model.py` (repo root) via load_unet(), ImageNet-normalizes
inputs the same way the dataset loader does, and scores the held-out test
split (seed 42, 3576/766/766).

Usage:
    python evaluate.py --model unet
    python evaluate.py --model unet_torch
    python evaluate.py --model unet_keras   # legacy Keras path
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

import config
from adapters.base import ModelAdapter
from adapters.deeplab_model import forest_prob_from_logits
from efficiency import count_torch_params, gflops_torch, measure_fps

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from unet_model import load_unet  # noqa: E402
from dataset import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    MASK_THRESHOLD,
    list_mask_files,
    make_splits,
    mask_name_to_image_name,
)

# thop measurement at 1x3x256x256 (used if thop missing)
BASELINE_GFLOPS = 109.48


class UnetTorchAdapter(ModelAdapter):
    name = "U-Net (CNN baseline)"

    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = str(config.UNET_CKPT)

    def load(self, checkpoint: Optional[str] = None) -> None:
        path = checkpoint or self.checkpoint
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"PyTorch U-Net checkpoint not found: {path}\n"
                "Expected a checkpoint at checkpoints/unet_baseline_best.pt "
                "(see scripts/train_unet_augmentation_ablation.py)"
            )
        self.model = load_unet(path, device=str(self.device))
        self.checkpoint = path

    def _to_tensor(self, images: np.ndarray) -> torch.Tensor:
        """(B,H,W,3) float in [0,1] → NCHW ImageNet-normalized.

        Flagged gotcha: this U-Net was trained from scratch, but dataset.py
        still applies ImageNet mean/std so inputs match SegFormer.
        Do NOT skip this (plain /255 would silently wreck Dice/IoU).
        """
        x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float()
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return (x - mean) / std

    def _load_split_arrays(
        self,
        max_samples: Optional[int],
        split: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Shuffle under seed 42, then 15% val / 15% test / rest train."""
        mask_files = list_mask_files()
        train_f, val_f, test_f = make_splits(
            mask_files, val_split=0.15, test_split=0.15, seed=config.SEED, subset=max_samples
        )
        files = {"train": train_f, "val": val_f, "test": test_f}[split]
        from dataset import DEFAULT_IMAGES_DIR, DEFAULT_MASKS_DIR

        images_dir = Path(DEFAULT_IMAGES_DIR)
        masks_dir = Path(DEFAULT_MASKS_DIR)

        imgs, masks = [], []
        size = config.IMG_SIZE
        for mask_fname in files:
            image_fname = mask_name_to_image_name(mask_fname)
            image = Image.open(images_dir / image_fname).convert("RGB")
            image = image.resize((size, size))
            mask = Image.open(masks_dir / mask_fname).convert("L")
            mask = mask.resize((size, size), resample=Image.NEAREST)
            imgs.append(np.array(image, dtype=np.float32) / 255.0)
            masks.append((np.array(mask, dtype=np.int64) > MASK_THRESHOLD).astype(np.float32))
        if not imgs:
            raise FileNotFoundError(f"No pairs loaded for split={split} from {images_dir}")
        return np.stack(imgs), np.stack(masks)

    def predict_dataset(
        self,
        max_samples: Optional[int] = None,
        split: str = "test",
        batch_size: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if self.model is None:
            self.load()

        if max_samples is None:
            env = os.environ.get("EVAL_MAX_SAMPLES")
            if env is not None:
                max_samples = int(env)
                if max_samples <= 0:
                    max_samples = None

        X, y = self._load_split_arrays(max_samples, split)
        probs_list = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                t = self._to_tensor(X[i : i + batch_size]).to(self.device)
                out = self.model(t)
                if out.shape[-2:] != (config.IMG_SIZE, config.IMG_SIZE):
                    import torch.nn.functional as F

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
        measured = gflops_torch(
            self.model, input_size=config.TORCH_FLOPS_INPUT, device=str(self.device)
        )
        if measured is None:
            return BASELINE_GFLOPS
        return measured

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
