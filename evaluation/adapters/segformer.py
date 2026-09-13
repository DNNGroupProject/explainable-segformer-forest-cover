"""
Real SegFormer-B0 adapter for the evaluation CLI.

Delegates model-building, attention extraction, and Grad-Rollout to the
`attention_consistency` package (repo root) rather than duplicating it, so
training and evaluation stay in sync with one implementation.

Loads whichever checkpoint was placed in this folder's checkpoints/ dir;
see config.SEGFORMER_CKPT / SEGFORMER_ATT_CKPT for the expected filenames.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from attention_consistency import build_segformer, grad_rollout_attention_map  # noqa: E402
from attention_consistency.data import load_pairs, make_splits, to_model_input  # noqa: E402
from attention_consistency.segformer_model import forest_prob  # noqa: E402

import config
from adapters.base import ModelAdapter
from efficiency import count_torch_params, gflops_torch, measure_fps

# Same split the smoke-scale training script trained with; must match so
# "test" here is the checkpoint's actual held-out set.
_SPLIT_SEED = 42
_N_TRAIN, _N_VAL, _N_TEST = 400, 60, 60

_LABELS = {
    "vanilla": "SegFormer-B0 (no attention loss)",
    "att": "SegFormer-B0 + Attention Consistency Loss",
}
_CKPT_ATTR = {
    "vanilla": "SEGFORMER_CKPT",
    "att": "SEGFORMER_ATT_CKPT",
}


class SegFormerAdapter(ModelAdapter):
    """SegFormer-B0, vanilla or +Attention Consistency Loss, with real Grad-Rollout attention."""

    def __init__(self, variant: str = "vanilla"):
        if variant not in _LABELS:
            raise ValueError(f"Unknown SegFormer variant: {variant} (use 'vanilla' or 'att')")
        self.variant = variant
        self.name = _LABELS[variant]
        self.model = None
        self.checkpoint = str(getattr(config, _CKPT_ATTR[variant]))

    def load(self, checkpoint: Optional[str] = None) -> None:
        path = checkpoint or self.checkpoint
        if not Path(path).exists():
            raise FileNotFoundError(
                f"SegFormer checkpoint not found: {path}\n"
                "Train with scripts/train_segformer.py first, then copy the "
                "checkpoint here (or pass --checkpoint)."
            )
        self.model = build_segformer(pretrained=False)
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.checkpoint = path

    def predict_dataset(
        self,
        max_samples: Optional[int] = None,
        split: str = "test",
        batch_size: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if self.model is None:
            self.load()

        splits = make_splits(_N_TRAIN, _N_VAL, _N_TEST, seed=_SPLIT_SEED)
        pairs = splits[split]
        if max_samples is not None:
            pairs = pairs[:max_samples]
        images, masks = load_pairs(pairs)

        probs_list, attn_list = [], []
        for i in range(len(images)):
            x = to_model_input(images[i : i + 1])
            with torch.enable_grad():
                attn_map, outputs = grad_rollout_attention_map(self.model, x)
            logits_up = F.interpolate(
                outputs.logits, size=(256, 256), mode="bilinear", align_corners=False
            )
            probs_list.append(forest_prob(logits_up)[0].detach().numpy())
            attn_list.append(attn_map.numpy())

        return images, masks, np.stack(probs_list), np.stack(attn_list)

    def count_params(self) -> Optional[int]:
        if self.model is None:
            self.load()
        return count_torch_params(self.model)

    def estimate_gflops(self) -> Optional[float]:
        if self.model is None:
            self.load()
        return gflops_torch(self.model, input_size=(1, 3, 256, 256))

    def measure_speed(self) -> Dict[str, float]:
        if self.model is None:
            self.load()
        dummy = to_model_input(np.zeros((1, 256, 256, 3), dtype=np.float32))

        def _run():
            with torch.no_grad():
                self.model(pixel_values=dummy)

        return measure_fps(_run, warmup=3, runs=10)

    def supports_attention(self) -> bool:
        return True

    def attention_map(self, image: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.load()
        x = to_model_input(image[None].astype(np.float32))
        with torch.enable_grad():
            attn_map, _ = grad_rollout_attention_map(self.model, x)
        return attn_map.numpy()
