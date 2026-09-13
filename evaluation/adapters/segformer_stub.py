"""
SegFormer-B0 adapter stub.

Person 2 owns training + attention extraction. When ready, replace NotImplemented
methods with a real PyTorch load/predict/attention_map implementation that matches
this interface so Person 4's evaluate.py and ablation_runner.py keep working.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

import config
from adapters.base import ModelAdapter


class SegFormerNotReadyError(RuntimeError):
    """Raised until Person 2 wires the real SegFormer checkpoint + attention hooks."""


_HELP = """
SegFormer evaluation is waiting on Person 2.

Expected deliverables from Person 2:
  1. Checkpoint at: {ckpt}
     (or pass --checkpoint /path/to/segformer.pt)
  2. A predict() that returns forest probability maps (H,W) in [0,1]
  3. Optional attention_map(image) -> (H,W) in [0,1] for AAMO

Wire them in adapters/segformer_stub.py (rename to segformer.py when ready).
""".strip()


class SegFormerStubAdapter(ModelAdapter):
    """Vanilla SegFormer-B0 (no attention loss) — stub until Person 2 delivers."""

    name = "SegFormer-B0 (no attention loss)"

    def __init__(self, variant: str = "vanilla"):
        self.variant = variant
        self.model = None
        if variant == "vanilla":
            self.checkpoint = str(config.SEGFORMER_CKPT)
            self.name = "SegFormer-B0 (no attention loss)"
        elif variant == "att":
            self.checkpoint = str(config.SEGFORMER_ATT_CKPT)
            self.name = "SegFormer-B0 + Attention Consistency Loss"
        elif variant == "boundary":
            self.checkpoint = str(config.SEGFORMER_BOUND_CKPT)
            self.name = "SegFormer-B0 + Attention Consistency + Boundary Loss"
        else:
            raise ValueError(f"Unknown SegFormer variant: {variant}")

    def load(self, checkpoint: Optional[str] = None) -> None:
        path = checkpoint or self.checkpoint
        raise SegFormerNotReadyError(_HELP.format(ckpt=path))

    def predict_dataset(
        self,
        max_samples: Optional[int] = None,
        split: str = "test",
        batch_size: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        self.load()
        raise SegFormerNotReadyError(_HELP.format(ckpt=self.checkpoint))

    def count_params(self) -> Optional[int]:
        return None

    def estimate_gflops(self) -> Optional[float]:
        return None

    def measure_speed(self) -> Dict[str, float]:
        return {"ms_per_image": float("nan"), "fps": float("nan")}

    def supports_attention(self) -> bool:
        # Will be True once Person 2 implements attention_map
        return False

    def attention_map(self, image: np.ndarray) -> np.ndarray:
        """
        Person 2: implement adapted Grad-Rollout → (H,W) attention in [0,1].
        """
        raise SegFormerNotReadyError(
            "attention_map() not implemented — Person 2 Grad-Rollout hook required."
        )
