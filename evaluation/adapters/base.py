"""Model adapter interface for Person 4 evaluation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BatchPrediction:
    """One evaluation batch."""
    images: np.ndarray          # (B,H,W,3) float 0..1 or uint8
    masks: np.ndarray           # (B,H,W) binary
    probs: np.ndarray           # (B,H,W) soft predictions
    attentions: Optional[np.ndarray] = None  # (B,H,W) optional attention maps


class ModelAdapter(ABC):
    name: str = "base"

    @abstractmethod
    def load(self, checkpoint: Optional[str] = None) -> None:
        ...

    @abstractmethod
    def predict_dataset(
        self,
        max_samples: Optional[int] = None,
        split: str = "test",
        batch_size: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Returns images, masks, probs, attentions (attentions may be None).
        All arrays are stacked over the full split.
        """
        ...

    @abstractmethod
    def count_params(self) -> Optional[int]:
        ...

    @abstractmethod
    def estimate_gflops(self) -> Optional[float]:
        ...

    @abstractmethod
    def measure_speed(self) -> Dict[str, float]:
        """Return ms_per_image, fps."""
        ...

    def supports_attention(self) -> bool:
        return False
