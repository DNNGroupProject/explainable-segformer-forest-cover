"""
Segmentation metrics for Person 4 (Evaluation Lead).

Accumulate TP/FP/FN/TN over the FULL test set, then compute ratios once.
(Do not average per-batch IoUs — that biases small batches.)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np


@dataclass
class ConfusionCounts:
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    tn: float = 0.0

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        """pred/target: binary arrays of any shape."""
        p = pred.astype(np.float64).ravel()
        t = target.astype(np.float64).ravel()
        self.tp += float(np.sum((p == 1) & (t == 1)))
        self.fp += float(np.sum((p == 1) & (t == 0)))
        self.fn += float(np.sum((p == 0) & (t == 1)))
        self.tn += float(np.sum((p == 0) & (t == 0)))

    def merge(self, other: "ConfusionCounts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tn += other.tn


def metrics_from_counts(c: ConfusionCounts, eps: float = 1e-6) -> Dict[str, float]:
    tp, fp, fn, tn = c.tp, c.fp, c.fn, c.tn
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    pixel_acc = (tp + tn) / (tp + fp + fn + tn + eps)
    return {
        "pixel_acc": float(pixel_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "dice": float(dice),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def binarize(prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (prob > threshold).astype(np.float32)


def compute_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    probs: soft predictions in [0,1]
    targets: binary {0,1}
    """
    preds = binarize(probs, threshold)
    c = ConfusionCounts()
    c.update(preds, targets)
    return metrics_from_counts(c)


def format_metrics(m: Dict[str, float], digits: int = 4) -> str:
    keys = ["dice", "iou", "f1", "precision", "recall", "pixel_acc"]
    parts = [f"{k}={m[k]:.{digits}f}" for k in keys if k in m]
    return " | ".join(parts)


def delta_iou_pct(iou_method: float, iou_baseline: float) -> float:
    if iou_baseline == 0:
        return 0.0
    return 100.0 * (iou_method - iou_baseline) / iou_baseline
