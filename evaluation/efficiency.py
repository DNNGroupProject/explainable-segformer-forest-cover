"""
Efficiency metrics: parameter count, FLOPs/GFLOPs, inference FPS / ms/image.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import time


def count_keras_params(model) -> int:
    return int(model.count_params())


def count_torch_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def gflops_keras(model, input_shape=(1, 256, 256, 3)) -> Optional[float]:
    """
    Try tensorflow profiler / keras-flops style estimate.
    Returns GFLOPs or None if unavailable.
    """
    try:
        # Prefer keras-flops if installed
        from keras_flops import get_flops  # type: ignore

        flops = get_flops(model, batch_size=1)
        return float(flops) / 1e9
    except Exception:
        pass

    try:
        import tensorflow as tf

        # Rough fallback: sum MultiplyAdd ops via profile if available
        concrete = tf.function(lambda x: model(x))
        # Can't always get exact FLOPs without TF profiler setup — report None
        _ = concrete
        return None
    except Exception:
        return None


def gflops_torch(model, input_size=(1, 3, 256, 256), device="cpu") -> Optional[float]:
    try:
        from thop import profile  # type: ignore
        import torch

        model = model.to(device)
        model.eval()
        dummy = torch.randn(*input_size, device=device)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return float(flops) / 1e9
    except Exception:
        return None


def measure_fps(
    predict_fn: Callable[[], Any],
    warmup: int = 3,
    runs: int = 20,
) -> Dict[str, float]:
    """
    predict_fn: zero-arg callable that runs one forward pass on a fixed batch/image.
    Returns ms_per_image and fps (assumes batch size 1 inside predict_fn).
    """
    for _ in range(warmup):
        predict_fn()

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        predict_fn()
        times.append(time.perf_counter() - t0)

    mean_s = float(np.mean(times))
    ms = mean_s * 1000.0
    fps = 1.0 / mean_s if mean_s > 0 else 0.0
    return {"ms_per_image": ms, "fps": fps}


def efficiency_row(
    params: Optional[int] = None,
    gflops: Optional[float] = None,
    fps: Optional[float] = None,
    ms_per_image: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "params": params if params is not None else "n/a",
        "gflops": round(gflops, 3) if isinstance(gflops, (int, float)) else "n/a",
        "fps": round(fps, 2) if isinstance(fps, (int, float)) else "n/a",
        "ms_per_image": round(ms_per_image, 2)
        if isinstance(ms_per_image, (int, float))
        else "n/a",
    }
