"""Legacy Keras U-Net adapter for the evaluation harness."""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

import config
from adapters.base import ModelAdapter
from adapters.data import load_pairs, split_dataset
from efficiency import count_keras_params, gflops_keras, measure_fps


class UnetKerasAdapter(ModelAdapter):
    name = "U-Net (Keras, superseded)"

    def __init__(self):
        self.model = None
        self.checkpoint = str(config.UNET_KERAS_CKPT)

    def load(self, checkpoint: Optional[str] = None) -> None:
        import tensorflow as tf

        path = checkpoint or self.checkpoint
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"U-Net checkpoint not found: {path}\n"
                "Old Keras path. Prefer --model unet (PyTorch). Pass --checkpoint to override."
            )
        # Custom objects may be needed if model used named losses; load with compile=False
        self.model = tf.keras.models.load_model(path, compile=False)
        self.checkpoint = path

    def predict_dataset(
        self,
        max_samples: Optional[int] = None,
        split: str = "test",
        batch_size: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if self.model is None:
            self.load()

        # Match training: if the checkpoint was trained with max_samples=1200,
        # use the same cap unless user overrides.
        if max_samples is None:
            # Heuristic: use 1200 when env not set (the legacy CPU default)
            max_samples = int(os.environ.get("EVAL_MAX_SAMPLES", "1200"))
            if max_samples <= 0:
                max_samples = None

        images, masks = load_pairs(max_samples=max_samples)
        splits = split_dataset(images, masks)
        X, y = splits[split]

        probs = self.model.predict(X, batch_size=batch_size, verbose=0)
        if probs.ndim == 4 and probs.shape[-1] == 1:
            probs = probs[..., 0]
        return X, y, probs.astype(np.float32), None

    def count_params(self) -> Optional[int]:
        if self.model is None:
            self.load()
        return count_keras_params(self.model)

    def estimate_gflops(self) -> Optional[float]:
        if self.model is None:
            self.load()
        return gflops_keras(self.model, input_shape=config.FLOPS_INPUT_SHAPE)

    def measure_speed(self) -> Dict[str, float]:
        if self.model is None:
            self.load()
        import tensorflow as tf

        dummy = np.zeros((1, config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.float32)

        def _run():
            _ = self.model(dummy, training=False)

        # Warm Keras/TF once
        _run()
        return measure_fps(_run, warmup=config.FPS_WARMUP, runs=config.FPS_RUNS)

    def supports_attention(self) -> bool:
        return False
