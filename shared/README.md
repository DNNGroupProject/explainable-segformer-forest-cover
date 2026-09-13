# shared

Small utilities used by more than one baseline, kept in one place so every
model trains on the same transforms and logs runs the same way.

| File | Contents |
|------|----------|
| `augmentation.py` | Torch-friendly train-time augmentation for image/mask pairs (flips and 90-degree rotations only, numpy-based, no interpolation, so masks stay exactly binary). Used by `scripts/train_unet_augmentation_ablation.py`; every other training script in this repo trains without augmentation by default. |
| `experiment_tracking.py` | A thin, optional Weights & Biases wrapper that no-ops when `wandb` isn't installed or enabled, so training scripts can log metrics without a hard dependency. |

Unit-tested on synthetic image/mask pairs in `tests/test_augmentation.py`.
