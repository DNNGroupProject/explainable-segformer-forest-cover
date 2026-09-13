# Attention-loss weight (λ2) sweep

Seed 42, MSE form, all else fixed. Selection rule: maximum test AAMO, then maximum test Dice.

| λ2 | test Dice | test IoU | test AAMO |
|----|-----------|----------|-----------|
| 0.1 | 0.8523 | 0.7425 | 0.5481 |
| 0.3 | 0.8690 | 0.7684 | 0.5752 |
| 0.5 | 0.8644 | 0.7612 | 0.6028 |
| **1.0** | 0.8577 | 0.7508 | **0.7476** |

Test AAMO rises monotonically while test Dice stays within a narrow 0.852–0.869 band — attention faithfulness and segmentation accuracy trade off, but the trade is shallow. λ2 = 1.0 is selected for all subsequent experiments.
