# Multi-seed ablation

Mean ± sample standard deviation (ddof=1) across seeds 42/43/44, all full-scale, same held-out test set.

| Config | Dice | IoU | AAMO |
|--------|------|-----|------|
| SegFormer-B0 (vanilla) | 0.8723 ± 0.0024 | 0.7735 ± 0.0038 | 0.0435 ± 0.0167 |
| + Attention Consistency (MSE) | 0.8564 ± 0.0026 | 0.7488 ± 0.0039 | 0.7028 ± 0.0392 |
| + Attention Consistency + Boundary Refinement (λ3=0.5) | 0.8505 ± 0.0139 | 0.7399 ± 0.0208 | 0.6881 ± 0.0719 |
| DeepLabV3+ (MobileNetV3) | 0.8404 ± 0.0278 | 0.7255 ± 0.0418 | n/a |

The vanilla and attention-MSE rows are stable across seeds (Dice spreads ≤ 0.0026); the boundary row is markedly noisier (5× the Dice std), driven by a seed-43 / boundary-loss interaction that also shifts its best-validation epoch much later than the other two seeds. Attention fidelity (AAMO) is noisy across all attention-loss configurations regardless of Dice stability; report it as mean ± std, not a bare number.
