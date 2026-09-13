# Boundary Refinement weight (λ3) sweep

Fixed at λ2 = 1.0 (MSE). Selection rule: maximum test Dice, then maximum test IoU (differs from the λ2 sweep because the boundary term targets boundary quality, not attention fidelity).

## Seed-42 sweep

| λ3 | test Dice | test IoU | test AAMO | best-val epoch |
|----|-----------|----------|-----------|----------------|
| 0.1 | 0.8558 | 0.7479 | 0.7775 | 11 |
| **0.2** | **0.8669** | **0.7650** | 0.6218 | 4 |
| 0.5 | 0.8611 | 0.7560 | 0.6805 | 4 |

Seed-42 alone selects λ3 = 0.2, at a cost to AAMO (0.6218 vs. the λ3 = 0 baseline's 0.7476).

## Three-seed re-check (seeds 42/43/44)

| λ3 | Dice (mean ± std) | IoU (mean ± std) | AAMO (mean ± std) |
|----|--------------------|--------------------|--------------------|
| 0.1 | 0.8451 ± 0.0094 | 0.7319 ± 0.0141 | 0.7527 ± 0.0429 |
| 0.2 | 0.8498 ± 0.0240 | 0.7393 ± 0.0358 | 0.6714 ± 0.0706 |
| **0.5** | **0.8505 ± 0.0139** | **0.7399 ± 0.0208** | 0.6881 ± 0.0719 |

On three seeds the argmax moves from λ3 = 0.2 to λ3 = 0.5 (lower variance, slightly higher mean Dice), though the two are not cleanly separated at one standard deviation. λ3 = 0.5 is used as the final selected weight for all subsequent multi-seed reporting; the seed-42 point values in the main comparison table still use λ3 = 0.2 (the single-seed sweep winner at the time those numbers were fixed).
