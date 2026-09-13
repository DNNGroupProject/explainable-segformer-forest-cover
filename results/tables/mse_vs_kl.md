# MSE vs. KL for the Attention Consistency Loss

λ2 = 1.0, same split, batch size 1. MSE seed 42 is the main-comparison-table row; the other rows are additional runs.

| Form | Seed | test Dice | test IoU | test AAMO | best-val Dice |
|------|------|-----------|----------|-----------|---------------|
| MSE | 42 | 0.8577 | 0.7508 | 0.7476 | ~0.78 |
| MSE | 43 | 0.8534 | 0.7443 | 0.6865 | 0.7801 |
| MSE | 44 | 0.8580 | 0.7513 | 0.6744 | 0.7831 |
| KL | 42 | **0.7462** | **0.5951** | **0.3412** | 0.7851 |

Replacing MSE with KL divergence costs 11 Dice points and more than halves AAMO at an identical λ2, seed, and split. MSE's seed-to-seed Dice spread (42/43/44) is only 0.0046, so the MSE–KL gap is far outside seed noise. KL's best-validation Dice is close to MSE's, but its test Dice collapses: the KL objective fits the validation attention distribution without a corresponding gain in mask-aligned attention. MSE is used as the default form everywhere else in this project.
