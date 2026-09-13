# results

The final tables and qualitative figure referenced from the repository
root [`README.md`](../README.md).

- [`tables/`](tables/): main comparison, the λ2/λ3 loss-weight sweeps, the
  MSE-vs-KL attention-loss comparison, a three-seed robustness check
  (vanilla / Attention Consistency / Boundary Refinement / DeepLabV3+), and
  the weight-decay/LR-schedule overfitting-mitigation grid. Each table has
  a `.csv` (machine-readable) and a `.md` (with the accompanying reading of
  the numbers) version.
- [`figures/`](figures/): a qualitative comparison of SegFormer-B0's
  attention (Grad-Rollout) before and after adding the Attention
  Consistency Loss, on a held-out test image.

These are the final, aggregated numbers; the raw per-seed training logs and
checkpoints they were computed from are not included in this repository.
