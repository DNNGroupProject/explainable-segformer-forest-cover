# scripts

Training entry points, loss-weight sweeps, and standalone diagnostics. All
scripts run as modules from the repository root (`python -m scripts.<name>`)
so their imports resolve against the top-level packages.

| File | Purpose |
|------|---------|
| `config.py` | Shared defaults (split sizes, epochs, batch size, seed) and output-path helpers used by the training/sweep scripts below. |
| `train_segformer.py` | Trains the vanilla SegFormer-B0 baseline and/or the Attention Consistency variant, with an opt-in `--lambda3` flag for the Boundary Refinement Module and `--weight-decay`/`--lr-schedule` for the overfitting-mitigation grid. |
| `run_lambda_sweep.py` | Sweeps the attention-loss weight λ2 (and MSE-vs-KL mode), one training run per cell. |
| `run_boundary_sweep.py` | Sweeps the boundary-loss weight λ3 at the λ2 winner. |
| `train_unet_augmentation_ablation.py` | Trains the U-Net baseline twice (with and without augmentation), same split/seed/starting weights, to isolate the effect of augmentation. |
| `train_deeplab_multiseed.py` | Trains the DeepLabV3+ (MobileNetV3) baseline across seeds 42/43/44 and evaluates each on the shared held-out test set. |
| `deeplab_data.py` | Dataset pairing/split helpers for the DeepLabV3+ training script (mirrors `dataset.py`'s split exactly). |
| `mask_audit.py` | Standalone dataset-integrity audit (pairing, geometry, value spread, class balance); exits non-zero on a hard failure so it can gate a training run. |
| `analyze_rollout_stages.py` | Diagnostic: quantifies how much mask-relevant attention signal is discarded by restricting Grad-Rollout to encoder stage 4. |
| `make_fixture_checkpoint.py` | Generates the small (< 1 MB) untrained U-Net checkpoint in `tests/fixtures/`, used to test the loader code without the real trained weights. |

See the repository root [`README.md`](../README.md) for concrete usage
examples of each script.
