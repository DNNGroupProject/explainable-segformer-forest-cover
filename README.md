# Explainability-Guided SegFormer for Forest Cover Segmentation

Code and results for a study that treats Vision Transformer attention as a
supervised training target rather than a post-hoc visualization. An
**Attention Consistency Loss** aligns SegFormer-B0's internal attention with
the ground-truth forest mask during training, and **AAMO** (Average
Attention–Mask Overlap) reports attention fidelity quantitatively alongside
Dice and IoU. A **Boundary Refinement Module** adds a boundary-aware Dice
term for canopy-edge accuracy.

On the full 5,108-image dataset, the Attention Consistency Loss raises AAMO
from 0.033 to 0.748 (~22x) over a vanilla SegFormer-B0 baseline, at a modest
segmentation cost (Dice 0.874 → 0.858). See [`results/`](results/) for the
full set of tables (loss-weight sweeps, an MSE-vs-KL comparison of the
attention loss, a multi-seed robustness check, and an overfitting-mitigation
grid) and a qualitative attention-map comparison.

## Method

- **Model.** SegFormer-B0 (MiT-B0 encoder + all-MLP decoder), with an
  explainability-supervision path used only during training; at inference
  the model runs as a standard SegFormer-B0.
- **Attention extraction.** SegFormer's Efficient Self-Attention uses a
  spatial-reduction ratio that makes attention rectangular (not square) at
  three of its four encoder stages, which breaks the recursive matrix
  product that Attention Rollout needs. Gradient-weighted Attention Rollout
  is adapted to run only on the one stage (`sr_ratio=1`) where the
  attention matrix is square and token-consistent — see
  `attention_consistency/rollout.py`.
- **Attention Consistency Loss.** The extracted attention map is compared
  against a Gaussian-smoothed ground-truth mask (MSE or KL form) and
  optimized jointly with the segmentation loss. Because the attention map is
  itself defined through a gradient, this requires double backpropagation —
  see `attention_consistency/loss.py` and `scripts/train_segformer.py`.
- **AAMO.** A quantitative attention-fidelity metric (`evaluation/aamo.py`):
  the overlap between thresholded attention and the ground-truth mask,
  reportable the way Dice or IoU are.
- **Boundary Refinement Module.** An optional boundary Dice term on
  morphological-gradient-extracted boundaries of the prediction and the mask
  (`boundary_refinement/`), added on top of the attention loss and weighted
  by a separate λ.

## Repository layout

```
attention_consistency/   SegFormer-B0 + Attention Consistency Loss + Grad-Rollout
boundary_refinement/     Boundary Refinement Module (morphological boundary Dice)
evaluation/              Model-agnostic evaluation harness (AAMO, Dice/IoU/F1, efficiency,
                         adapters for SegFormer / U-Net / DeepLabV3+)
shared/                  Augmentation and experiment-tracking utilities used by
                         more than one baseline
scripts/                 Training entry points, loss-weight sweeps, and diagnostics
tests/                   Unit tests (pure logic + synthetic tensors; no dataset needed)
results/                 Final result tables and the qualitative attention-map figure
unet_model.py            U-Net baseline (PyTorch)
dataset.py               Dataset loading + split for the U-Net baseline
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and (for full-scale training) a CUDA GPU; training
also runs on CPU at reduced batch size / sample count for smoke-testing.

### Dataset

Experiments use a 5,108-image RGB aerial forest/non-forest dataset at
256x256 resolution (the "Forest Segmented" dataset), split 70/15/15
(3576/766/766) under a fixed seed. The dataset is not distributed with this
repository — place it under:

```
data/
├── images/   *_sat_*.jpg
└── masks/    *_mask_*.jpg
```

Then verify integrity before training:

```bash
python -m scripts.mask_audit
```

This checks pairing, image/mask geometry, value spread, and class balance,
and exits non-zero on a hard integrity failure, so it can act as a
pre-flight gate in a training pipeline.

## Usage

Train the vanilla baseline and the Attention Consistency variant:

```bash
python -m scripts.train_segformer --variant both
```

Add the Boundary Refinement Module (λ3 > 0 enables it):

```bash
python -m scripts.train_segformer --variant att --lambda3 0.5
```

Reproduce the loss-weight sweeps:

```bash
python -m scripts.run_lambda_sweep
python -m scripts.run_boundary_sweep
```

Train the U-Net baseline (with the augmentation ablation used in
`results/tables/`):

```bash
python -m scripts.train_unet_augmentation_ablation --epochs 20 \
    --features 64,128,256,512 --subset 0 --batch-size 8 --device cuda --amp
```

Train the DeepLabV3+ (MobileNetV3) baseline, seeds 42/43/44:

```bash
DEEPLAB_MAX_SAMPLES=100000 DEEPLAB_EPOCHS=20 DEEPLAB_BATCH=8 \
    python -m scripts.train_deeplab_multiseed
```

Evaluate any trained checkpoint (Dice/IoU/F1, AAMO, parameter count,
GFLOPs, FPS) with the shared harness:

```bash
python evaluation/evaluate.py --model segformer-att
```

Quantify how much attention signal is discarded by restricting Grad-Rollout
to the final encoder stage:

```bash
python -m scripts.analyze_rollout_stages --checkpoint path/to/segformer_b0_att_best.pt
```

### Tests

```bash
python -m pytest tests/ -v
```

All tests run on synthetic tensors or the small (< 1 MB) fixture checkpoint
in `tests/fixtures/` — no dataset or GPU required.

## Results

| Model | Dice | IoU | AAMO | Params |
|-------|------|-----|------|--------|
| U-Net (CNN) | 0.8615 | 0.7568 | n/a | 31.0M |
| DeepLabV3+ (MobileNetV3) | 0.8716 | 0.7724 | n/a | 11.0M |
| SegFormer-B0 (vanilla) | 0.8743 | 0.7766 | 0.0334 | 3.7M |
| + Attention Consistency Loss (MSE) | 0.8577 | 0.7508 | 0.7476 | 3.7M |
| + Attention Consistency + Boundary Refinement | 0.8669 | 0.7650 | 0.6218 | 3.7M |

Full tables — the λ2/λ3 sweeps, the MSE-vs-KL comparison, a three-seed
robustness check, and the overfitting-mitigation grid — are in
[`results/tables/`](results/tables/). A qualitative comparison of attention
maps before and after the Attention Consistency Loss is in
[`results/figures/`](results/figures/).

## Authors

Dhinanjaya Fernando, Dinura Ginige, Kalana Lakshan, Chanupa Gurusinghe,
Lasana Pahanga, Dileesha Manamperi, Thimira Sahan, Sandareka Wickramanayake

Dept. of Computer Science & Engineering, University of Moratuwa, Sri Lanka

## License

Code is released under the [MIT License](LICENSE).
