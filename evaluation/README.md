# evaluation

A model-agnostic evaluation harness: one CLI and one set of metrics shared
by every baseline (SegFormer variants, U-Net, DeepLabV3+), so numbers are
comparable across models by construction.

| File | Contents |
|------|----------|
| `evaluate.py` | CLI entry point. `python evaluate.py --model <name>` loads a checkpoint through the matching adapter, runs it over the held-out test split, and reports Dice/IoU/F1, AAMO (where the model has attention), parameter count, GFLOPs, and FPS. |
| `config.py` | Paths and defaults shared by the harness: dataset location, per-model checkpoint paths, split ratios, thresholds, and the efficiency-probe input shape. |
| `metrics.py` | `ConfusionCounts` and the Dice/IoU/F1/precision/recall/pixel-accuracy computation shared by every model. |
| `aamo.py` | The Average Attention-Mask Overlap metric: `compute_aamo`, its Dice/IoU symmetric variants, and `mean_aamo` for averaging across a test set. |
| `efficiency.py` | Parameter counting, GFLOPs estimation, and FPS measurement for both PyTorch and (legacy) Keras models. |
| `adapters/` | One adapter per model family behind a common interface. See [`adapters/README.md`](adapters/README.md). |

Run `python evaluate.py --model <name>` from this directory (or point it at
a checkpoint elsewhere with `--checkpoint`); see the repository root
[`README.md`](../README.md) for a full example.
