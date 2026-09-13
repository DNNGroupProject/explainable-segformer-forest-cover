# Main comparison

Full 5,108-image dataset (3576/766/766 split, seed 42, 20 epochs). All rows share the same 766-image held-out test set.

| Model | Dice | IoU | F1 | AAMO | Params |
|-------|------|-----|----|------|--------|
| U-Net (CNN) | 0.8615 | 0.7568 | 0.8615 | n/a | 31.0M |
| DeepLabV3+ (MobileNetV3) | 0.8716 | 0.7724 | 0.8716 | n/a | 11.0M |
| SegFormer-B0 (vanilla) | 0.8743 | 0.7766 | 0.8743 | 0.0334 | 3.7M |
| + Attention Consistency Loss (MSE) | 0.8577 | 0.7508 | 0.8577 | 0.7476 | 3.7M |
| + Attention Consistency Loss (KL) | 0.7462 | 0.5951 | 0.7462 | 0.3412 | 3.7M |
| + Attention Consistency + Boundary Refinement | 0.8669 | 0.7650 | 0.8669 | 0.6218 | 3.7M |

See `main_comparison.csv` for the same data in machine-readable form.
