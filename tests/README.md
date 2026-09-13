# tests

Unit tests, all runnable without a dataset or GPU: pure logic against
synthetic tensors, plus one small fixture checkpoint.

```bash
python -m pytest tests/ -v
```

| File | Covers |
|------|--------|
| `test_attention_consistency_loss.py` | `AttentionConsistencyLoss` and `gaussian_soft_target` (MSE/KL forms, target smoothing). |
| `test_rollout_shapes.py` | Shape and gradient sanity checks for `AttentionExtractor` and `grad_rollout_attention_map`. |
| `test_boundary_refinement.py` | `morphological_gradient_boundary` and `BoundaryDiceLoss` on synthetic masks. |
| `test_aamo.py` | The AAMO metric and its Dice/IoU variants against hand-computed cases. |
| `test_augmentation.py` | The flip/rotation augmentation transforms preserve mask binarity. |
| `test_unet_model.py` | U-Net loader helpers, including the fixture checkpoint below. |
| `test_unet_torch.py` | The U-Net evaluation adapter, against the fixture checkpoint. |
| `test_mask_audit.py` | The dataset-audit binarization-delta logic on synthetic arrays. |
| `test_deeplab_split.py` | The DeepLabV3+ data split matches `dataset.py`'s split exactly. |
| `fixtures/unet_fixture_random.pt` | A tiny (< 1 MB), untrained U-Net checkpoint (deliberately narrow, so code that hardcodes the real baseline's width fails loudly instead of silently loading noise). Regenerate with `scripts/make_fixture_checkpoint.py`. |
