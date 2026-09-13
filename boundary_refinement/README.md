# boundary_refinement

The optional Boundary Refinement Module: a boundary-aware Dice term on
morphological-gradient-extracted edges of the prediction and the mask, added
on top of the Attention Consistency Loss and weighted by its own λ.

| File | Contents |
|------|----------|
| `boundary_ops.py` | `morphological_gradient_boundary`: extracts a binary boundary band from a mask or soft prediction with a morphological gradient (dilation minus erosion) at a configurable kernel size. |
| `loss.py` | `BoundaryDiceLoss` (Dice computed only on the extracted boundary band) and `total_objective_with_boundary`, which combines it with the segmentation and attention losses. |

Enabled in training via `scripts/train_segformer.py --lambda3 <weight>`
(default `0.0`, which disables it and leaves the rest of the training loop
unchanged). Unit-tested on synthetic tensors in
`tests/test_boundary_refinement.py`, independent of any trained model.
