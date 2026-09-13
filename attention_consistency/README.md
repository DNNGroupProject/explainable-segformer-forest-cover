# attention_consistency

SegFormer-B0 with the Attention Consistency Loss: the model builder, the
adapted Gradient-weighted Attention Rollout, and the loss itself.

| File | Contents |
|------|----------|
| `segformer_model.py` | Builds SegFormer-B0 (pretrained MiT-B0 encoder, binary decode head) and `forest_prob`, the forest-class probability from the decoder logits. |
| `hooks.py` | `AttentionExtractor`: forward hooks that pull raw `attention_probs` out of the encoder's stage-4 self-attention blocks. |
| `rollout.py` | Adapts Gradient-weighted Attention Rollout to SegFormer's spatial-reduction attention, restricted to encoder stage 4 (the only stage where the attention matrix is square and token-consistent). `grad_rollout_attention_map` is the entry point used during training. |
| `loss.py` | `AttentionConsistencyLoss` (MSE or KL form) and `gaussian_soft_target`, which turns the ground-truth mask into the soft attention target the rollout map is compared against. |
| `data.py` | Dataset loading and the fixed 3576/766/766 split used across every baseline in this repo. |

Requires a trained-model context (this package is imported by
`scripts/train_segformer.py`); the shape/gradient behavior is unit-tested
independently of any checkpoint in `tests/test_rollout_shapes.py` and
`tests/test_attention_consistency_loss.py`.
