"""
Explainability-guided SegFormer-B0 pipeline (Person 3 — Loss & Training).

Also carries the two pieces this package depends on end-to-end so the whole
chain is runnable from one place:
  - segformer_model.py : Person 2's model (SegFormer-B0, pretrained MiT-B0 encoder)
  - hooks.py            : Person 2's attention-extraction hooks
  - rollout.py          : Person 2's adapted Gradient-weighted Attention Rollout
  - loss.py             : Person 3's Attention Consistency Loss (this role's core deliverable)
"""
from .segformer_model import build_segformer
from .hooks import AttentionExtractor
from .rollout import grad_rollout_attention_map
from .loss import AttentionConsistencyLoss, gaussian_soft_target

__all__ = [
    "build_segformer",
    "AttentionExtractor",
    "grad_rollout_attention_map",
    "AttentionConsistencyLoss",
    "gaussian_soft_target",
]
