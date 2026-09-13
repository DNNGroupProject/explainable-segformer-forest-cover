"""
Shape/gradient sanity checks for Person 2's attention hooks + adapted
Grad-Rollout, using a from-scratch (randomly initialized, no internet
needed) SegFormer-B0 so this runs fast and offline.

Run:
    python tests/test_rollout_shapes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attention_consistency import build_segformer, grad_rollout_attention_map
from attention_consistency.hooks import AttentionExtractor


def test_stage4_attention_is_square_and_matches_expected_token_count():
    model = build_segformer(pretrained=False)
    model.eval()
    extractor = AttentionExtractor(model, stage_index=4)
    x = torch.randn(1, 3, 256, 256)
    outputs, stage_attentions = extractor.forward_with_attention(x)

    assert len(stage_attentions) == model.config.depths[3]  # 2 blocks
    for a in stage_attentions:
        assert a.shape[-1] == a.shape[-2] == 64, a.shape  # square, 8x8 grid
    assert outputs.logits.shape[0] == 1


def test_hooks_capture_gradients_when_requested():
    model = build_segformer(pretrained=False)
    model.train()
    extractor = AttentionExtractor(model, stage_index=4)
    x = torch.randn(1, 3, 256, 256)
    outputs, stage_attentions = extractor.forward_with_attention(x, retain_grad=True)
    outputs.logits.sum().backward()
    for a in stage_attentions:
        assert a.grad is not None
        assert a.grad.shape == a.shape


def test_grad_rollout_attention_map_shape_and_range():
    model = build_segformer(pretrained=False)
    model.train()
    x = torch.randn(1, 3, 256, 256)
    attn_map, outputs = grad_rollout_attention_map(model, x, out_size=(256, 256))
    assert attn_map.shape == (256, 256)
    assert torch.all(attn_map >= 0) and torch.all(attn_map <= 1.0 + 1e-6)
    assert outputs.logits.shape == (1, 2, 64, 64)


def test_grad_rollout_rejects_batch_greater_than_one():
    model = build_segformer(pretrained=False)
    model.train()
    x = torch.randn(2, 3, 256, 256)
    try:
        grad_rollout_attention_map(model, x)
        assert False, "expected ValueError for batch size > 1"
    except ValueError:
        pass


def test_stage_index_out_of_range_raises():
    model = build_segformer(pretrained=False)
    try:
        AttentionExtractor(model, stage_index=5)
        assert False, "expected ValueError for stage_index=5"
    except ValueError:
        pass


ALL_TESTS = [
    test_stage4_attention_is_square_and_matches_expected_token_count,
    test_hooks_capture_gradients_when_requested,
    test_grad_rollout_attention_map_shape_and_range,
    test_grad_rollout_rejects_batch_greater_than_one,
    test_stage_index_out_of_range_raises,
]

if __name__ == "__main__":
    failed = 0
    for t in ALL_TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - failed}/{len(ALL_TESTS)} passed")
    if failed:
        raise SystemExit(1)
