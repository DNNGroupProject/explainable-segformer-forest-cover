"""Quantify the cost of restricting Grad-Rollout to SegFormer's stage 4
(Threats-to-Validity item 4, paper_acm/main.tex).

rollout.py restricts Gradient-weighted Attention Rollout to encoder stage 4
because it's the only stage where sr_ratio=1 makes the recursive rollout
matrix product mathematically valid (see that file's docstring) -- stages
1-3 have rectangular attention matrices the recursion can't chain through.
That leaves an open question never quantified: how much does stage 4 alone
actually discard from stages 1-3?

Every SegFormer-B0 stage's attention_probs tensor has shape
(heads, N_query, N_kv) where N_kv is always 64 (an 8x8 grid) regardless of
stage -- spatial reduction (sr_ratio 8/4/2/1 for stages 1-4) always reduces
keys/values down to the same 8x8 grid. That's a genuine architectural
coincidence that makes a cross-stage comparison tractable without needing
the (invalid) full rollout recursion: average each stage's raw
attention_probs over (heads, query) to get one 8x8 "received attention"
grid per stage, upsample to 256x256, and compare.

Two comparisons, per stage s in {1,2,3,4}:
  1. corr(stage_s_raw_grid, stage4_grad_rollout_map) -- how much does raw
     stage-s attention already look like the actual (gradient-weighted)
     map the model trains against?
  2. corr(stage_s_raw_grid, ground_truth_mask) -- how much forest-relevant
     signal is present at that stage BEFORE any gradient weighting?

This is a diagnostic only -- no model/training code is changed, no claim
is made that stages 1-3 could be used instead (their rectangular attention
still can't be rolled out the same way).

    python -m scripts.analyze_rollout_stages --checkpoint path/to/segformer_b0_att_best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.config as config  # noqa: E402

config.apply_data_dirs()

from attention_consistency import build_segformer, grad_rollout_attention_map  # noqa: E402
from attention_consistency.data import list_pairs, load_pairs, to_model_input  # noqa: E402
from attention_consistency.hooks import AttentionExtractor  # noqa: E402

DEVICE = torch.device("cpu")


def load_model(checkpoint: Path) -> torch.nn.Module:
    model = build_segformer(pretrained=False)
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    return model


def stage_grid(model: torch.nn.Module, x: torch.Tensor, stage: int) -> np.ndarray:
    """Average raw attention_probs over (heads, query) -> 8x8 grid -> flat 64-vec."""
    extractor = AttentionExtractor(model, stage_index=stage)
    with torch.no_grad():
        _, stage_attentions = extractor.forward_with_attention(x, retain_grad=False)
    # Average this stage's blocks, then over (heads, query) dims.
    avg = torch.stack([a.mean(dim=(0, 1, 2)) for a in stage_attentions]).mean(dim=0)  # (64,)
    grid = avg.reshape(8, 8).numpy()
    return grid


def upsample_and_norm(grid: np.ndarray, size: int = 256) -> np.ndarray:
    t = torch.tensor(grid, dtype=torch.float32).reshape(1, 1, 8, 8)
    up = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    up = up[0, 0].numpy()
    rng = up.max() - up.min()
    return (up - up.min()) / rng if rng > 0 else up


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    import argparse
    import time

    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to a trained segformer_b0_att_best.pt (attention-consistency variant, from train_segformer.py)",
    )
    p.add_argument("--n", type=int, default=3, help="number of held-out images to sample")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    model = load_model(args.checkpoint)
    pairs = list_pairs(max_samples=args.n * 4, seed=args.seed)[-args.n :]
    images, masks = load_pairs(pairs)

    t0 = time.time()
    per_image_rows = []
    for i in range(len(images)):
        x = to_model_input(images[i : i + 1]).to(DEVICE)
        y = masks[i]

        with torch.enable_grad():
            rollout_map, _ = grad_rollout_attention_map(model, x)
        rollout_map = rollout_map.detach().numpy()

        row = {"image": i + 1}
        for stage in (1, 2, 3, 4):
            grid = stage_grid(model, x, stage)
            up = upsample_and_norm(grid)
            row[f"stage{stage}_vs_rollout"] = corr(up, rollout_map)
            row[f"stage{stage}_vs_mask"] = corr(up, y.astype(np.float32))
        per_image_rows.append(row)
        if args.n <= 10 or (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"image {i+1}/{args.n} ({elapsed:.1f}s elapsed): " +
                  ", ".join(f"{k}={v:.3f}" for k, v in row.items() if k != "image"))

    print(f"\nMean across {args.n} images ({time.time()-t0:.1f}s total):")
    for stage in (1, 2, 3, 4):
        vals_r = [r[f"stage{stage}_vs_rollout"] for r in per_image_rows]
        vals_m = [r[f"stage{stage}_vs_mask"] for r in per_image_rows]
        vr, sr = np.nanmean(vals_r), np.nanstd(vals_r)
        vm, sm = np.nanmean(vals_m), np.nanstd(vals_m)
        print(f"  stage {stage}: vs_stage4_rollout={vr:.3f}+/-{sr:.3f}  vs_ground_truth_mask={vm:.3f}+/-{sm:.3f}")


if __name__ == "__main__":
    main()
