"""Run the Boundary Loss λ3 sweep.

Fixes λ2 / att-mode at the λ2-sweep winner (`l2_1_mse`: λ2=1.0, MSE) and
sweeps λ3 for the Boundary Refinement Module. Mirrors run_lambda_sweep.py's
pattern.

Each cell writes under:
  checkpoints/runs/l2_1_mse_bnd<λ3>/
  results/runs/l2_1_mse_bnd<λ3>/
Skips a cell when that run's best checkpoint already exists (resume-safe).

Usage:
    python -m scripts.run_boundary_sweep
    python -m scripts.run_boundary_sweep --lambda3 0.1 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.config as paths
from scripts.config import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    N_TEST,
    N_TRAIN,
    N_VAL,
    SEED,
    SIGMA,
    apply_data_dirs,
)
import scripts.train_segformer as T

DEFAULT_LAMBDA3_SWEEP = (0.1, 0.2, 0.5)
WINNER_LAMBDA2 = 1.0  # λ2-sweep winner (l2_1_mse)
WINNER_ATT_MODE = "mse"


def boundary_tag(lambda2: float, att_mode: str, lambda3: float) -> str:
    """Must match train_full_scale.py's main() CLI naming exactly."""
    return f"{paths.run_tag(lambda2, att_mode)}_bnd{float(lambda3):g}"


def use_boundary_run_dirs(lambda2: float, att_mode: str, lambda3: float) -> str:
    tag = boundary_tag(lambda2, att_mode, lambda3)
    paths.set_output_dirs(
        paths.OUTPUT_ROOT_CKPT / "runs" / tag,
        paths.OUTPUT_ROOT_RESULTS / "runs" / tag,
    )
    return tag


def parse_args():
    p = argparse.ArgumentParser(description="Boundary Refinement Module λ3 sweep")
    p.add_argument("--lambda3", type=float, nargs="+", default=list(DEFAULT_LAMBDA3_SWEEP))
    p.add_argument("--lambda2", type=float, default=WINNER_LAMBDA2)
    p.add_argument("--att-mode", default=WINNER_ATT_MODE, choices=["mse", "kl"])
    p.add_argument("--boundary-kernel", type=int, default=3)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--sigma", type=float, default=SIGMA)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--n-train", type=int, default=N_TRAIN)
    p.add_argument("--n-val", type=int, default=N_VAL)
    p.add_argument("--n-test", type=int, default=N_TEST)
    p.add_argument("--img-dir", type=Path, default=None)
    p.add_argument("--mask-dir", type=Path, default=None)
    p.add_argument("--force", action="store_true", help="Retrain even if a best checkpoint already exists")
    return p.parse_args()


def main():
    args = parse_args()
    apply_data_dirs(args.img_dir or paths.DATA_IMG_DIR, args.mask_dir or paths.DATA_MASK_DIR)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(
        f"device={T.DEVICE}  λ2={args.lambda2}  mode={args.att_mode}  "
        f"λ3={args.lambda3}  boundary_kernel={args.boundary_kernel}  epochs={args.epochs}"
    )
    for lam3 in args.lambda3:
        tag = use_boundary_run_dirs(args.lambda2, args.att_mode, lam3)
        best = paths.CKPT_DIR / "segformer_b0_att_best.pt"
        if best.exists() and not args.force:
            print(f"SKIP train {tag}: {best} already exists")
            continue

        class SweepArgs:
            n_train, n_val, n_test = args.n_train, args.n_val, args.n_test
            epochs = args.epochs
            batch_size = args.batch_size
            lr = args.lr
            lambda2 = float(args.lambda2)
            sigma = args.sigma
            att_mode = args.att_mode
            seed = args.seed
            lambda3 = float(lam3)
            boundary_kernel = args.boundary_kernel

        print(f"\n>>> Training boundary sweep cell {tag}")
        T.train_variant("att", SweepArgs())

    print("\nBoundary sweep train pass done. Next: python eval_boundary_sweep.py")


if __name__ == "__main__":
    main()
