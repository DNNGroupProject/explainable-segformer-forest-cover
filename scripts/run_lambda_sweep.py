"""Run the Attention Consistency λ2 / att-mode sweep.

Trains the attention variant only. Each (λ2, att_mode) cell writes under:
  checkpoints/runs/l2_<λ>_<mode>/
  results/runs/l2_<λ>_<mode>/

Skips a cell when that run's best checkpoint already exists (resume-safe).

Usage:
    python -m scripts.run_lambda_sweep
    python -m scripts.run_lambda_sweep --lambda2 0.1 0.5 1.0 --att-mode mse
    python -m scripts.run_lambda_sweep --lambda2 0.3 --att-mode kl
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
    ATT_MODE,
    BATCH_SIZE,
    DEFAULT_LAMBDA2_SWEEP,
    EPOCHS,
    LR,
    N_TEST,
    N_TRAIN,
    N_VAL,
    SEED,
    SIGMA,
    apply_data_dirs,
    use_run_dirs,
)
import scripts.train_segformer as T


def parse_args():
    p = argparse.ArgumentParser(description="Attention-loss λ2 / att-mode sweep")
    p.add_argument(
        "--lambda2",
        type=float,
        nargs="+",
        default=list(DEFAULT_LAMBDA2_SWEEP),
        help="λ2 values to sweep (default: 0.1 0.3 0.5 1.0)",
    )
    p.add_argument("--att-mode", default=ATT_MODE, choices=["mse", "kl"])
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
    p.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if a best checkpoint for that cell already exists",
    )
    return p.parse_args()


def main():
    args = parse_args()
    apply_data_dirs(args.img_dir or paths.DATA_IMG_DIR, args.mask_dir or paths.DATA_MASK_DIR)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"device={T.DEVICE}  mode={args.att_mode}  λ2={args.lambda2}  epochs={args.epochs}")
    for lam in args.lambda2:
        tag = use_run_dirs(lam, args.att_mode)
        best = paths.CKPT_DIR / "segformer_b0_att_best.pt"
        if best.exists() and not args.force:
            print(f"SKIP train {tag}: {best} already exists")
            continue

        class SweepArgs:
            n_train, n_val, n_test = args.n_train, args.n_val, args.n_test
            epochs = args.epochs
            batch_size = args.batch_size
            lr = args.lr
            lambda2 = float(lam)
            sigma = args.sigma
            att_mode = args.att_mode
            seed = args.seed

        print(f"\n>>> Training sweep cell {tag}")
        T.train_variant("att", SweepArgs())

    print("\nSweep train pass done. Next: python eval_lambda_sweep.py")


if __name__ == "__main__":
    main()
