"""Shared defaults and output-path helpers for the training/sweep scripts.

Consolidates what used to be several near-identical per-contributor
`paths.py` files (each supporting a multi-person Colab layout) into one
flat-repo version. All full-scale numbers (split sizes, epochs, batch size,
seed) match the values reported in the paper.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

N_TRAIN, N_VAL, N_TEST = 3576, 766, 766
SEED = 42
EPOCHS = 20
BATCH_SIZE = 16
LR = 6e-5
LAMBDA2 = 1.0
SIGMA = 8.0
ATT_MODE = "mse"

DEFAULT_LAMBDA2_SWEEP = (0.1, 0.3, 0.5, 1.0)
DEFAULT_LAMBDA3_SWEEP = (0.1, 0.2, 0.5)

DATA_DIR = REPO_ROOT / "data"
DATA_IMG_DIR = DATA_DIR / "images"
DATA_MASK_DIR = DATA_DIR / "masks"

OUTPUT_ROOT_CKPT = REPO_ROOT / "checkpoints"
OUTPUT_ROOT_RESULTS = REPO_ROOT / "results" / "runs"
CKPT_DIR = OUTPUT_ROOT_CKPT
RESULTS_DIR = REPO_ROOT / "results"


def run_tag(lambda2: float, att_mode: str) -> str:
    """Folder name for one sweep cell, e.g. l2_0.3_mse."""
    return f"l2_{float(lambda2):g}_{att_mode}"


def run_ckpt_dir(lambda2: float, att_mode: str) -> Path:
    return OUTPUT_ROOT_CKPT / "runs" / run_tag(lambda2, att_mode)


def run_results_dir(lambda2: float, att_mode: str) -> Path:
    return OUTPUT_ROOT_RESULTS / run_tag(lambda2, att_mode)


def ensure_output_dirs() -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_output_dirs(ckpt_dir: Path | None = None, results_dir: Path | None = None) -> None:
    """Point active write dirs at one sweep cell (or a Drive folder on Colab)."""
    global CKPT_DIR, RESULTS_DIR
    if ckpt_dir is not None:
        CKPT_DIR = Path(ckpt_dir)
    if results_dir is not None:
        RESULTS_DIR = Path(results_dir)
    ensure_output_dirs()


def use_run_dirs(lambda2: float, att_mode: str) -> str:
    """Point CKPT_DIR / RESULTS_DIR at one cell under the current roots."""
    tag = run_tag(lambda2, att_mode)
    set_output_dirs(run_ckpt_dir(lambda2, att_mode), run_results_dir(lambda2, att_mode))
    return tag


def apply_data_dirs(img_dir: Path | None = None, mask_dir: Path | None = None) -> None:
    """Override attention_consistency.data's dataset root."""
    global DATA_IMG_DIR, DATA_MASK_DIR, DATA_DIR
    DATA_IMG_DIR = Path(img_dir) if img_dir is not None else DATA_IMG_DIR
    DATA_MASK_DIR = Path(mask_dir) if mask_dir is not None else DATA_MASK_DIR
    DATA_DIR = DATA_IMG_DIR.parent
    import attention_consistency.data as data_mod

    data_mod.IMG_DIR = DATA_IMG_DIR
    data_mod.MASK_DIR = DATA_MASK_DIR
