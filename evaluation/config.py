"""Paths and defaults for the evaluation harness."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent  # repo root

# Dataset not included in the repo — see README for how to obtain it and
# where to place it.
DATA_DIR = PROJECT / "data"

IMG_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "masks"

# CNN baseline: PyTorch U-Net (paper row). Not included — train with
# scripts/train_unet_augmentation_ablation.py or provide your own checkpoint.
UNET_CKPT = PROJECT / "checkpoints" / "unet_baseline_best.pt"
# Superseded Keras checkpoint (evaluate.py --model unet_keras only)
UNET_KERAS_CKPT = PROJECT / "checkpoints" / "unet_baseline_best.keras"

# SegFormer checkpoints (copied into this folder)
SEGFORMER_CKPT = ROOT / "checkpoints" / "segformer_b0_vanilla.pt"
SEGFORMER_ATT_CKPT = ROOT / "checkpoints" / "segformer_b0_att_consistency.pt"
SEGFORMER_BOUND_CKPT = ROOT / "checkpoints" / "segformer_b0_att_boundary.pt"

# Extra baseline: DeepLabV3+ MobileNetV3-Large
DEEPLAB_CKPT = ROOT / "checkpoints" / "deeplabv3_mobilenet_best.pt"

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "checkpoints").mkdir(parents=True, exist_ok=True)

SEED = 42
IMG_SIZE = 256
MASK_THRESHOLD = 127
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
DEFAULT_THRESHOLD = 0.5
AAMO_THRESHOLD = 0.5

# Efficiency probe input
FLOPS_INPUT_SHAPE = (1, IMG_SIZE, IMG_SIZE, 3)  # Keras NHWC
TORCH_FLOPS_INPUT = (1, 3, IMG_SIZE, IMG_SIZE)  # PyTorch NCHW
FPS_WARMUP = 3
FPS_RUNS = 20
