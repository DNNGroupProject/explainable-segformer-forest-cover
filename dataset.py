"""Dataset + split for the U-Net baseline, lifted out of
`unet_baseline_colab.ipynb` (Steps 1, 2 and 3).

Same pairing rule, same resize/normalize/binarize, same seeded split, so
anything importing this trains on exactly the data the committed baseline
trained on. The one addition is the `augment` flag, which is what makes the
Weeks 3-4 with/without ablation possible -- it defaults to False, so the
baseline path is unchanged.

Defaults to `data/{images,masks}` at the repo root (not included in the
repo, see README for how to obtain the dataset), matching the same choice
`scripts/mask_audit.py` and `attention_consistency/data.py` both make.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.augmentation import DEFAULT_CONFIG, AugmentConfig, augment_pair  # noqa: E402

DEFAULT_IMAGES_DIR = REPO_ROOT / "data" / "images"
DEFAULT_MASKS_DIR = REPO_ROOT / "data" / "masks"

# ImageNet stats, kept identical to the SegFormer baseline so both models see
# exactly the same inputs.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MASK_THRESHOLD = 127


def mask_name_to_image_name(mask_filename: str) -> str:
    """Converts '855_mask_01.jpg' -> '855_sat_01.jpg'."""
    return mask_filename.replace("_mask", "_sat")


class ForestSegDataset(Dataset):
    """One (image, mask) pair per mask filename.

    Returns (float32 CHW ImageNet-normalized image, int64 HW mask in {0,1}).

    augment: apply `shared.augmentation` to the training pairs. Leave it off
             for val/test -- the point is to vary what the model trains on,
             not what it is scored on.
    seed:    seeds this dataset's own augmentation RNG, separate from the
             split seed so changing one doesn't reshuffle the other.
    """

    def __init__(
        self,
        mask_filenames,
        images_dir=DEFAULT_IMAGES_DIR,
        masks_dir=DEFAULT_MASKS_DIR,
        img_size: int = 256,
        augment: bool = False,
        aug_config: AugmentConfig = DEFAULT_CONFIG,
        seed: int = 42,
    ):
        self.mask_filenames = list(mask_filenames)
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        self.augment = augment
        self.aug_config = aug_config
        # One generator for the whole dataset, so a run is reproducible from a
        # single seed. With num_workers > 0 each worker inherits a copy of it
        # and would replay the identical stream -- pass seed_worker as the
        # loader's worker_init_fn to give each one its own.
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, idx):
        mask_fname = self.mask_filenames[idx]
        image_fname = mask_name_to_image_name(mask_fname)

        image = Image.open(self.images_dir / image_fname).convert("RGB")
        image = image.resize((self.img_size, self.img_size))

        mask = Image.open(self.masks_dir / mask_fname).convert("L")
        mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)

        image = np.array(image, dtype=np.float32) / 255.0
        # White (>127) = forest = 1, black = non-forest = 0
        mask = (np.array(mask, dtype=np.int64) > MASK_THRESHOLD).astype(np.int64)

        # Augment here: pixels are in [0,1] and the mask is already binary,
        # which is what shared.augmentation expects. After the normalize below
        # a "brightness" shift would mean nothing.
        if self.augment:
            image, mask = augment_pair(image, mask, self.rng, self.aug_config)

        image = (image - self.mean) / self.std
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()

        return image, mask


def worker_rng(seed: int, worker_id: int) -> np.random.Generator:
    """The augmentation stream for one DataLoader worker.

    Split out from seed_worker so it can be tested without spawning a real
    loader. Seeding from the pair keeps the run reproducible while making each
    worker's stream independent.
    """
    return np.random.default_rng([seed, worker_id])


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: give each worker its own augmentation stream.

    Workers are forked after __init__, so without this they all hold a copy of
    the same generator and hand out the same sequence of augmentations -- the
    run still trains and still logs a curve, it just sees far less variety
    than it reports. Nothing downstream would catch that.
    """
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset.rng = worker_rng(info.dataset.seed, worker_id)


def set_seed(seed: int) -> None:
    """Seed every RNG the training path touches (the notebook's Step 3 helper)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def list_mask_files(masks_dir=DEFAULT_MASKS_DIR) -> list[str]:
    return sorted(os.listdir(Path(masks_dir)))


def make_splits(
    mask_files,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    subset: int | None = None,
):
    """The notebook's split, verbatim: shuffle once under `seed`, then take
    val / test / train off the front. Same seed 42 the whole team uses.

    subset: keep only the first N files after shuffling. Used to get a run
            that finishes on a CPU; None means the full dataset.
    """
    all_files = sorted(mask_files)
    random.seed(seed)
    random.shuffle(all_files)

    if subset is not None:
        all_files = all_files[:subset]

    n = len(all_files)
    n_val = int(n * val_split)
    n_test = int(n * test_split)

    val_files = all_files[:n_val]
    test_files = all_files[n_val : n_val + n_test]
    train_files = all_files[n_val + n_test :]
    return train_files, val_files, test_files
