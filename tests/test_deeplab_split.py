"""Verify the DeepLab split matches dataset.py's U-Net split (3576/766/766)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deeplab_data import MASK_DIR, N_TEST, N_TRAIN, N_VAL, list_mask_files, make_splits_files  # noqa: E402


class TestDeeplabSplitIdentity(unittest.TestCase):
    def test_sizes(self):
        if not MASK_DIR.is_dir():
            self.skipTest(f"dataset missing: {MASK_DIR}")
        splits = make_splits_files(seed=42)
        self.assertEqual(len(splits["train"]), N_TRAIN)
        self.assertEqual(len(splits["val"]), N_VAL)
        self.assertEqual(len(splits["test"]), N_TEST)

    def test_matches_unet_make_splits(self):
        if not MASK_DIR.is_dir():
            self.skipTest(f"dataset missing: {MASK_DIR}")
        import dataset as unet_dataset  # repo-root dataset.py

        masks = list_mask_files()
        ours = make_splits_files(masks, seed=42)
        train_c, val_c, test_c = unet_dataset.make_splits(
            masks, val_split=0.15, test_split=0.15, seed=42
        )
        self.assertEqual(ours["train"], train_c)
        self.assertEqual(ours["val"], val_c)
        self.assertEqual(ours["test"], test_c)
        self.assertEqual(len(test_c), 766)


if __name__ == "__main__":
    unittest.main()
