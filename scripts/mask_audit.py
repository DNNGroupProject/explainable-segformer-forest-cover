"""Dataset integrity audit for the Forest Segmented image/mask pairs.

Written because the repo has no data-validation utility at all: every loader
just assumes the masks are clean, and the one place that documents mask
preprocessing justified thresholding with claims about
the data ("JPEG masks are not perfectly 0/255", "soft interpolation creates
gray in-between pixels") that nobody had actually measured.

This script measures them. It reports:
  1. pairing        -- every image has a mask and vice versa
  2. geometry       -- the distinct image/mask sizes present
  3. value spread   -- where mask pixel values actually sit in 0..255
  4. binarisation   -- what `mask / 255.0` costs versus `mask > 127`
  5. class balance  -- forest pixel ratio, for anyone weighing the loss

Exits non-zero if a hard integrity check fails (orphan pairs, unreadable
files, or sizes that disagree with --expect-size), so it can be used as a
pre-flight gate before a training run.

Run:
    python mask_audit.py
    python mask_audit.py --sample-every 10        # faster spot check
    python mask_audit.py --images ... --masks ...

Needs numpy + Pillow only. Deliberately no cv2: most loaders in this repo
use it, but it isn't installed everywhere the team works, and Pillow reads
these JPEGs identically for our purposes.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Same threshold used by the evaluation harness (evaluation/config.py)
MASK_THRESHOLD = 127

# A pixel this far from 0 or 255 is genuinely ambiguous rather than just
# rounding noise. JPEG ringing on a hard black/white edge lands within a few
# levels of the extremes; real soft-interpolation artefacts land mid-range.
MIDGRAY_LO, MIDGRAY_HI = 32, 223

# How far a soft target has to sit from its thresholded counterpart before it
# would plausibly change what the network learns.
DELTA_TOL = 0.1

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMG_DIR = REPO_ROOT / "data" / "images"
DEFAULT_MASK_DIR = REPO_ROOT / "data" / "masks"


def mask_name_for(image_filename: str) -> str:
    """'10452_sat_08.jpg' -> '10452_mask_08.jpg' (dataset naming convention)."""
    stem, ext = os.path.splitext(image_filename)
    return (stem.replace("_sat_", "_mask_") if "_sat_" in stem else stem) + ext


def find_mask_path(mask_dir: Path, image_filename: str) -> Path | None:
    """Locate an image's mask, trying the sibling extensions the other
    loaders in this repo try (the dataset is all .jpg, but
    attention_consistency/data.py allows fallbacks, so stay compatible)."""
    stem, ext = os.path.splitext(mask_name_for(image_filename))
    for candidate_ext in (ext, *(e for e in IMAGE_EXTS if e != ext)):
        candidate = mask_dir / f"{stem}{candidate_ext}"
        if candidate.exists():
            return candidate
    return None


def summarise_histogram(hist: np.ndarray) -> dict:
    """Turn a 256-bin pixel-value histogram into the percentages we care about."""
    total = int(hist.sum())
    if total == 0:
        raise ValueError("empty histogram -- no mask pixels were read")
    pct = lambda n: 100.0 * float(n) / total
    present = np.nonzero(hist)[0]
    return {
        "pixels_total": total,
        "pct_exactly_0": pct(hist[0]),
        "pct_exactly_255": pct(hist[255]),
        # Split on the threshold itself, so each bucket maps to exactly one
        # label: `mask > 127` sends 127 to background, so it belongs in the
        # lower bucket even though it looks mid-range.
        "pct_1_to_threshold": pct(hist[1 : MASK_THRESHOLD + 1].sum()),
        "pct_threshold_to_254": pct(hist[MASK_THRESHOLD + 1 : 255].sum()),
        "pct_midgray": pct(hist[MIDGRAY_LO : MIDGRAY_HI + 1].sum()),
        "distinct_values": len(present),
        # How far toward mid-gray any pixel strays: 0 means every pixel is
        # exactly 0 or 255, 127 means something sits on the threshold itself.
        "max_drift_toward_middle": int(np.minimum(present, 255 - present).max()),
    }


def binarisation_delta(mask: np.ndarray) -> tuple[float, float]:
    """Compare the two ways this repo turns a mask into a target.

    Returns (mean absolute difference, fraction of pixels differing by more
    than DELTA_TOL) between `mask / 255.0` and `mask > MASK_THRESHOLD`.
    A near-zero result means the missing threshold is harmless *on this data*;
    it is not a general guarantee.
    """
    soft = mask.astype(np.float32) / 255.0
    hard = (mask > MASK_THRESHOLD).astype(np.float32)
    diff = np.abs(soft - hard)
    return float(diff.mean()), float((diff > DELTA_TOL).mean())


def audit(img_dir: Path, mask_dir: Path, sample_every: int, expect_size: int | None) -> dict:
    """Read the dataset once and return every number the report needs.

    Flow: list the files -> check pairing -> read each sampled mask and
    accumulate stats -> check the sizes -> pack it all into one dict.
    """
    # --- 1. What's on disk -------------------------------------------------
    image_files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMAGE_EXTS))
    mask_files = sorted(f for f in os.listdir(mask_dir) if f.lower().endswith(IMAGE_EXTS))
    if not image_files:
        raise FileNotFoundError(f"No images found under {img_dir}")

    # --- 2. Pairing: an unmatched file in either direction is a real problem,
    #        because a silently skipped pair shrinks the training set.
    orphan_images = [f for f in image_files if find_mask_path(mask_dir, f) is None]
    paired_masks = {find_mask_path(mask_dir, f).name for f in image_files if f not in orphan_images}
    orphan_masks = [f for f in mask_files if f not in paired_masks]

    # --- 3. Walk the pairs and accumulate ----------------------------------
    # One pass over the data. The histogram is summed across every mask so the
    # percentages describe the whole dataset, while the per-mask numbers
    # (delta, forest ratio) are collected in lists and averaged at the end.
    sampled = image_files[::sample_every]
    hist = np.zeros(256, dtype=np.int64)
    img_sizes: set[tuple[int, int]] = set()
    mask_sizes: set[tuple[int, int]] = set()
    unreadable: list[str] = []
    deltas: list[float] = []
    over_tol: list[float] = []
    forest_ratios: list[float] = []

    for filename in sampled:
        mask_path = find_mask_path(mask_dir, filename)
        if mask_path is None:
            continue
        try:
            # Sizes come from the headers -- only the mask is decoded to pixels,
            # which keeps a full 5,108-pair run to about a minute.
            with Image.open(img_dir / filename) as im:
                img_sizes.add(im.size)
            with Image.open(mask_path) as mk:
                mask_sizes.add(mk.size)
                mask = np.array(mk.convert("L"))
        except OSError as exc:
            # Note it and keep going -- one corrupt file shouldn't kill the run.
            # Either file in the pair could be the bad one, so name both.
            unreadable.append(f"{filename} + {mask_path.name}: {exc}")
            continue

        hist += np.bincount(mask.ravel(), minlength=256)
        mean_delta, frac_over = binarisation_delta(mask)
        deltas.append(mean_delta)
        over_tol.append(frac_over)
        forest_ratios.append(float((mask > MASK_THRESHOLD).mean()))

    if not deltas:
        raise FileNotFoundError(f"No readable image/mask pairs under {img_dir} / {mask_dir}")

    # --- 4. Geometry: if every file already matches the training IMG_SIZE,
    #        the resize in the loaders does nothing and can't distort labels.
    unexpected_sizes = (
        sorted(s for s in img_sizes | mask_sizes if s != (expect_size, expect_size))
        if expect_size
        else []
    )

    # --- 5. Pack up. "ok" drives the exit code, so only genuine integrity
    #        failures belong in it -- not the measurements themselves.
    result = {
        "images_found": len(image_files),
        "masks_found": len(mask_files),
        "masks_sampled": len(deltas),
        "orphan_images": orphan_images,
        "orphan_masks": orphan_masks,
        "unreadable": unreadable,
        "image_sizes": sorted(img_sizes),
        "mask_sizes": sorted(mask_sizes),
        "unexpected_sizes": unexpected_sizes,
        "soft_vs_hard_mean_delta": float(np.mean(deltas)),
        "soft_vs_hard_pct_over_tol": 100.0 * float(np.mean(over_tol)),
        "forest_ratio_mean": float(np.mean(forest_ratios)),
        "forest_ratio_std": float(np.std(forest_ratios)),
    }
    result.update(summarise_histogram(hist))
    result["ok"] = not (orphan_images or orphan_masks or unreadable or unexpected_sizes)
    return result


def build_rows(r: dict) -> list[tuple[str, str, str]]:
    """(check, value, note) rows -- the same table gets written as .csv and .md."""
    sizes = ", ".join(f"{w}x{h}" for w, h in r["mask_sizes"])
    return [
        ("Image files", f"{r['images_found']}", "Under the audited images/ folder"),
        ("Mask files", f"{r['masks_found']}", "Under the audited masks/ folder"),
        ("Masks sampled", f"{r['masks_sampled']}", "Pairs actually read for this report"),
        ("Orphan images", f"{len(r['orphan_images'])}", "Images with no matching mask"),
        ("Orphan masks", f"{len(r['orphan_masks'])}", "Masks with no matching image"),
        ("Unreadable files", f"{len(r['unreadable'])}", "Failed to decode"),
        ("Distinct mask sizes", sizes, "Resize is a no-op when this equals the configured IMG_SIZE"),
        ("Mask pixels sampled", f"{r['pixels_total']:,}", "Basis for the percentages below"),
        (
            "Distinct mask values",
            f"{r['distinct_values']} of 256",
            "How many grey levels actually occur",
        ),
        (
            "Furthest drift toward mid-gray",
            f"{r['max_drift_toward_middle']} levels",
            "Largest distance any pixel sits from pure 0 or pure 255",
        ),
        ("Pixels exactly 0", f"{r['pct_exactly_0']:.4f}%", "Pure non-forest"),
        ("Pixels exactly 255", f"{r['pct_exactly_255']:.4f}%", "Pure forest"),
        # Ranges are built from MASK_THRESHOLD rather than hardcoded, so the
        # labels stay honest if the threshold is ever changed.
        (
            f"Pixels 1-{MASK_THRESHOLD}",
            f"{r['pct_1_to_threshold']:.4f}%",
            "Off-black, thresholds to 0",
        ),
        (
            f"Pixels {MASK_THRESHOLD + 1}-254",
            f"{r['pct_threshold_to_254']:.4f}%",
            "Off-white, thresholds to 1",
        ),
        (
            f"Pixels {MIDGRAY_LO}-{MIDGRAY_HI} (mid-gray)",
            f"{r['pct_midgray']:.4f}%",
            "Genuinely ambiguous pixels -- the ones thresholding is supposed to rescue",
        ),
        (
            "mask/255 vs mask>127, mean abs diff",
            f"{r['soft_vs_hard_mean_delta']:.6f}",
            "What skipping the threshold actually costs per pixel",
        ),
        (
            f"...pixels differing by >{DELTA_TOL}",
            f"{r['soft_vs_hard_pct_over_tol']:.4f}%",
            "Pixels where the two targets meaningfully disagree",
        ),
        (
            "Forest pixel ratio",
            f"{r['forest_ratio_mean']:.4f} +/- {r['forest_ratio_std']:.4f}",
            "Mean +/- std per mask, after thresholding",
        ),
    ]


def write_reports(r: dict, out_dir: Path) -> tuple[Path, Path]:
    """Write the same table twice: .csv to compute with, .md to read in a PR.

    Both come from build_rows(), so the two files can never drift apart.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(r)

    csv_path = out_dir / "mask_audit.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["check", "value", "note"])
        writer.writerows(rows)

    md_path = out_dir / "mask_audit.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Mask audit (Person 1)\n\n")
        f.write(
            "Generated by `mask_audit.py`. Measures the Forest Segmented masks "
            "instead of assuming their properties.\n\n"
        )
        f.write("| Check | Value | Note |\n|---|---|---|\n")
        for check, value, note in rows:
            f.write(f"| {check} | {value} | {note} |\n")
        f.write(f"\nIntegrity checks: **{'PASS' if r['ok'] else 'FAIL'}**\n")
    return csv_path, md_path


def main(argv: list[str] | None = None) -> int:
    """Read args -> check the folders exist -> audit -> write -> print.

    Exit codes: 0 all checks passed, 1 an integrity check failed,
    2 a folder was missing. That's what makes it usable as a pre-flight
    gate -- `python mask_audit.py && python train.py` stops on bad data.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", type=Path, default=DEFAULT_IMG_DIR)
    parser.add_argument("--masks", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument(
        "--sample-every", type=int, default=1, help="Audit every Nth pair (default: all)"
    )
    parser.add_argument(
        "--expect-size", type=int, default=256, help="Flag any file not this square size; 0 disables"
    )
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args(argv)

    # Fail early and clearly on a bad path -- otherwise the error surfaces
    # much later as a confusing "no pairs found".
    for d in (args.images, args.masks):
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            return 2

    r = audit(args.images, args.masks, max(1, args.sample_every), args.expect_size or None)
    csv_path, md_path = write_reports(r, args.out_dir)

    # Print the same rows to the terminal, padded into two aligned columns.
    width = max(len(c) for c, _, _ in build_rows(r))
    for check, value, _ in build_rows(r):
        print(f"{check:<{width}}  {value}")

    # Only list problems that actually occurred, and cap it at 5 names so a
    # broken dataset doesn't scroll thousands of lines past you.
    for label, items in (
        ("orphan images", r["orphan_images"]),
        ("orphan masks", r["orphan_masks"]),
        ("unreadable files", r["unreadable"]),
        ("unexpected sizes", r["unexpected_sizes"]),
    ):
        if items:
            print(f"\n{len(items)} {label}: {items[:5]}{' ...' if len(items) > 5 else ''}")

    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")
    print("Integrity checks:", "PASS" if r["ok"] else "FAIL")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
