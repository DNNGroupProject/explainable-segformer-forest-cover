"""Augmentation ablation for the U-Net baseline (Person 1, Weeks 3-4).

Trains the same U-Net twice -- once without augmentation, once with
`shared.augmentation` on the training set -- and reports what the difference
buys. Everything else is held fixed: same split, same seed, same starting
weights, same loss, same loop. The augmentation flag is the only variable,
which is what makes the delta attributable to it.

Uses the notebook's own model, dataset and training loop (via unet_model.py
and dataset.py, both lifted out of `unet_baseline_colab.ipynb`), so arm A is
directly comparable to the committed baseline in `test_metrics.txt`.

Run (CPU smoke scale -- a few hundred pairs, narrow U-Net, ~15 min):
    python augmentation_ablation.py

Run (full scale, needs a GPU -- this is the number for the paper):
    python augmentation_ablation.py --epochs 20 --features 64,128,256,512 \
        --subset 0 --batch-size 8 --device cuda --amp

--amp is roughly 3x faster on a T4 and applies to both arms identically, so
it moves the wall clock and not the comparison. Without it the full run is
5-6 hours, which is more than a free Colab session reliably lasts.

Exits 0 on a completed run, 1 if a directory is missing or the split comes
out empty.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import (  # noqa: E402
    ForestSegDataset,
    list_mask_files,
    make_splits,
    seed_worker,
    set_seed,
)
from unet_model import build_model, dice_iou_score  # noqa: E402

HERE = Path(__file__).resolve().parent

# Smoke defaults. The full-scale settings live in the docstring above rather
# than here, so running the script by accident can't burn a GPU-day.
SMOKE_SUBSET = 600
SMOKE_EPOCHS = 4
SMOKE_FEATURES = (16, 32, 64, 128)


def run_epoch(model, loader, device, optimizer=None, scaler=None):
    """One pass over a loader. The notebook's Step 5 loop -- plain
    cross-entropy, no scheduler, no gradient clipping.

    scaler: pass a GradScaler to run in mixed precision. On a T4 that is
    roughly 3x faster, because the fp32 loop never touches the tensor cores.
    Both arms get the same treatment, so it changes the wall clock, not the
    comparison. None keeps the loop bit-for-bit as the notebook has it.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    use_amp = scaler is not None

    total_loss, total_dice, total_iou, n_batches = 0.0, 0.0, 0.0, 0

    with torch.set_grad_enabled(is_train):
        for images, masks in loader:
            # non_blocking pairs with pin_memory in the loader: the next
            # batch copies to the GPU while this one is still computing.
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                logits = F.interpolate(
                    logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )
                loss = F.cross_entropy(logits, masks)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    # fp16 gradients underflow without scaling; the scaler
                    # picks the factor itself and skips any step that overflows.
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            # Metrics in fp32 -- argmax is precision-insensitive, but keeping
            # the accumulation out of autocast avoids any doubt about whether
            # the reported Dice is affected by the speedup.
            preds = logits.float().argmax(dim=1).bool()
            dice, iou = dice_iou_score(preds, masks.bool())

            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            n_batches += 1

    return total_loss / n_batches, total_dice / n_batches, total_iou / n_batches


def train_one_arm(name, augment, splits, args, epoch_rows):
    """Train and test a single arm of the ablation.

    Both arms call set_seed(args.seed) right before build_model(), so they
    start from byte-identical weights and see batches in the same order. If
    that ever stops being true the comparison stops meaning anything.
    """
    train_files, val_files, test_files = splits
    set_seed(args.seed)

    common = dict(
        images_dir=args.images, masks_dir=args.masks, img_size=args.img_size
    )
    train_ds = ForestSegDataset(
        train_files, augment=augment, seed=args.seed, **common
    )
    # Val and test are never augmented -- the point is to vary what the model
    # trains on, not to move the ruler it's measured with.
    val_ds = ForestSegDataset(val_files, **common)
    test_ds = ForestSegDataset(test_files, **common)

    # seed_worker gives each worker its own augmentation stream; without it
    # forked workers replay the same one. pin_memory only helps on CUDA.
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        pin_memory=args.device == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)

    model = build_model(device=args.device, features=args.features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda") if args.amp else None

    best_val_dice = 0.0
    best_state = None
    started = time.time()

    print(f"\n=== arm: {name} (augment={augment}) ===")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice, train_iou = run_epoch(model, train_loader, args.device, optimizer, scaler)
        val_loss, val_dice, val_iou = run_epoch(model, val_loader, args.device, scaler=scaler)

        epoch_rows.append({
            "arm": name, "epoch": epoch,
            "train_loss": train_loss, "train_dice": train_dice, "train_iou": train_iou,
            "val_loss": val_loss, "val_dice": val_dice, "val_iou": val_iou,
        })
        print(
            f"epoch {epoch:02d} | train_loss={train_loss:.4f} dice={train_dice:.4f} | "
            f"val_loss={val_loss:.4f} dice={val_dice:.4f} iou={val_iou:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            # Kept in memory rather than on disk: the ablation wants the two
            # test numbers, not two more checkpoints to review.
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_dice, test_iou = run_epoch(model, test_loader, args.device, scaler=scaler)
    minutes = (time.time() - started) / 60

    print(f"test: loss={test_loss:.4f} dice={test_dice:.4f} iou={test_iou:.4f}  ({minutes:.1f} min)")
    return {
        "arm": name,
        "augment": augment,
        "best_val_dice": best_val_dice,
        "test_loss": test_loss,
        "test_dice": test_dice,
        "test_iou": test_iou,
        "minutes": minutes,
    }


def build_rows(baseline, augmented):
    """(arm, best val Dice, test Dice, test IoU, test loss) -- the same rows
    get written as .csv and .md, so the two files can't drift apart."""
    rows = [
        (
            r["arm"],
            f"{r['best_val_dice']:.4f}",
            f"{r['test_dice']:.4f}",
            f"{r['test_iou']:.4f}",
            f"{r['test_loss']:.4f}",
        )
        for r in (baseline, augmented)
    ]
    # The delta row is the actual deliverable; the two arms above are just
    # how it was arrived at. Signed, so the direction is unmissable.
    rows.append((
        "Delta (aug - none)",
        f"{augmented['best_val_dice'] - baseline['best_val_dice']:+.4f}",
        f"{augmented['test_dice'] - baseline['test_dice']:+.4f}",
        f"{augmented['test_iou'] - baseline['test_iou']:+.4f}",
        f"{augmented['test_loss'] - baseline['test_loss']:+.4f}",
    ))
    return rows


def write_reports(rows, epoch_rows, args, out_dir):
    """Write the table twice (.csv to compute with, .md to read in a PR) plus
    the per-epoch log, following the convention in CONTRIBUTING.md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    header = ["arm", "best_val_dice", "test_dice", "test_iou", "test_loss"]

    csv_path = out_dir / "augmentation_ablation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    scale = (
        f"{args.n_pairs} pairs (train/val/test = {args.n_train}/{args.n_val}/{args.n_test}), "
        f"{args.epochs} epochs, batch {args.batch_size}, lr {args.lr}, "
        f"{args.img_size}x{args.img_size}, features {','.join(str(f) for f in args.features)}, "
        f"seed {args.seed}, {args.device}"
    )

    md_path = out_dir / "augmentation_ablation.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Augmentation ablation (Person 1)\n\n")
        f.write(
            "Generated by `augmentation_ablation.py`. Same U-Net, same split, same "
            "seed, same starting weights -- the only difference is whether "
            "`shared.augmentation` is applied to the training set.\n\n"
        )
        f.write(f"Run: {scale}\n\n")
        f.write("| Arm | Best val Dice | Test Dice | Test IoU | Test loss |\n|---|---|---|---|---|\n")
        for row in rows:
            f.write("| " + " | ".join(row) + " |\n")

    log_path = out_dir / "augmentation_ablation_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(epoch_rows[0].keys()))
        writer.writeheader()
        writer.writerows(epoch_rows)

    return csv_path, md_path, log_path


def parse_features(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--masks", type=Path, default=None)
    parser.add_argument(
        "--subset", type=int, default=SMOKE_SUBSET,
        help="Use only the first N pairs after shuffling; 0 means the full dataset",
    )
    parser.add_argument("--epochs", type=int, default=SMOKE_EPOCHS)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument(
        "--features", type=parse_features, default=SMOKE_FEATURES,
        help="U-Net channel widths. 64,128,256,512 is the real baseline; the "
             "narrower default is what makes a CPU run finish.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--num-workers", type=int, default=2,
        help="DataLoader workers. >0 needs seed_worker, which is wired up below.",
    )
    parser.add_argument(
        "--amp", action="store_true",
        help="Mixed precision. ~3x faster on a T4; applied to both arms equally.",
    )
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = parser.parse_args(argv)

    # Fall back to dataset.py's defaults
    # so a bare `python augmentation_ablation.py` works on a fresh clone.
    import dataset as ds

    args.images = args.images or ds.DEFAULT_IMAGES_DIR
    args.masks = args.masks or ds.DEFAULT_MASKS_DIR
    for d in (args.images, args.masks):
        if not Path(d).is_dir():
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            return 1

    splits = make_splits(
        list_mask_files(args.masks), seed=args.seed, subset=args.subset or None
    )
    args.n_train, args.n_val, args.n_test = (len(s) for s in splits)
    args.n_pairs = args.n_train + args.n_val + args.n_test
    if min(args.n_train, args.n_val, args.n_test) == 0:
        print(f"ERROR: --subset {args.subset} is too small to split", file=sys.stderr)
        return 1

    print(
        f"{args.n_pairs} pairs -> train/val/test = "
        f"{args.n_train}/{args.n_val}/{args.n_test} | device={args.device} | "
        f"features={args.features} | {args.epochs} epochs"
    )

    epoch_rows = []
    baseline = train_one_arm("No augmentation", False, splits, args, epoch_rows)
    augmented = train_one_arm("With augmentation", True, splits, args, epoch_rows)

    rows = build_rows(baseline, augmented)
    csv_path, md_path, log_path = write_reports(rows, epoch_rows, args, args.out_dir)

    print("\n" + "-" * 68)
    width = max(len(r[0]) for r in rows)
    print(f"{'Arm':<{width}}  {'val Dice':>9}  {'test Dice':>9}  {'test IoU':>9}")
    for arm, val_dice, test_dice, test_iou, _ in rows:
        print(f"{arm:<{width}}  {val_dice:>9}  {test_dice:>9}  {test_iou:>9}")
    print("-" * 68)

    for p in (csv_path, md_path, log_path):
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
