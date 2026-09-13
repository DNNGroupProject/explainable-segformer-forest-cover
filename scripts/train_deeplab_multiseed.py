"""
DeepLabV3+ multi-seed training + evaluation.

Full scale (paper numbers):
    DEEPLAB_MAX_SAMPLES=100000 DEEPLAB_EPOCHS=20 DEEPLAB_BATCH=8 \\
      python -m scripts.train_deeplab_multiseed

Uses the 3576/766/766 seed-42 split (`deeplab_data.py`). Trains seeds
42/43/44 from scratch. Writes `n_test` into `deeplab_multiseed.json` —
must be 766.

Smoke defaults (local CPU):
    DEEPLAB_MAX_SAMPLES=400 DEEPLAB_EPOCHS=5  (not paper-comparable)

Usage
-----
    python -m scripts.train_deeplab_multiseed
    python -m scripts.train_deeplab_multiseed --skip-train
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
EVAL_DIR = REPO_ROOT / "evaluation"
RESULTS = REPO_ROOT / "results"
CKPT_DIR = REPO_ROOT / "checkpoints"
RESULTS.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from adapters.deeplab_model import build_deeplabv3, forest_prob_from_logits  # noqa: E402
from metrics import ConfusionCounts, binarize, metrics_from_counts  # noqa: E402

import config  # noqa: E402
from deeplab_data import (  # noqa: E402
    IMG_DIR,
    MASK_DIR,
    N_TEST,
    N_TRAIN,
    N_VAL,
    load_split_arrays,
    make_splits_files,
)

SEEDS = [42, 43, 44]
DEEPLAB_LABEL = "DeepLabV3+ (MobileNetV3) — extra baseline"

MAX_SAMPLES = int(os.environ.get("DEEPLAB_MAX_SAMPLES", "400"))
EPOCHS = int(os.environ.get("DEEPLAB_EPOCHS", "5"))
BATCH_SIZE = int(os.environ.get("DEEPLAB_BATCH", "8" if MAX_SAMPLES >= 5000 else "2"))
LR = float(os.environ.get("DEEPLAB_LR", "1e-4"))

# Full-scale when the env asks for more samples than the smoke subset.
FULL_SCALE = MAX_SAMPLES >= (N_TRAIN + N_VAL + N_TEST)


def aggregate_mean_std(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sample std (ddof=1) across seeds; bare mean when n<2."""
    numeric = ["dice", "iou", "f1", "precision", "recall", "pixel_acc"]
    out: Dict[str, Any] = {"model": rows[0]["model"], "n_seeds": len(rows)}

    def fmt(vals: List[float]) -> str:
        if not vals:
            return "n/a"
        m = float(np.mean(vals))
        if len(vals) < 2:
            return f"{m:.4f}"
        return f"{m:.4f} ± {float(np.std(vals, ddof=1)):.4f}"

    for k in numeric:
        vals: List[float] = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (TypeError, ValueError, KeyError):
                pass
        out[k] = fmt(vals)

    avals: List[float] = []
    for r in rows:
        v = r.get("aamo")
        if v not in (None, "n/a", ""):
            try:
                avals.append(float(v))
            except (TypeError, ValueError):
                pass
    out["aamo"] = fmt(avals)
    out["params"] = rows[0].get("params", "n/a")
    out["gflops"] = rows[0].get("gflops", "n/a")
    return out


def _to_tensor_images(images: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def dice_loss_from_probs(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = probs.reshape(probs.size(0), -1)
    t = targets.reshape(targets.size(0), -1)
    inter = (p * t).sum(dim=1)
    return (1 - (2 * inter + eps) / (p.sum(dim=1) + t.sum(dim=1) + eps)).mean()


@torch.no_grad()
def eval_split(model, images, masks, device, batch_size=4) -> dict:
    model.eval()
    counts = ConfusionCounts()
    for i in range(0, len(images), batch_size):
        xb = _to_tensor_images(images[i : i + batch_size]).to(device)
        out = model(xb)["out"]
        if out.shape[-2:] != (config.IMG_SIZE, config.IMG_SIZE):
            out = F.interpolate(
                out, size=(config.IMG_SIZE, config.IMG_SIZE), mode="bilinear", align_corners=False
            )
        probs = forest_prob_from_logits(out).cpu().numpy()
        preds = binarize(probs, 0.5)
        counts.update(preds, masks[i : i + batch_size])
    return metrics_from_counts(counts)


def _load_baseline_arrays():
    """Load full train/val/test arrays (fixed seed-42 split)."""
    if not IMG_DIR.is_dir() or not MASK_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset missing:\n  {IMG_DIR}\n  {MASK_DIR}\n"
            "Place the dataset under data/images and data/masks (see README)."
        )
    X_train, y_train, _ = load_split_arrays("train", seed=42)
    X_val, y_val, _ = load_split_arrays("val", seed=42)
    X_test, y_test, test_names = load_split_arrays("test", seed=42)
    assert len(X_test) == N_TEST, f"n_test={len(X_test)} expected {N_TEST}"
    return X_train, y_train, X_val, y_val, X_test, y_test, test_names


def train_one_seed(seed: int, device: torch.device, splits_arrays) -> Path:
    """Train DeepLabV3+ for one init seed on the fixed dataset split."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X_train, y_train, X_val, y_val, _, _, _ = splits_arrays

    train_x = _to_tensor_images(X_train)
    train_y_f = torch.from_numpy(y_train).float()
    loader = DataLoader(
        TensorDataset(train_x, train_y_f),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    model = build_deeplabv3(num_classes=2, pretrained_backbone=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_iou = -1.0
    best_path = CKPT_DIR / f"deeplabv3_mobilenet_seed{seed}_best.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)["out"]
            if out.shape[-2:] != (config.IMG_SIZE, config.IMG_SIZE):
                out = F.interpolate(
                    out, size=(config.IMG_SIZE, config.IMG_SIZE), mode="bilinear", align_corners=False
                )
            ce = F.cross_entropy(out, yb.long())
            probs = forest_prob_from_logits(out)
            dsc = dice_loss_from_probs(probs, yb)
            loss = 0.5 * ce + 0.5 * dsc
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        val = eval_split(model, X_val, y_val, device, batch_size=BATCH_SIZE)
        print(
            f"  seed={seed} epoch {epoch}/{EPOCHS}  loss={np.mean(losses):.4f}  "
            f"val_dice={val['dice']:.4f}  val_iou={val['iou']:.4f}"
        )
        ckpt = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "val_dice": val["dice"],
            "val_iou": val["iou"],
            "variant": "deeplabv3_mobilenet",
            "seed": seed,
            "max_samples": MAX_SAMPLES,
            "full_scale": FULL_SCALE,
            "n_train": N_TRAIN if FULL_SCALE else len(X_train),
            "n_val": N_VAL if FULL_SCALE else len(X_val),
            "n_test": N_TEST if FULL_SCALE else None,
            "split": "3576_766_766_seed42",
        }
        if val["iou"] > best_iou:
            best_iou = val["iou"]
            torch.save(ckpt, best_path)
            print(f"    saved best -> {best_path} (iou={best_iou:.4f})")

    return best_path


def evaluate_checkpoint(
    ckpt_path: Path, seed: int, device: torch.device, X_test, y_test
) -> Dict[str, Any]:
    """Evaluate on the fixed seed-42 held-out test set."""
    model = build_deeplabv3(num_classes=2, pretrained_backbone=False).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    metrics = eval_split(model, X_test, y_test, device, batch_size=BATCH_SIZE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    try:
        ckpt_str = ckpt_path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        ckpt_str = ckpt_path.name

    n_test = int(len(X_test))
    if FULL_SCALE and n_test != N_TEST:
        raise RuntimeError(f"n_test={n_test} but full-scale requires {N_TEST}")

    return {
        "model": DEEPLAB_LABEL,
        "seed": seed,
        "dice": round(float(metrics["dice"]), 4),
        "iou": round(float(metrics["iou"]), 4),
        "f1": round(float(metrics["f1"]), 4),
        "precision": round(float(metrics["precision"]), 4),
        "recall": round(float(metrics["recall"]), 4),
        "pixel_acc": round(float(metrics["pixel_acc"]), 4),
        "aamo": "n/a",
        "params": n_params,
        "gflops": "n/a",
        "checkpoint": ckpt_str,
        "max_samples": MAX_SAMPLES,
        "n_test": n_test,
        "full_scale": FULL_SCALE,
        "split": "3576_766_766_seed42",
        "status": "ok",
    }


def _ckpt_for_seed(seed: int) -> Path:
    return CKPT_DIR / f"deeplabv3_mobilenet_seed{seed}_best.pt"


def load_folded_full_scale() -> List[Dict[str, str]]:
    path = RESULTS / "baseline_comparison_full_scale.csv"
    if not path.exists():
        from fold_full_scale_results import fold, write_tables

        write_tables(fold())
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_ablation_tables(
    deeplab_per_seed: List[Dict[str, Any]],
    folded: List[Dict[str, str]],
) -> None:
    per_seed_rows: List[Dict[str, Any]] = []

    for r in folded:
        model = r.get("model", "")
        if "DeepLab" in model:
            continue
        per_seed_rows.append(
            {
                "model": model,
                "seed": 42,
                "dice": r.get("dice", "-"),
                "iou": r.get("iou", "-"),
                "f1": r.get("f1", "-"),
                "aamo": r.get("aamo", "n/a"),
                "params": r.get("params", "n/a"),
                "gflops": r.get("gflops", "n/a"),
                "status": "single_seed_full_scale",
            }
        )

    for r in deeplab_per_seed:
        per_seed_rows.append(r)

    per_path = RESULTS / "ablation_per_seed.csv"
    fields = [
        "model",
        "seed",
        "dice",
        "iou",
        "f1",
        "precision",
        "recall",
        "pixel_acc",
        "aamo",
        "params",
        "gflops",
        "status",
        "checkpoint",
        "max_samples",
        "n_test",
        "full_scale",
    ]
    with open(per_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in per_seed_rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"Wrote {per_path}")

    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in per_seed_rows:
        if r.get("dice") in ("-", "", None):
            continue
        by_model.setdefault(r["model"], []).append(r)

    summary: List[Dict[str, Any]] = []
    for name in [r["model"] for r in folded]:
        if name not in by_model:
            continue
        summary.append(aggregate_mean_std(by_model[name]))

    sum_csv = RESULTS / "ablation_mean_std.csv"
    sum_md = RESULTS / "ablation_mean_std.md"
    sum_fields = ["model", "n_seeds", "dice", "iou", "f1", "aamo", "params", "gflops"]
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields, extrasaction="ignore")
        w.writeheader()
        for r in summary:
            w.writerow({k: r.get(k, "") for k in sum_fields})

    scale_note = (
        f"full-scale split 3576/766/766, {EPOCHS} epochs"
        if FULL_SCALE
        else f"smoke (max_samples={MAX_SAMPLES}, {EPOCHS} epochs) — not paper-comparable"
    )
    lines = [
        "# Ablation results (mean ± std)",
        "",
        "Std is **sample** standard deviation (ddof=1). Single-seed SegFormer/U-Net",
        "rows show the bare value. DeepLabV3+: " + scale_note + ".",
        "",
        "| Model | Seeds | Dice | IoU | F1 | AAMO |",
        "|-------|-------|------|-----|----|------|",
    ]
    for r in summary:
        lines.append(
            f"| {r.get('model')} | {r.get('n_seeds')} | {r.get('dice')} | "
            f"{r.get('iou')} | {r.get('f1')} | {r.get('aamo')} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- DeepLab split = `dataset.py` make_splits (3576/766/766 seed 42).",
            "- Confirm `n_test=766` in `deeplab_multiseed.json` before trusting numbers.",
            "- Sample std (ddof=1) for multi-seed rows.",
            "",
        ]
    )
    sum_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {sum_csv}")
    print(f"Wrote {sum_md}")

    dl_json = RESULTS / "deeplab_multiseed.json"
    dl_json.write_text(json.dumps(deeplab_per_seed, indent=2), encoding="utf-8")
    print(f"Wrote {dl_json}")


def main() -> None:
    p = argparse.ArgumentParser(description="DeepLabV3+ multi-seed + ablation tables")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} | max_samples={MAX_SAMPLES} | epochs={EPOCHS} | "
        f"batch={BATCH_SIZE} | full_scale={FULL_SCALE} | seeds={args.seeds}"
    )
    if not FULL_SCALE:
        print(
            "WARNING: not full-scale. For Part C set "
            "DEEPLAB_MAX_SAMPLES=100000 DEEPLAB_EPOCHS=20"
        )

    # Always use the aligned split for the paper path; smoke still uses it
    # but only when dataset is present — for tiny smoke without wanting 5k
    # images, require FULL_SCALE or explicit env.
    if not FULL_SCALE:
        # Legacy smoke: load first MAX_SAMPLES via Phase1 adapters (unchanged sizes)
        from adapters.data import load_pairs, split_dataset  # local smoke only

        images, masks = load_pairs(max_samples=MAX_SAMPLES)
        splits = split_dataset(images, masks, seed=42)
        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]
        X_test, y_test = splits["test"]
        splits_arrays = (X_train, y_train, X_val, y_val, X_test, y_test, [])
        print(f"smoke split sizes train/val/test = {len(X_train)}/{len(X_val)}/{len(X_test)}")
    else:
        print("Loading 3576/766/766 arrays (seed 42)...")
        splits_arrays = _load_baseline_arrays()
        X_train, y_train, X_val, y_val, X_test, y_test, _ = splits_arrays
        print(f"split sizes train/val/test = {len(X_train)}/{len(X_val)}/{len(X_test)}")
        assert len(X_test) == N_TEST

    deeplab_rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        ckpt = _ckpt_for_seed(seed)
        if not args.skip_train:
            if not ckpt.exists():
                print(f"\n=== Training DeepLabV3+ seed={seed} ===")
                ckpt = train_one_seed(seed, device, splits_arrays)
            else:
                print(f"\n=== Reusing existing checkpoint for seed={seed}: {ckpt} ===")
        elif not ckpt.exists():
            print(f"[skip] seed={seed}: no checkpoint at {ckpt}")
            continue

        print(f"\n=== Evaluating DeepLabV3+ seed={seed} | {ckpt} ===")
        row = evaluate_checkpoint(ckpt, seed, device, X_test, y_test)
        print(
            f"  seed={seed} test dice={row['dice']:.4f} iou={row['iou']:.4f} "
            f"f1={row['f1']:.4f} n_test={row['n_test']}"
        )
        deeplab_rows.append(row)

    folded = load_folded_full_scale()
    write_ablation_tables(deeplab_rows, folded)
    print("\nMulti-seed DeepLab + ablation tables done.")


if __name__ == "__main__":
    main()
