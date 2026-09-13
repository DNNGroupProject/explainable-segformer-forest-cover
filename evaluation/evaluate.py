"""
Person 4 shared evaluation CLI.

Usage
-----
    cd Person4_Evaluation
    python evaluate.py --model unet
    python evaluate.py --model unet --max-samples 200
    python evaluate.py --model segformer          # waits on Person 2
    python evaluate.py --model unet --attention-npy path/to/att.npy
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import config
from aamo import compute_aamo, mean_aamo
from efficiency import efficiency_row
from metrics import ConfusionCounts, binarize, format_metrics, metrics_from_counts


def get_adapter(model_name: str, checkpoint: Optional[str] = None):
    name = model_name.lower().strip()
    if name in ("unet", "u-net", "unet_torch"):
        from adapters.unet_torch import UnetTorchAdapter

        ad = UnetTorchAdapter()
    elif name in ("unet_keras", "unet-keras", "keras"):
        from adapters.unet_keras import UnetKerasAdapter

        ad = UnetKerasAdapter()
    elif name in ("segformer", "segformer-b0", "vanilla"):
        from adapters.segformer import SegFormerAdapter

        ad = SegFormerAdapter("vanilla")
    elif name in ("segformer-att", "att"):
        from adapters.segformer import SegFormerAdapter

        ad = SegFormerAdapter("att")
    elif name in ("segformer-boundary", "boundary"):
        # Boundary Refinement Module is a Phase 2 stretch goal (proposal
        # §3.4) — not implemented yet, still a stub.
        from adapters.segformer_stub import SegFormerStubAdapter

        ad = SegFormerStubAdapter("boundary")
    elif name in ("deeplab", "deeplabv3", "deeplabv3+", "deeplabv3plus"):
        from adapters.deeplabv3 import DeepLabV3Adapter

        ad = DeepLabV3Adapter()
    else:
        raise SystemExit(
            f"Unknown --model {model_name}. "
            "Use: unet | unet_torch | unet_keras | segformer | segformer-att | "
            "segformer-boundary | deeplab"
        )
    ad.load(checkpoint)
    return ad


def save_prediction_grid(
    images: np.ndarray,
    masks: np.ndarray,
    probs: np.ndarray,
    out_path: Path,
    n: int = 6,
    threshold: float = 0.5,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(n, len(images))
    preds = binarize(probs[:n], threshold)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    for i in range(n):
        axes[i, 0].imshow(np.clip(images[i], 0, 1))
        axes[i, 0].set_title("Image")
        axes[i, 1].imshow(masks[i], cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("GT")
        axes[i, 2].imshow(preds[i], cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title("Pred")
        for j in range(3):
            axes[i, j].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def row_dict(
    model_label: str,
    seg: Dict[str, float],
    eff: Dict[str, Any],
    aamo_val: Any,
) -> Dict[str, Any]:
    return {
        "model": model_label,
        "dice": round(seg["dice"], 4),
        "iou": round(seg["iou"], 4),
        "f1": round(seg["f1"], 4),
        "precision": round(seg["precision"], 4),
        "recall": round(seg["recall"], 4),
        "pixel_acc": round(seg["pixel_acc"], 4),
        "aamo": aamo_val if aamo_val != "n/a" else "n/a",
        "params": eff["params"],
        "gflops": eff["gflops"],
        "fps": eff["fps"],
        "ms_per_image": eff["ms_per_image"],
    }


def write_comparison_table(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "baseline_comparison.csv"
    md_path = out_dir / "baseline_comparison.md"
    fields = [
        "model",
        "dice",
        "iou",
        "f1",
        "precision",
        "recall",
        "pixel_acc",
        "aamo",
        "params",
        "gflops",
        "fps",
        "ms_per_image",
    ]

    # Merge with existing rows (same model name replaced)
    existing: List[Dict[str, Any]] = []
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    by_name = {r["model"]: r for r in existing}
    for r in rows:
        by_name[r["model"]] = {k: r.get(k, "") for k in fields}
    merged = list(by_name.values())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in merged:
            w.writerow({k: r.get(k, "") for k in fields})

    lines = [
        "# Baseline comparison (Person 4)",
        "",
        "| Model | Dice | IoU | F1 | AAMO | Params | GFLOPs | FPS |",
        "|-------|------|-----|----|------|--------|--------|-----|",
    ]
    for r in merged:
        lines.append(
            f"| {r.get('model','')} | {r.get('dice','')} | {r.get('iou','')} | "
            f"{r.get('f1','')} | {r.get('aamo','')} | {r.get('params','')} | "
            f"{r.get('gflops','')} | {r.get('fps','')} |"
        )
    lines.append("")
    lines.append("Proposal Table 2 columns + efficiency. AAMO = n/a until attention maps exist.")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def evaluate_one(args) -> Dict[str, Any]:
    adapter = get_adapter(args.model, args.checkpoint)
    print(f"Evaluating: {adapter.name}")
    print(f"Checkpoint: {adapter.checkpoint}")

    images, masks, probs, attentions = adapter.predict_dataset(
        max_samples=args.max_samples,
        split=args.split,
        batch_size=args.batch_size,
    )

    # Full-set confusion
    counts = ConfusionCounts()
    preds = binarize(probs, args.threshold)
    counts.update(preds, masks)
    seg = metrics_from_counts(counts)
    print(format_metrics(seg))

    # Efficiency
    params = adapter.count_params()
    gflops = adapter.estimate_gflops()
    speed = adapter.measure_speed()
    eff = efficiency_row(
        params=params,
        gflops=gflops,
        fps=speed.get("fps"),
        ms_per_image=speed.get("ms_per_image"),
    )
    print(
        f"params={eff['params']} | gflops={eff['gflops']} | "
        f"fps={eff['fps']} | ms/img={eff['ms_per_image']}"
    )

    # AAMO
    aamo_val: Any = "n/a"
    if args.attention_npy:
        att = np.load(args.attention_npy)
        # Expect (N,H,W) aligned with this split, or single map for demo
        if att.ndim == 2:
            aamo_val = round(compute_aamo(att, masks[0], thr=args.aamo_thr)["aamo"], 4)
        elif att.ndim == 3:
            n = min(len(att), len(masks))
            aamo_val = round(
                mean_aamo(list(att[:n]), list(masks[:n]), thr=args.aamo_thr)["aamo"], 4
            )
        print(f"AAMO (from --attention-npy) = {aamo_val}")
    elif attentions is not None:
        aamo_val = round(
            mean_aamo(list(attentions), list(masks), thr=args.aamo_thr)["aamo"], 4
        )
        print(f"AAMO = {aamo_val}")
    else:
        print("AAMO = n/a (no attention maps; U-Net has none / SegFormer pending Person 2)")

    if not args.no_grid:
        grid_path = config.RESULTS_DIR / f"prediction_grid_{args.model}.png"
        save_prediction_grid(images, masks, probs, grid_path, threshold=args.threshold)
        print(f"Wrote {grid_path}")

    row = row_dict(adapter.name, seg, eff, aamo_val)
    write_comparison_table([row], config.RESULTS_DIR)

    detail_path = config.RESULTS_DIR / f"eval_{args.model}.json"
    detail_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def parse_args():
    p = argparse.ArgumentParser(description="Person 4 — shared evaluation")
    p.add_argument(
        "--model",
        default="unet",
        help="unet | unet_torch | unet_keras | segformer | segformer-att | segformer-boundary | deeplab",
    )
    p.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--threshold", type=float, default=config.DEFAULT_THRESHOLD)
    p.add_argument("--aamo-thr", type=float, default=config.AAMO_THRESHOLD)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap paired images before split. Omit for the full 5108-pair split.",
    )
    p.add_argument(
        "--attention-npy",
        default=None,
        help="Optional precomputed attention maps for AAMO (Person 2 dump)",
    )
    p.add_argument("--no-grid", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        evaluate_one(args)
    except Exception as e:
        # Friendly message for SegFormer stub
        from adapters.segformer_stub import SegFormerNotReadyError

        if isinstance(e, SegFormerNotReadyError):
            print(str(e), file=sys.stderr)
            sys.exit(2)
        raise


if __name__ == "__main__":
    main()
