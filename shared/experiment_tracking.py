"""
Shared Weights & Biases config for all four technical members (Persons 1-4).

Goal: one `wandb.init(project=...)` config so everyone's runs land in the
same project and are comparable, without forcing wandb as a hard dependency
for people who haven't set it up yet (e.g. mid-training-loop CPU/Colab runs).

Usage (from any person's training/eval script):

    from shared.experiment_tracking import init_run, log

    run = init_run(person="person3", task="attention_consistency_train", config={
        "model": "segformer_b0_att",
        "lambda1": 1.0,
        "lambda2": 0.3,
        "sigma": 8.0,
        "epochs": 8,
    })
    ...
    log({"epoch": epoch, "dice": dice, "iou": iou, "loss": loss})

If `wandb` isn't installed or `WANDB_DISABLED=1` is set, `init_run` returns a
no-op run and `log` silently does nothing — existing scripts keep working
unmodified if you don't opt in.
"""
from __future__ import annotations

import os
from typing import Any, Optional

PROJECT_NAME = "dnn-forest-segformer-xai"
ENTITY = None  # set WANDB_ENTITY env var, or fill in your team/org name here


class _NoOpRun:
    """Drop-in stand-in when wandb isn't installed/enabled — every call is a no-op."""

    def log(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass


_active_run: Optional[Any] = None


def init_run(person: str, task: str, config: Optional[dict] = None, **kwargs) -> Any:
    """
    Start (or no-op) a shared W&B run.

    person: e.g. "person1", "person2", "person3", "person4"
    task:   short slug for what this run is, e.g. "unet_baseline",
            "segformer_vanilla", "attention_consistency_train", "ablation"
    config: hyperparameters/settings to log alongside the run (model name,
            lr, epochs, lambda1/lambda2, seed, etc.)
    """
    global _active_run

    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true"):
        _active_run = _NoOpRun()
        return _active_run

    try:
        import wandb
    except ImportError:
        print(
            "[experiment_tracking] wandb not installed — "
            "run `pip install wandb` to enable shared tracking. Logging disabled for this run."
        )
        _active_run = _NoOpRun()
        return _active_run

    run_name = f"{person}-{task}"
    _active_run = wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY or os.environ.get("WANDB_ENTITY"),
        name=run_name,
        group=person,
        job_type=task,
        config=config or {},
        **kwargs,
    )
    return _active_run


def log(data: dict, **kwargs) -> None:
    """Log a dict of metrics to the active run (no-op if init_run wasn't called or is disabled)."""
    if _active_run is None:
        print("[experiment_tracking] log() called before init_run() — ignoring:", data)
        return
    _active_run.log(data, **kwargs)


def finish() -> None:
    if _active_run is not None:
        _active_run.finish()
