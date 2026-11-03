"""Classification pipeline: train / evaluate / predict (wired to the CLI).

Every training run writes to ``outputs/runs/<experiment.name>/``:

    config.yaml     exact config used (after --set overrides)
    best.pt         checkpoint with the best validation score (+ config + class names)
    history.json    per-epoch metrics
    curves.png      learning curves
    summary.json    final validation metrics of the best checkpoint, timing, parameters

The test set is only touched by ``evaluate --split test``, never during training.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import torch

from src.utils.commands import register
from src.utils.config import config_from_dict, load_config


def _run_dir(cfg, resume: str | None) -> Path:
    if resume:
        return Path(resume).parent
    base = Path(cfg.experiment.output_dir) / "runs"
    path = base / cfg.experiment.name
    if path.exists():                       # never overwrite an earlier run
        path = base / f"{cfg.experiment.name}-{datetime.now():%Y%m%d-%H%M%S}"
    return path


def train(cfg, resume: str | None = None, data=None) -> dict:
    from src.data.classification import build_classification_data, build_dataloaders
    from src.data.transforms import normalization_stats
    from src.models import build_classifier, count_parameters
    from src.training.optim import build_optimizer, build_scheduler
    from src.training.tasks import ClassificationTask
    from src.training.trainer import Trainer
    from src.utils.reproducibility import get_device, git_commit_hash, set_seed
    from src.utils.tracking import build_tracker
    from src.visualization.training_plots import plot_history

    set_seed(cfg.experiment.seed)
    device = get_device()
    run_dir = _run_dir(cfg, resume)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.yaml")

    data = data or build_classification_data(cfg)
    loaders = build_dataloaders(cfg, data)
    model = build_classifier(cfg.model)
    params = count_parameters(model)
    steps = len(loaders["train"])
    if cfg.train.limit_train_batches:
        steps = min(steps, cfg.train.limit_train_batches)
    optimizer = build_optimizer(model, cfg.train)
    scheduler = build_scheduler(optimizer, cfg.train, steps)
    task = ClassificationTask(cfg.model.num_classes, cfg.train.label_smoothing)

    mean, std = normalization_stats(cfg.data)
    run_info = {"config": cfg.to_dict(), "class_names": data.class_names, "norm_mean": mean, "norm_std": std}
    tracker = build_tracker(cfg.tracking.backend, cfg.tracking.project, cfg.tracking.tracking_uri,
                            cfg.tracking.entity)
    extra = {"run.dir": str(run_dir), "run.device": str(device), "run.git_commit": git_commit_hash(),
             "model.params_total": params["total"], "model.params_trainable": params["trainable"],
             **{f"data.size_{k}": v for k, v in data.sizes().items()}}
    tracker.start_run(run_dir.name, {**cfg.flat(), **extra}, cfg.experiment.tags)

    print(f"Run dir   : {run_dir}")
    print(f"Device    : {device}   |   params: {params['total']:,} ({params['trainable']:,} trainable)")
    print(f"Data      : {data.sizes()}   |   {steps} steps/epoch, batch {cfg.data.batch_size}")

    trainer = Trainer(model, task, optimizer, scheduler, device, tracker, cfg.train, run_dir, run_info)
    if resume:
        trainer.resume(resume)
    start = time.perf_counter()
    try:
        result = trainer.fit(loaders["train"], loaders["val"])
        train_time = time.perf_counter() - start

        # Final, full validation pass with the best weights
        best = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best["model_state"])
        val_metrics, _, _ = trainer.evaluate(loaders["val"], prefix="val")
        summary = {"run": run_dir.name, **result, "train_time_min": round(train_time / 60, 2),
                   "params": params, "best_val": val_metrics}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        curves = plot_history(trainer.history, run_dir / "curves.png", title=run_dir.name,
                              best_epoch=result["best_epoch"])
        tracker.log_metrics({f"best_{k}": v for k, v in val_metrics.items()})
        for artifact in (curves, run_dir / "summary.json", run_dir / "history.json", run_dir / "config.yaml"):
            tracker.log_artifact(artifact)
    finally:
        tracker.end_run()
    print(f"\nBest epoch {result['best_epoch']}: " +
          ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
    print(f"Saved -> {run_dir}")
    return summary


def load_model(checkpoint: str | Path, device=None):
    """Rebuild the model exactly as it was trained (config is stored in the checkpoint)."""
    from src.models import build_classifier
    from src.utils.reproducibility import get_device

    device = device or get_device()
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = config_from_dict(ckpt["config"])
    cfg.model.freeze_backbone = False            # irrelevant at inference
    model = build_classifier(cfg.model)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), cfg, ckpt, device


def evaluate(checkpoint: str | Path, split: str = "test", overrides: list[str] | None = None) -> dict:
    import numpy as np

    from src.data.classification import build_classification_data, build_dataloaders
    from src.training.tasks import ClassificationTask
    from src.training.trainer import evaluate_model

    model, cfg, ckpt, device = load_model(checkpoint)
    cfg = config_from_dict(cfg.to_dict(), overrides)
    data = build_classification_data(cfg)
    loader = build_dataloaders(cfg, data)[split]
    task = ClassificationTask(cfg.model.num_classes, cfg.train.label_smoothing)
    metrics, logits, targets = evaluate_model(model, task, loader, device, prefix=split)

    run_dir = Path(checkpoint).parent
    (run_dir / f"eval_{split}.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(run_dir / f"predictions_{split}.npz", logits=logits.numpy(),
                        targets=targets.numpy(), indices=np.array(getattr(loader.dataset, "indices", [])))
    print(f"{split} ({len(targets)} images): " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    print(f"Saved -> {run_dir / f'eval_{split}.json'} and predictions_{split}.npz")
    return metrics


def predict(checkpoint: str | Path, image: str | Path, topk: int = 5, output: str | None = None) -> list[dict]:
    from PIL import Image

    from src.data.transforms import build_eval_transform

    model, cfg, ckpt, device = load_model(checkpoint)
    names = ckpt["class_names"]
    tf = build_eval_transform(cfg.data)
    path = Path(image)
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}) \
        if path.is_dir() else [path]
    results = []
    with torch.no_grad():
        for f in files:
            x = tf(Image.open(f).convert("RGB")).unsqueeze(0).to(device)
            probs = model(x).softmax(dim=1)[0].cpu()
            top = probs.topk(min(topk, len(names)))
            preds = [{"class": names[i], "probability": round(float(p), 4)}
                     for p, i in zip(top.values, top.indices.tolist())]
            results.append({"image": str(f), "predictions": preds})
            print(f"\n{f.name}")
            for rank, pr in enumerate(preds, 1):
                print(f"  {rank}. {pr['class']:<32} {pr['probability']:.1%}")
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(results, indent=2))
        print(f"\nSaved -> {output}")
    return results


# ------------------------------------------------------------------ CLI hooks
@register("train", "classification")
def _cli_train(args) -> int:
    cfg = load_config(args.config, args.set)
    if cfg.experiment.task != "classification":
        raise ValueError(f"Config is for task '{cfg.experiment.task}', not classification")
    train(cfg, resume=args.resume)
    return 0


@register("evaluate", "classification")
def _cli_evaluate(args) -> int:
    evaluate(args.checkpoint, args.split, args.set)
    return 0


@register("predict", "classification")
def _cli_predict(args) -> int:
    predict(args.checkpoint, args.image, args.topk, args.output)
    return 0
