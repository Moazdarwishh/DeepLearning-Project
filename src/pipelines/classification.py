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
    if path.exists():
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
    cfg.model.freeze_backbone = False
    model = build_classifier(cfg.model)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), cfg, ckpt, device


def evaluate(checkpoint: str | Path, split: str = "test", overrides: list[str] | None = None,
             analysis: bool = True, data=None) -> dict:
    import numpy as np

    from src.data.classification import build_classification_data, build_dataloaders
    from src.training.tasks import ClassificationTask
    from src.training.trainer import evaluate_model

    model, cfg, ckpt, device = load_model(checkpoint)
    cfg = config_from_dict(cfg.to_dict(), overrides)
    data = data or build_classification_data(cfg)
    loader = build_dataloaders(cfg, data)[split]
    task = ClassificationTask(cfg.model.num_classes, cfg.train.label_smoothing)
    metrics, logits, targets = evaluate_model(model, task, loader, device, prefix=split)

    run_dir = Path(checkpoint).parent
    (run_dir / f"eval_{split}.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(run_dir / f"predictions_{split}.npz", logits=logits.numpy(),
                        targets=targets.numpy(), indices=np.array(getattr(loader.dataset, "indices", [])))
    print(f"{split} ({len(targets)} images): " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    print(f"Saved -> {run_dir / f'eval_{split}.json'} and predictions_{split}.npz")

    if analysis:
        from src.evaluation.analysis import analyze, print_summary

        summary = analyze(logits.numpy(), targets.numpy(), ckpt["class_names"], run_dir / f"analysis_{split}",
                          dataset=loader.dataset, reference=data.train,
                          norm=(ckpt["norm_mean"], ckpt["norm_std"]))
        print_summary(summary)
        print(f"Analysis -> {run_dir / f'analysis_{split}'}")
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


def gradcam(checkpoints: list[str | Path], image: str | None = None, split: str = "val", num_samples: int = 6,
            output: str | Path = "outputs/gradcam", overrides: list[str] | None = None, data=None) -> list[Path]:
    """Grad-CAM for one image, or for random CORRECT and WRONG predictions of the first model."""
    import random

    from PIL import Image

    from src.data.classification import build_classification_data, build_dataloaders
    from src.data.transforms import build_eval_transform, denormalize
    from src.training.tasks import ClassificationTask
    from src.training.trainer import evaluate_model
    from src.visualization.gradcam import GradCAM, gradcam_grid

    loaded = [load_model(c) for c in checkpoints]
    names = [Path(c).parent.name for c in checkpoints]
    model0, cfg0, ckpt0, device = loaded[0]
    cfg0 = config_from_dict(cfg0.to_dict(), overrides)
    class_names = ckpt0["class_names"]
    norm0 = (ckpt0["norm_mean"], ckpt0["norm_std"])
    cams = [GradCAM(m) for m, *_ in loaded]
    out_dir = Path(output) / "_vs_".join(names)

    def explain(x0: torch.Tensor, label: int | None) -> dict:
        """Return the display image and one Grad-CAM map per model."""
        raw = denormalize(x0, *norm0)
        item = {"image": raw.permute(1, 2, 0).numpy(), "cams": [], "captions": [],
                "label": f"TRUE: {class_names[label]}" if label is not None else "input"}
        for (model, cfg, ckpt, _), cam in zip(loaded, cams):
            mean = torch.tensor(ckpt["norm_mean"]).view(3, 1, 1)
            std = torch.tensor(ckpt["norm_std"]).view(3, 1, 1)
            x = ((raw - mean) / std).unsqueeze(0).to(device)
            heat, logits, idx = cam(x)
            prob = logits.softmax(1)[0, idx[0]].item()
            pred = int(idx[0])
            mark = "" if label is None else (" OK" if pred == label else " WRONG")
            item["cams"].append(heat[0].cpu().numpy())
            item["captions"].append(f"pred: {class_names[pred][:22]} ({prob:.0%}){mark}")
        return item

    saved = []
    if image:
        x0 = build_eval_transform(cfg0.data)(Image.open(image).convert("RGB"))
        path = gradcam_grid([explain(x0, None)], names, out_dir / f"{Path(image).stem}_gradcam.png",
                            f"Grad-CAM: {Path(image).name}")
        saved.append(path)
    else:
        data = data or build_classification_data(cfg0)
        loader = build_dataloaders(cfg0, data)[split]
        task = ClassificationTask(cfg0.model.num_classes)
        _, logits, targets = evaluate_model(model0, task, loader, device, prefix=split)
        preds = logits.argmax(1)
        rng = random.Random(cfg0.experiment.seed)
        correct = [i for i in range(len(targets)) if preds[i] == targets[i]]
        wrong = [i for i in range(len(targets)) if preds[i] != targets[i]]
        for kind, pool in (("correct", correct), ("wrong", wrong)):
            if not pool:
                continue
            chosen = rng.sample(pool, min(num_samples, len(pool)))
            items = []
            for i in chosen:
                x0, y = loader.dataset[i]
                items.append(explain(x0, int(y)))
            title = (f"Grad-CAM on {kind.upper()} predictions of {names[0]} ({split} set) - "
                     "red = regions that drove the prediction")
            saved.append(gradcam_grid(items, names, out_dir / f"gradcam_{split}_{kind}.png", title))
    for c in cams:
        c.remove()
    for p in saved:
        print(f"saved -> {p}")
    return saved


def compare(run_dirs: list[str | Path], split: str = "test", output: str | Path = "outputs") -> list[dict]:
    from src.evaluation.analysis import compare_runs
    return compare_runs(run_dirs, split, output)


@register("train", "classification")
def _cli_train(args) -> int:
    cfg = load_config(args.config, args.set)
    if cfg.experiment.task != "classification":
        raise ValueError(f"Config is for task '{cfg.experiment.task}', not classification")
    train(cfg, resume=args.resume)
    return 0


@register("evaluate", "classification")
def _cli_evaluate(args) -> int:
    evaluate(args.checkpoint, args.split, args.set, analysis=not args.no_analysis)
    return 0


@register("gradcam", "classification")
def _cli_gradcam(args) -> int:
    gradcam(args.checkpoint, args.image, args.split, args.num_samples, args.output, args.set)
    return 0


@register("compare", "classification")
def _cli_compare(args) -> int:
    compare(args.runs, args.split, args.output)
    return 0


@register("predict", "classification")
def _cli_predict(args) -> int:
    predict(args.checkpoint, args.image, args.topk, args.output)
    return 0


@register("tune", "classification")
def _cli_tune(args) -> int:
    from src.training.tuning import SearchRunner, compare_searches

    if args.compare:
        compare_searches(args.compare)
        return 0
    if not args.config:
        raise SystemExit("tune needs --config (e.g. configs/tune_grid.yaml) or --compare DIR DIR")
    SearchRunner(load_config(args.config, args.set), base_config_path=args.config).run()
    return 0
