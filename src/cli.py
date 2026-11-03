from __future__ import annotations

import argparse
import platform
import sys

from src.utils.commands import HANDLERS, register  # noqa: F401  (re-exported)
from src.utils.config import load_config

TASKS = ("classification", "colorization", "captioning")


def cmd_info(args: argparse.Namespace) -> int:
    """Print environment + resolved config."""
    print(f"Python      : {platform.python_version()} ({platform.system()} {platform.machine()})")
    try:
        import torch
        from src.utils.reproducibility import get_device
        print(f"PyTorch     : {torch.__version__}")
        print(f"Device      : {get_device()}")
    except ImportError:
        print("PyTorch     : NOT INSTALLED  ->  pip install -r requirements.txt")
    cfg = load_config(args.config, args.set)
    print(f"Config file : {args.config or '(defaults)'}")
    for key, value in cfg.flat().items():
        print(f"  {key:<32} {value}")
    return 0


def cmd_data(args: argparse.Namespace) -> int:
    """Dataset utilities."""
    import json
    from pathlib import Path

    from src.data.datasets import get_class_names, get_labels, load_cub, summarize

    cfg = load_config(args.config, args.set)
    out_dir = Path(cfg.experiment.output_dir) / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.action == "inspect":
        for split_name, repo in (("train", cfg.data.train_dataset), ("test", cfg.data.test_dataset)):
            print(f"\n=== {split_name}: {repo} ===")
            ds = load_cub(repo, cfg.data.cache_dir)
            summary = summarize(ds)
            summary["class_names_first_5"] = get_class_names(ds)[:5]
            print(json.dumps(summary, indent=2))
            path = out_dir / f"dataset_summary_{split_name}.json"
            path.write_text(json.dumps(summary, indent=2))
            print(f"saved -> {path}")
        return 0

    if args.action == "split":
        from src.data.splits import SPLIT_FILE, make_stratified_split, save_split

        target = Path(cfg.data.splits_dir) / SPLIT_FILE
        if target.exists() and not args.force:
            print(f"{target} already exists (use --force to overwrite).", file=sys.stderr)
            return 1
        ds = load_cub(cfg.data.train_dataset, cfg.data.cache_dir)
        labels = get_labels(ds)
        split = make_stratified_split(labels, cfg.data.val_fraction, cfg.experiment.seed,
                                      source_dataset=cfg.data.train_dataset)
        path = save_split(split, cfg.data.splits_dir)
        print(json.dumps(split.summary(labels), indent=2))
        print(f"saved -> {path}")
        return 0

    if args.action == "eda":
        from src.data.eda import run_eda

        summary = run_eda(cfg)
        short = {k: summary[k] for k in ("rows", "images_per_class", "color_modes", "quality_flags",
                                         "near_duplicates", "train_channel_mean_std", "cleaning")}
        short["corrupt_images"] = len(summary["corrupt_images"])
        print(json.dumps(short, indent=2, default=str))
        print(f"\nfull summary -> {out_dir / 'eda_summary.json'}")
        print(f"figures      -> {out_dir / 'figures'} ({len(summary['figures'])} files)")
        return 0

    if args.action == "augment-preview":
        import random

        from src.data.classification import build_classification_data
        from src.data.transforms import (build_eval_transform, build_train_transform,
                                         normalization_stats)
        from src.visualization.augmentation_plots import augmentation_grid

        data = build_classification_data(cfg)
        rng = random.Random(cfg.experiment.seed)
        picks = rng.sample(range(len(data.train)), 4)
        samples = []
        for i in picks:
            row = data.train.data[data.train.indices[i]]
            samples.append((row["image"].convert("RGB"), data.class_names[int(row["label"])]))
        mean, std = normalization_stats(cfg.data)
        path = augmentation_grid(samples, build_train_transform(cfg.data), build_eval_transform(cfg.data),
                                 mean, std, out_dir / "figures" / "11_augmentations.png", seed=cfg.experiment.seed)
        print(f"saved -> {path}")
        return 0

    if args.action == "loader-check":
        import time

        from src.data.classification import build_classification_data, build_dataloaders

        data = build_classification_data(cfg)
        loaders = build_dataloaders(cfg, data)
        print(f"dataset sizes  : {data.sizes()}")
        print(f"excluded       : {data.excluded}")
        print(f"normalization  : {cfg.data.normalization}, augment={cfg.data.augment}")
        for split in ("train", "val"):
            start = time.perf_counter()
            for n, (x, y) in enumerate(loaders[split], start=1):
                if n == 1:
                    print(f"{split:<5} batch    : x={tuple(x.shape)} {x.dtype}, y={tuple(y.shape)}, "
                          f"labels {int(y.min())}-{int(y.max())}, "
                          f"pixel mean={x.mean():.3f} std={x.std():.3f}")
                if n == 10:
                    break
            print(f"{split:<5} speed    : {(time.perf_counter() - start) / n:.3f} s/batch "
                  f"({cfg.data.batch_size} images, {cfg.data.num_workers} workers)")
        return 0

    raise ValueError(args.action)


def _load_pipelines() -> None:
    import importlib

    for module in ("src.pipelines.classification",):
        importlib.import_module(module)


def dispatch(args: argparse.Namespace) -> int:
    _load_pipelines()
    handler = HANDLERS.get((args.command, args.task))
    if handler is None:
        print(f"'{args.command}' for task '{args.task}' is not implemented yet.", file=sys.stderr)
        return 2
    return handler(args)


def _add_common(p: argparse.ArgumentParser, needs_task: bool = True) -> None:
    if needs_task:
        p.add_argument("--task", choices=TASKS, default="classification")
    p.add_argument("--config", help="Path to a YAML config in configs/")
    p.add_argument("--set", nargs="*", default=[], metavar="SECTION.KEY=VALUE",
                   help="Override config values, e.g. --set train.lr=3e-4 model.dropout=0.5")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli",
                                     description="From Pixels to Language - project CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="Show environment, device and resolved config")
    _add_common(p, needs_task=False)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("data", help="Dataset utilities (inspect, split, eda, augment-preview, loader-check)")
    _add_common(p, needs_task=False)
    p.add_argument("--action", choices=("inspect", "split", "eda", "augment-preview", "loader-check"), required=True)
    p.add_argument("--force", action="store_true", help="Overwrite an existing split")
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("train", help="Train a model for a task")
    _add_common(p)
    p.add_argument("--resume", help="Checkpoint to resume from")
    p.set_defaults(func=dispatch)

    p = sub.add_parser("evaluate", help="Evaluate a checkpoint on the val or test split")
    _add_common(p)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--no-analysis", action="store_true", help="Skip per-class / confusion analysis")
    p.set_defaults(func=dispatch)

    p = sub.add_parser("predict", help="Run inference on one image or a folder")
    _add_common(p)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True, help="Image file or directory")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--output", help="Where to save the prediction/visualization")
    p.set_defaults(func=dispatch)

    p = sub.add_parser("tune", help="Hyperparameter search (grid | random) or compare searches")
    _add_common(p)
    p.add_argument("--compare", nargs="+", metavar="SEARCH_DIR",
                   help="Compare finished searches, e.g. outputs/tuning/tune-grid outputs/tuning/tune-random")
    p.set_defaults(func=dispatch)

    p = sub.add_parser("gradcam", help="Grad-CAM visualizations (correct vs wrong, model vs model)")
    _add_common(p)
    p.add_argument("--checkpoint", required=True, nargs="+",
                   help="One or more best.pt files; with several, the same images are shown for each model")
    p.add_argument("--image", help="Single image; omit to sample correct and wrong predictions")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--num-samples", type=int, default=6, help="Images per figure (correct / wrong)")
    p.add_argument("--output", default="outputs/gradcam")
    p.set_defaults(func=dispatch)

    p = sub.add_parser("compare", help="Compare evaluated runs side by side (table + bar chart)")
    _add_common(p)
    p.add_argument("--runs", required=True, nargs="+", help="Run folders, e.g. outputs/runs/cls-scratch")
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--output", default="outputs")
    p.set_defaults(func=dispatch)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
