"""Command-line interface for every task.

Examples::

    python -m src.cli info     --config configs/cls_resnet50.yaml
    python -m src.cli data     --action inspect --config configs/base.yaml
    python -m src.cli data     --action split   --config configs/base.yaml
    python -m src.cli train    --task classification --config configs/cls_resnet50.yaml
    python -m src.cli train    --task classification --config configs/cls_scratch.yaml --set train.lr=3e-4
    python -m src.cli evaluate --task classification --checkpoint outputs/cls_resnet50/best.pt
    python -m src.cli predict  --task classification --checkpoint outputs/cls_resnet50/best.pt --image bird.jpg
    python -m src.cli tune     --task classification --config configs/tune_grid.yaml
    python -m src.cli gradcam  --task classification --checkpoint outputs/cls_resnet50/best.pt --image bird.jpg

Commands are dispatched through ``HANDLERS`` so each task registers its own
functions without editing the parser (heavy imports such as torch stay lazy).
"""
from __future__ import annotations

import argparse
import platform
import sys
from typing import Callable

from src.utils.config import load_config

TASKS = ("classification", "colorization", "captioning")

# (command, task) -> handler(args). Filled in as each milestone is implemented.
HANDLERS: dict[tuple[str, str], Callable[[argparse.Namespace], int]] = {}


def register(command: str, task: str):
    def decorator(fn):
        HANDLERS[(command, task)] = fn
        return fn
    return decorator


# --------------------------------------------------------------------------- #
# Built-in commands
# --------------------------------------------------------------------------- #
def cmd_info(args: argparse.Namespace) -> int:
    """Print environment + resolved config: a quick check that setup works."""
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
    """Dataset utilities: inspect the HF repos, create the train/val split."""
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

    raise ValueError(args.action)


def dispatch(args: argparse.Namespace) -> int:
    handler = HANDLERS.get((args.command, args.task))
    if handler is None:
        print(f"'{args.command}' for task '{args.task}' is not implemented yet.", file=sys.stderr)
        return 2
    return handler(args)


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
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

    p = sub.add_parser("data", help="Dataset utilities (inspect, split)")
    _add_common(p, needs_task=False)
    p.add_argument("--action", choices=("inspect", "split"), required=True)
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
    p.set_defaults(func=dispatch)

    p = sub.add_parser("predict", help="Run inference on one image or a folder")
    _add_common(p)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True, help="Image file or directory")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--output", help="Where to save the prediction/visualization")
    p.set_defaults(func=dispatch)

    p = sub.add_parser("tune", help="Hyperparameter search (grid | random | optuna)")
    _add_common(p)
    p.set_defaults(func=dispatch)

    p = sub.add_parser("gradcam", help="Grad-CAM visualizations")
    _add_common(p)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", help="Single image; omit to sample correct/wrong val predictions")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--output", default="outputs/gradcam")
    p.set_defaults(func=dispatch)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
