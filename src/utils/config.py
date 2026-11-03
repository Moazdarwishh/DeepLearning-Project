"""Typed experiment configuration loaded from YAML.

Every run is fully described by one YAML file (see ``configs/``). Values can be
overridden from the CLI with ``--set section.key=value`` so that tuning scripts
and teammates never need to edit code.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    name: str = "default"
    task: str = "classification"          # classification | colorization | captioning
    seed: int = 42
    output_dir: str = "outputs"
    tags: list[str] = field(default_factory=list)


@dataclass
class DataConfig:
    train_dataset: str = "Multimodal-Fatima/CUB_train"
    test_dataset: str = "Multimodal-Fatima/CUB_test"
    cache_dir: str | None = None
    splits_dir: str = "data/splits"
    val_fraction: float = 0.15              # stratified split carved from train
    image_size: int = 224
    resize_size: int = 256                  # eval: resize -> center crop
    normalization: str = "imagenet"         # imagenet | dataset
    # CUB train-split statistics measured in the EDA (5,094 train images only)
    dataset_mean: list[float] = field(default_factory=lambda: [0.4848, 0.4994, 0.4327])
    dataset_std: list[float] = field(default_factory=lambda: [0.2323, 0.2277, 0.2662])
    exclusions_file: str = "data/splits/exclusions.json"   # images dropped after the EDA
    batch_size: int = 32
    num_workers: int = 2

    # --- augmentation (train only; see src/data/transforms.py for the reasoning)
    augment: bool = True
    crop_scale_min: float = 0.5             # RandomResizedCrop keeps 50-100 % of the image
    hflip_prob: float = 0.5
    rotation_degrees: float = 10.0
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.2                 # mild; hue is never changed


@dataclass
class ModelConfig:
    name: str = "resnet50"                  # key in the model registry
    pretrained: bool = True
    num_classes: int = 200
    dropout: float = 0.3
    freeze_backbone: bool = False
    params: dict[str, Any] = field(default_factory=dict)   # model-specific extras


@dataclass
class TrainConfig:
    epochs: int = 30
    optimizer: str = "adamw"                # adamw | sgd
    lr: float = 1e-3
    backbone_lr: float | None = None        # lower LR for pretrained layers
    weight_decay: float = 1e-4
    momentum: float = 0.9
    scheduler: str = "cosine"               # cosine | step | none
    step_size: int = 10                     # "step" scheduler: decay every N epochs
    gamma: float = 0.1                      # "step" scheduler: decay factor
    warmup_epochs: int = 1
    label_smoothing: float = 0.1
    amp: bool = True                        # mixed precision on CUDA
    grad_clip: float | None = None
    early_stopping_patience: int = 10
    monitor: str = "val/accuracy"
    monitor_mode: str = "max"               # max | min
    save_every_epoch: bool = False          # also keep last.pt each epoch (for --resume)
    # quick smoke runs on a laptop: only use the first N batches per epoch (None = all)
    limit_train_batches: int | None = None
    limit_val_batches: int | None = None


@dataclass
class TrackingConfig:
    backend: str = "mlflow"                 # mlflow | wandb | none
    project: str = "pixels-to-language"
    # mlflow only. MLflow 3.x stores runs in a SQLite file; view with:
    #   mlflow ui --backend-store-uri sqlite:///mlflow.db
    tracking_uri: str | None = "sqlite:///mlflow.db"
    entity: str | None = None               # wandb team/user


@dataclass
class TuningConfig:
    method: str = "grid"                    # grid | random | optuna
    n_trials: int = 12                      # used by random / optuna
    epochs_per_trial: int = 8
    search_space: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def flat(self) -> dict[str, Any]:
        """Flatten to ``{"train.lr": 1e-3, ...}`` for experiment trackers."""
        out: dict[str, Any] = {}
        for section, values in self.to_dict().items():
            for key, value in values.items():
                out[f"{section}.{key}"] = value
        return out

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def _build_section(cls, values: dict[str, Any] | None):
    values = values or {}
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown keys for '{cls.__name__}': {sorted(unknown)}")
    return cls(**values)


def _parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def apply_overrides(cfg: Config, overrides: list[str] | None) -> Config:
    """Apply ``section.key=value`` overrides in place and return the config."""
    for item in overrides or []:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"Override must look like section.key=value, got '{item}'")
        dotted, raw = item.split("=", 1)
        section_name, key = dotted.split(".", 1)
        section = getattr(cfg, section_name, None)
        if section is None or not is_dataclass(section):
            raise ValueError(f"Unknown config section '{section_name}'")
        if not hasattr(section, key):
            raise ValueError(f"Unknown key '{key}' in section '{section_name}'")
        setattr(section, key, _parse_value(raw))
    return cfg


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: str | Path) -> dict[str, Any]:
    """Read YAML, resolving an optional ``_base_: other.yaml`` (relative path) first."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    base_name = raw.pop("_base_", None)
    if base_name:
        raw = _deep_merge(_read_yaml(path.parent / base_name), raw)
    return raw


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a YAML config (missing keys fall back to defaults) and apply overrides."""
    raw: dict[str, Any] = _read_yaml(path) if path is not None else {}
    return config_from_dict(raw, overrides)


def config_from_dict(raw: dict[str, Any], overrides: list[str] | None = None) -> Config:
    """Build a Config from a plain dict (e.g. the one saved inside a checkpoint)."""
    raw = dict(raw)
    section_types = {f.name: f.type for f in fields(Config)}
    unknown = set(raw) - set(section_types)
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    cfg = Config(
        experiment=_build_section(ExperimentConfig, raw.get("experiment")),
        data=_build_section(DataConfig, raw.get("data")),
        model=_build_section(ModelConfig, raw.get("model")),
        train=_build_section(TrainConfig, raw.get("train")),
        tracking=_build_section(TrackingConfig, raw.get("tracking")),
        tuning=_build_section(TuningConfig, raw.get("tuning")),
    )
    return apply_overrides(cfg, overrides)
