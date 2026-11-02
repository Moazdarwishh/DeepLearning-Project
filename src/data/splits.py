"""Stratified train/validation split carved from the official CUB train set.

The split is saved to JSON and committed to git so every teammate and every
run uses exactly the same images for validation. The official test set is
never used for model selection or tuning.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

SPLIT_FILE = "cub_train_val_split.json"


@dataclass
class Split:
    train_indices: list[int]
    val_indices: list[int]
    seed: int
    val_fraction: float
    source_dataset: str
    num_source_rows: int
    created_at: str = ""

    def summary(self, labels: list[int]) -> dict:
        tr = Counter(labels[i] for i in self.train_indices)
        va = Counter(labels[i] for i in self.val_indices)
        return {
            "train_size": len(self.train_indices),
            "val_size": len(self.val_indices),
            "classes_in_train": len(tr),
            "classes_in_val": len(va),
            "val_per_class_min": min(va.values()),
            "val_per_class_max": max(va.values()),
            "train_per_class_min": min(tr.values()),
            "train_per_class_max": max(tr.values()),
        }


def make_stratified_split(labels: list[int], val_fraction: float, seed: int,
                          source_dataset: str = "") -> Split:
    """Every class keeps the same train/val proportion (≈4-5 val images per class)."""
    from sklearn.model_selection import train_test_split

    indices = list(range(len(labels)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_fraction, random_state=seed, stratify=labels, shuffle=True
    )
    return Split(
        train_indices=sorted(train_idx),
        val_indices=sorted(val_idx),
        seed=seed,
        val_fraction=val_fraction,
        source_dataset=source_dataset,
        num_source_rows=len(labels),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def save_split(split: Split, splits_dir: str | Path) -> Path:
    path = Path(splits_dir) / SPLIT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(split), f)
    return path


def load_split(splits_dir: str | Path, expected_rows: int | None = None) -> Split:
    path = Path(splits_dir) / SPLIT_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Create it with: python -m src.cli data --action split"
        )
    with open(path) as f:
        split = Split(**json.load(f))
    if expected_rows is not None and split.num_source_rows != expected_rows:
        raise ValueError(
            f"Split was made for {split.num_source_rows} rows but dataset has {expected_rows}."
        )
    overlap = set(split.train_indices) & set(split.val_indices)
    if overlap:
        raise ValueError(f"Train/val overlap detected ({len(overlap)} indices)")
    return split
