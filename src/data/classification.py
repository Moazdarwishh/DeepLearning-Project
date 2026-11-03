from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from torch.utils.data import DataLoader, Dataset

from .datasets import get_class_names, load_cub
from .splits import load_split
from .transforms import build_eval_transform, build_train_transform


class CUBClassificationDataset(Dataset):
    """Wraps an HF CUB dataset."""

    def __init__(self, hf_dataset, indices: list[int] | None = None, transform=None):
        cols = [c for c in ("image", "label") if c in hf_dataset.column_names]
        self.data = hf_dataset.select_columns(cols)
        self.indices = list(range(len(hf_dataset))) if indices is None else list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        row = self.data[int(self.indices[i])]
        image = row["image"].convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])

    @property
    def labels(self) -> list[int]:
        all_labels = self.data["label"]
        return [int(all_labels[i]) for i in self.indices]


def load_exclusions(path: str | Path) -> dict[str, list[int]]:
    """Row indices to drop per split (empty if the file is missing)."""
    path = Path(path)
    out = {"train": [], "val": [], "test": []}
    if not path.exists():
        return out
    for item in json.loads(path.read_text()).get("exclude", []):
        out[item["split"]].append(int(item["index"]))
    return out


@dataclass
class ClassificationData:
    train: CUBClassificationDataset
    val: CUBClassificationDataset
    test: CUBClassificationDataset
    class_names: list[str]
    excluded: dict[str, list[int]] = field(default_factory=dict)

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def build_classification_data(cfg, train_hf=None, test_hf=None) -> ClassificationData:
    """Create train/val/test datasets from the config (HF datasets can be injected for tests)."""
    d = cfg.data
    train_hf = train_hf if train_hf is not None else load_cub(d.train_dataset, d.cache_dir)
    test_hf = test_hf if test_hf is not None else load_cub(d.test_dataset, d.cache_dir)
    split = load_split(d.splits_dir, expected_rows=len(train_hf))
    excl = load_exclusions(d.exclusions_file)

    def keep(indices, drop):
        drop = set(drop)
        return [i for i in indices if i not in drop]

    train_idx = keep(split.train_indices, excl["train"])
    val_idx = keep(split.val_indices, excl["val"])
    test_idx = keep(range(len(test_hf)), excl["test"])
    return ClassificationData(
        train=CUBClassificationDataset(train_hf, train_idx, build_train_transform(d)),
        val=CUBClassificationDataset(train_hf, val_idx, build_eval_transform(d)),
        test=CUBClassificationDataset(test_hf, test_idx, build_eval_transform(d)),
        class_names=get_class_names(train_hf, cfg.model.num_classes),
        excluded=excl,
    )


def build_dataloaders(cfg, data: ClassificationData) -> dict[str, DataLoader]:
    """Seeded, reproducible loaders."""
    import torch

    from src.utils.reproducibility import seed_worker

    d = cfg.data
    generator = torch.Generator().manual_seed(cfg.experiment.seed)
    common = dict(
        batch_size=d.batch_size,
        num_workers=d.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=d.num_workers > 0,
        worker_init_fn=seed_worker,
    )
    return {
        "train": DataLoader(data.train, shuffle=True, drop_last=True, generator=generator, **common),
        "val": DataLoader(data.val, shuffle=False, **common),
        "test": DataLoader(data.test, shuffle=False, **common),
    }
