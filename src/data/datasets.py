from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


def load_cub(name: str, cache_dir: str | None = None):
    """Return the single split of an HF CUB repo as a ``datasets.Dataset``."""
    from datasets import DatasetDict, load_dataset

    ds = load_dataset(name, cache_dir=cache_dir)
    if isinstance(ds, DatasetDict):
        if len(ds) == 1:
            return next(iter(ds.values()))
        for preferred in ("train", "test", "validation"):
            if preferred in ds:
                return ds[preferred]
    return ds


def get_labels(ds) -> list[int]:
    """All labels without decoding images (fast column access)."""
    return [int(x) for x in ds["label"]]


def _pretty(raw: str) -> str:
    name = raw.split(".", 1)[-1] if raw[:3].isdigit() else raw
    return name.replace("_", " ").strip()


def get_class_names(ds, num_classes: int = 200) -> list[str]:
    """Class names from a ClassLabel feature, else from ``file_name`` folders, else 'class_i'."""
    feature = ds.features.get("label")
    names = getattr(feature, "names", None)
    if names and len(names) >= num_classes:
        return [_pretty(n) for n in names]

    if "file_name" in ds.column_names:
        found: dict[int, str] = {}
        for label, fname in zip(ds["label"], ds["file_name"]):
            parent = Path(str(fname)).parent.name
            if parent and int(label) not in found:
                found[int(label)] = _pretty(parent)
        if len(found) >= num_classes:
            return [found[i] for i in range(num_classes)]

    return [f"class_{i}" for i in range(num_classes)]


def split_captions(description: str) -> list[str]:
    """The ``description`` field holds 10 captions separated by newlines."""
    return [c.strip() for c in str(description).split("\n") if c.strip()]


def summarize(ds, sample_images: int = 200) -> dict[str, Any]:
    """Quick structural summary used by ``python -m src.cli data --action inspect``."""
    labels = get_labels(ds)
    counts = Counter(labels)
    n = len(ds)
    caps_per_image = Counter(len(split_captions(d)) for d in ds["description"]) \
        if "description" in ds.column_names else {}

    modes: Counter = Counter()
    sizes: list[tuple[int, int]] = []
    for i in range(min(sample_images, n)):
        img = ds[i]["image"]
        modes[img.mode] += 1
        sizes.append(img.size)

    first = ds[0]
    return {
        "num_rows": n,
        "columns": ds.column_names,
        "features": {k: str(v) for k, v in ds.features.items()},
        "num_classes": len(counts),
        "label_range": [min(labels), max(labels)],
        "images_per_class": {"min": min(counts.values()), "max": max(counts.values()),
                             "mean": round(n / len(counts), 2)},
        "captions_per_image": dict(sorted(caps_per_image.items())),
        f"image_modes_first_{len(sizes)}": dict(modes),
        f"image_size_range_first_{len(sizes)}": {
            "width": [min(s[0] for s in sizes), max(s[0] for s in sizes)],
            "height": [min(s[1] for s in sizes), max(s[1] for s in sizes)],
        },
        "example": {
            "id": str(first.get("id")),
            "label": int(first["label"]),
            "file_name": str(first.get("file_name")),
            "description_first_2": split_captions(first.get("description", ""))[:2],
        },
    }
