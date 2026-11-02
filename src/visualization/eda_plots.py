"""Matplotlib figures for the EDA (saved as PNG into outputs/eda/figures)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "font.size": 10})
BLUE, ORANGE, GREEN, RED, GREY = "#3b6ea8", "#e08a2e", "#4a9a5b", "#c0504d", "#8c8c8c"
SPLIT_COLORS = {"train": BLUE, "val": ORANGE, "test": GREEN}


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def class_distribution(counts_by_split: dict[str, np.ndarray], out_dir: Path) -> Path:
    fig, axes = plt.subplots(len(counts_by_split), 1, figsize=(14, 2.6 * len(counts_by_split)),
                             sharex=True)
    for ax, (split, counts) in zip(np.atleast_1d(axes), counts_by_split.items()):
        ax.bar(np.arange(len(counts)), counts, color=SPLIT_COLORS.get(split, GREY), width=0.9)
        ax.axhline(counts.mean(), color="black", lw=1, ls="--", label=f"mean = {counts.mean():.1f}")
        ax.set_title(f"{split}: {counts.sum()} images, {counts.min()}–{counts.max()} per class")
        ax.set_ylabel("images")
        ax.legend(loc="lower right", frameon=False)
    np.atleast_1d(axes)[-1].set_xlabel("class index (0–199)")
    return _save(fig, out_dir, "01_class_distribution.png")


def image_sizes(df, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    for split, sub in df.groupby("split"):
        c = SPLIT_COLORS.get(split, GREY)
        axes[0].hist(sub["width"], bins=40, alpha=0.6, color=c, label=split)
        axes[1].hist(sub["height"], bins=40, alpha=0.6, color=c, label=split)
        axes[2].hist(sub["aspect"], bins=40, alpha=0.6, color=c, label=split)
        axes[3].scatter(sub["width"], sub["height"], s=3, alpha=0.3, color=c, label=split)
    for ax, t in zip(axes, ["width (px)", "height (px)", "aspect ratio (w/h)", "width vs height"]):
        ax.set_title(t)
    axes[0].legend(frameon=False)
    axes[3].set_xlabel("width"); axes[3].set_ylabel("height")
    return _save(fig, out_dir, "02_image_sizes.png")


def quality(df, out_dir: Path, thresholds: dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    cols = [("brightness", "brightness (mean gray, 0–1)"), ("contrast", "contrast (std gray)"),
            ("blur_laplacian_var", "sharpness (Laplacian variance, log)")]
    for ax, (col, title) in zip(axes, cols):
        data = df[col].dropna()
        if col == "blur_laplacian_var":
            data = np.log10(data + 1)
        ax.hist(data, bins=50, color=BLUE)
        ax.set_title(title)
    axes[0].axvline(thresholds["dark"], color=RED, ls="--"); axes[0].axvline(thresholds["bright"], color=RED, ls="--")
    axes[1].axvline(thresholds["low_contrast"], color=RED, ls="--")
    axes[2].axvline(np.log10(thresholds["blurry"] + 1), color=RED, ls="--", label="flag threshold")
    axes[2].legend(frameon=False)
    return _save(fig, out_dir, "03_image_quality.png")


def image_grid(images, titles, out_dir: Path, name: str, suptitle: str, ncols: int = 6) -> Path | None:
    if not images:
        return None
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.8 * nrows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, img, title in zip(np.atleast_1d(axes).ravel(), images, titles):
        ax.imshow(img, cmap="gray" if getattr(img, "mode", "") == "L" else None)
        ax.set_title(title, fontsize=8)
    fig.suptitle(suptitle, fontweight="bold")
    return _save(fig, out_dir, name)


def pair_grid(pairs, out_dir: Path, name: str, suptitle: str) -> Path | None:
    """pairs: list of (img_a, title_a, img_b, title_b)."""
    if not pairs:
        return None
    fig, axes = plt.subplots(len(pairs), 2, figsize=(6, 3.3 * len(pairs)))
    axes = np.atleast_2d(axes)
    for row, (a, ta, b, tb) in zip(axes, pairs):
        for ax, img, t in ((row[0], a, ta), (row[1], b, tb)):
            ax.imshow(img); ax.set_title(t, fontsize=8); ax.axis("off")
    fig.suptitle(suptitle, fontweight="bold")
    return _save(fig, out_dir, name)


def caption_lengths(lengths_by_split: dict[str, list[int]], p95: float, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    max_len = max(max(v) for v in lengths_by_split.values())
    bins = np.arange(0, max_len + 2) - 0.5
    for split, lengths in lengths_by_split.items():
        ax.hist(lengths, bins=bins, alpha=0.6, color=SPLIT_COLORS.get(split, GREY), label=split, density=True)
    ax.axvline(p95, color=RED, ls="--", label=f"95th percentile = {p95:.0f} words")
    ax.set_xlabel("caption length (words)"); ax.set_ylabel("fraction of captions")
    ax.set_title("Caption length distribution"); ax.legend(frameon=False)
    return _save(fig, out_dir, "08_caption_lengths.png")


def top_words(top: list[tuple[str, int]], out_dir: Path) -> Path:
    words, counts = zip(*top[::-1])
    fig, ax = plt.subplots(figsize=(7, 0.25 * len(words) + 1))
    ax.barh(words, counts, color=BLUE)
    ax.set_title("Most frequent content words (stopwords removed)"); ax.set_xlabel("count")
    return _save(fig, out_dir, "09_top_words.png")


def color_words(counts: dict[str, int], out_dir: Path) -> Path:
    palette = {"black": "#222222", "white": "#dddddd", "brown": "#8b5a2b", "grey": "#888888",
               "gray": "#999999", "yellow": "#e8c547", "red": "#c0392b", "blue": "#2e6fbf",
               "orange": "#e67e22", "green": "#3f8f4f", "tan": "#d2b48c", "buff": "#f0dc82",
               "olive": "#808000", "pink": "#e89ab0", "purple": "#7d4b9b", "rufous": "#a8441f",
               "iridescent": "#4cc3c9", "cream": "#f3e5ab"}
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar([k for k, _ in items], [v for _, v in items],
           color=[palette.get(k, GREY) for k, _ in items], edgecolor="black", lw=0.4)
    ax.set_title("Colour words in captions (colour is central to both colorization and captioning)")
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, out_dir, "10_color_words.png")
