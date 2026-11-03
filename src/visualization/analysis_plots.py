from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, GREEN, RED, GREY = "#3b6ea8", "#e08a2e", "#4a9a5b", "#c0504d", "#8c8c8c"


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def per_class_figure(per_class: list[dict], path: Path, n: int = 15) -> Path:
    rows = [r for r in per_class if r["support"]]
    ranked = sorted(rows, key=lambda r: r["accuracy"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 0.28 * n + 1.8), gridspec_kw={"width_ratios": [1, 1.2, 1.2]})
    axes[0].hist([r["accuracy"] for r in rows], bins=20, range=(0, 1), color=BLUE)
    axes[0].set_xlabel("per-class accuracy")
    axes[0].set_ylabel("number of species")
    axes[0].set_title("How accuracy is spread over the 200 species")
    for ax, subset, color, title in ((axes[1], ranked[:n][::-1], RED, f"{n} hardest species"),
                                     (axes[2], ranked[-n:], GREEN, f"{n} easiest species")):
        ax.barh([f"{r['class'][:26]} (n={r['support']})" for r in subset], [r["accuracy"] for r in subset], color=color)
        ax.set_xlim(0, 1)
        ax.set_title(title)
        ax.tick_params(axis="y", labelsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def confusion_figure(cm: np.ndarray, class_names: list[str], pairs: list[dict], path: Path, zoom: int = 16) -> Path:
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ids = []
    for p in pairs:
        for c in (p["true_id"], p["pred_id"]):
            if c not in ids:
                ids.append(c)
    ids = sorted(ids[:zoom])
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [1, 1.15]})
    im = axes[0].imshow(norm, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title("Confusion matrix (rows = true species, normalised)")
    axes[0].set_xlabel("predicted class id")
    axes[0].set_ylabel("true class id")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    if ids:
        sub = norm[np.ix_(ids, ids)]
        axes[1].imshow(sub, cmap="Blues", vmin=0, vmax=max(sub.max(), 1e-6))
        labels = [class_names[i][:22] for i in ids]
        axes[1].set_xticks(range(len(ids)), labels, rotation=90, fontsize=8)
        axes[1].set_yticks(range(len(ids)), labels, fontsize=8)
        for (r, c), v in np.ndenumerate(cm[np.ix_(ids, ids)]):
            if v:
                axes[1].text(c, r, str(v), ha="center", va="center", fontsize=7,
                             color="white" if sub[r, c] > 0.5 else "black")
        axes[1].set_title("Zoom: species involved in the most frequent mistakes (counts)")
    else:
        axes[1].axis("off")
    return _save(fig, path)


def _as_image(dataset, i, norm):
    from src.data.transforms import denormalize

    x = dataset[i][0]
    return denormalize(x, *norm).permute(1, 2, 0).numpy()


def confused_pairs_figure(pairs, logits, targets, dataset, reference, class_names, norm, path: Path) -> Path | None:
    """One row per confused pair."""
    if not pairs:
        return None
    preds = logits.argmax(axis=1)
    ref_labels = reference.labels if reference is not None and hasattr(reference, "labels") else None
    fig, axes = plt.subplots(len(pairs), 2, figsize=(6.4, 3.0 * len(pairs)))
    axes = np.atleast_2d(axes)
    for row, p in zip(axes, pairs):
        wrong = np.where((targets == p["true_id"]) & (preds == p["pred_id"]))[0]
        row[0].imshow(_as_image(dataset, int(wrong[0]), norm))
        row[0].set_title(f"TRUE: {p['true'][:24]}\npredicted: {p['predicted'][:24]} ({p['count']}x)", fontsize=8)
        if ref_labels is not None and p["pred_id"] in ref_labels:
            row[1].imshow(_as_image(reference, ref_labels.index(p["pred_id"]), norm))
            row[1].set_title(f"a real {p['predicted'][:24]}\n(training image)", fontsize=8)
        for ax in row:
            ax.axis("off")
    fig.suptitle("Most frequent mistakes: what the model saw vs what it thought", fontweight="bold")
    return _save(fig, path)


def comparison_figure(rows: list[dict], split: str, path: Path) -> Path:
    metrics = [("accuracy", "accuracy"), ("top5_accuracy", "top-5"), ("macro_f1", "macro F1")]
    x = np.arange(len(rows))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(rows)), 4))
    for k, ((key, label), color) in enumerate(zip(metrics, (BLUE, ORANGE, GREEN))):
        vals = [r[key] for r in rows]
        bars = ax.bar(x + (k - 1) * width, vals, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, [r["run"] for r in rows], rotation=15)
    ax.set_ylim(0, 1)
    ax.set_title(f"Model comparison on the {split} set")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)
