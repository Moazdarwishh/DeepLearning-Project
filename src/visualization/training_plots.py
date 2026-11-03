"""Learning curves from a run's history.json."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TRAIN, VAL = "#3b6ea8", "#e08a2e"


def plot_history(history: list[dict], out_path: Path, title: str = "", best_epoch: int | None = None) -> Path:
    epochs = [r["epoch"] for r in history]

    def series(key):
        return [r.get(key) for r in history]

    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    axes[0].plot(epochs, series("train/loss"), color=TRAIN, label="train")
    axes[0].plot(epochs, series("val/loss"), color=VAL, label="val")
    axes[0].set_title("Loss (cross-entropy)")
    if "train/accuracy" in history[0]:
        axes[1].plot(epochs, series("train/accuracy"), color=TRAIN, label="train (augmented)")
    axes[1].plot(epochs, series("val/accuracy"), color=VAL, label="val")
    axes[1].set_title("Top-1 accuracy")
    axes[2].plot(epochs, series("val/top5_accuracy"), color=VAL, ls="--", label="val top-5")
    axes[2].plot(epochs, series("val/macro_f1"), color=VAL, label="val macro F1")
    axes[2].set_title("Validation top-5 / macro F1")
    axes[3].plot(epochs, series("lr"), color="#555555")
    axes[3].set_title("Learning rate (head)")
    if all((v or 0) > 0 for v in series("lr")):
        axes[3].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.spines[["top", "right"]].set_visible(False)
        if best_epoch:
            ax.axvline(best_epoch, color="#999999", ls=":", lw=1)
    for ax in axes[:3]:
        ax.legend(frameon=False, fontsize=8)
    if title:
        fig.suptitle(title, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path
