"""Figure showing what the training augmentation does to real CUB images."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def augmentation_grid(samples, train_tf, eval_tf, mean, std, out_path: Path,
                      n_aug: int = 6, seed: int = 0) -> Path:
    """One row per image: original | eval view (resize + center crop) | n_aug random train views.

    samples: list of (PIL image, title).
    """
    import torch

    from src.data.transforms import denormalize

    torch.manual_seed(seed)
    cols = 2 + n_aug
    fig, axes = plt.subplots(len(samples), cols, figsize=(2.2 * cols, 2.75 * len(samples)))
    for r, (img, title) in enumerate(samples):
        row = axes[r] if len(samples) > 1 else axes
        row[0].imshow(img)
        row[0].set_title(f"{title}\noriginal {img.size[0]}x{img.size[1]}", fontsize=8)
        views = [("eval: resize+center crop", eval_tf(img))]
        views += [(f"train aug #{k + 1}", train_tf(img)) for k in range(n_aug)]
        for c, (name, tensor) in enumerate(views, start=1):
            row[c].imshow(denormalize(tensor, mean, std).permute(1, 2, 0).numpy())
            row[c].set_title(name, fontsize=8)
        for ax in row:
            ax.axis("off")
    fig.suptitle("Training augmentation: rotation +/-10 deg, random crop (50-100 %), flip, "
                 "brightness/contrast/saturation jitter - hue unchanged", fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path
