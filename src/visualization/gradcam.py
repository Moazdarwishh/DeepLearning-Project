from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model.eval()
        layer = target_layer if target_layer is not None else model.encoder.last_conv_layer()
        self.activations = None
        self.gradients = None
        self._hooks = [layer.register_forward_hook(self._save_activation)]

    def _save_activation(self, module, inputs, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()

    def __call__(self, x: torch.Tensor, class_idx: torch.Tensor | int | None = None):
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x)
            if class_idx is None:
                idx = logits.argmax(dim=1)
            elif isinstance(class_idx, int):
                idx = torch.full((x.shape[0],), class_idx, device=x.device)
            else:
                idx = class_idx.to(x.device)
            self.model.zero_grad(set_to_none=True)
            logits.gather(1, idx.view(-1, 1)).sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        lo = cam.amin(dim=(1, 2), keepdim=True)
        hi = cam.amax(dim=(1, 2), keepdim=True)
        cam = (cam - lo) / (hi - lo + 1e-8)
        return cam.detach(), logits.detach(), idx.detach()


def overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a Grad-CAM heatmap over an image."""
    import matplotlib

    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return np.clip((1 - alpha) * image + alpha * heat, 0, 1)


def gradcam_grid(items: list[dict], model_names: list[str], path: Path, title: str) -> Path:
    """One row per image: original image followed by one Grad-CAM overlay per model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 1 + len(model_names)
    fig, axes = plt.subplots(len(items), cols, figsize=(3.1 * cols, 3.25 * len(items)), layout="constrained")
    axes = np.atleast_2d(axes)
    for row, it in zip(axes, items):
        row[0].imshow(it["image"])
        row[0].set_title(it["label"], fontsize=8)
        for ax, cam, cap in zip(row[1:], it["cams"], it["captions"]):
            ax.imshow(overlay(it["image"], cam))
            ax.set_title(cap, fontsize=8)
        for ax in row:
            ax.axis("off")
    for ax, name in zip(axes[0][1:], model_names):
        ax.set_title(f"[{name}]\n" + ax.get_title(), fontsize=8)
    fig.suptitle(title, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=105, bbox_inches="tight")
    plt.close(fig)
    return path
