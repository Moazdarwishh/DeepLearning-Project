"""Image preprocessing and augmentation for CUB-200-2011 (Milestone 1, Task 1.1).

Augmentation strategy and why (numbers come from the EDA):

* RandomRotation(10 deg), applied first and filled with the mean colour
    Photos are taken at slight angles (branches, handheld cameras). Rotating
    before cropping means the crop usually cuts away the empty corners, so the
    model does not learn "black triangles in the corners" as a pattern.
* RandomResizedCrop(224, scale=(0.5, 1.0))
    Images have varied sizes (120-500 px) and aspect ratios (0.4-3.6) and the
    bird is not always centred. Random crops teach scale/position invariance.
    The minimum scale is 0.5 (not torchvision's 0.08) because birds are small
    in many photos; tiny crops would often cut the bird out entirely.
* RandomHorizontalFlip(p=0.5)
    A bird facing left is the same species facing right.
* ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0)
    The EDA found ~1 % dark, bright and low-contrast images; jitter makes the
    model robust to lighting. Hue is NEVER changed: plumage colour is a key
    feature to tell species apart (red vs. orange vs. yellow), and colour is the
    target of the colorization task in Milestone 2.

Deliberately NOT used:
* Vertical flip / large rotations - birds are never upside down in real photos.
* Random grayscale - would destroy the colour cue (see hue above).

Evaluation (val/test) is deterministic: Resize(256) -> CenterCrop(224).
Normalization uses ImageNet statistics for pretrained models (they were trained
with them) and our own train-split statistics for the from-scratch CNN.
"""
from __future__ import annotations

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalization_stats(data_cfg) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if data_cfg.normalization == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    if data_cfg.normalization == "dataset":
        return tuple(data_cfg.dataset_mean), tuple(data_cfg.dataset_std)
    raise ValueError(f"Unknown normalization '{data_cfg.normalization}' (imagenet | dataset)")


def build_train_transform(data_cfg):
    """Random augmentation pipeline for training images (PIL RGB in -> normalized tensor out)."""
    import torch
    from torchvision.transforms import v2

    mean, std = normalization_stats(data_cfg)
    if not data_cfg.augment:
        return build_eval_transform(data_cfg)
    # Rotate BEFORE cropping so the crop usually removes the empty corners; any gap left
    # is filled with the mean colour (normalizes to ~0) instead of black.
    fill = [int(round(255 * m)) for m in mean]
    return v2.Compose([
        v2.ToImage(),
        v2.RandomRotation(data_cfg.rotation_degrees, fill=fill),
        v2.RandomResizedCrop(data_cfg.image_size, scale=(data_cfg.crop_scale_min, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=data_cfg.hflip_prob),
        v2.ColorJitter(brightness=data_cfg.brightness, contrast=data_cfg.contrast,
                       saturation=data_cfg.saturation, hue=0.0),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])


def build_eval_transform(data_cfg):
    """Deterministic pipeline for validation, test and inference."""
    import torch
    from torchvision.transforms import v2

    mean, std = normalization_stats(data_cfg)
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(data_cfg.resize_size, antialias=True),
        v2.CenterCrop(data_cfg.image_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std),
    ])


def denormalize(tensor, mean, std):
    """Undo Normalize for plotting: (C,H,W) or (B,C,H,W) tensor -> values in [0, 1]."""
    import torch

    shape = (-1, 1, 1) if tensor.dim() == 3 else (1, -1, 1, 1)
    m = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(shape)
    s = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(shape)
    return (tensor * s + m).clamp(0, 1)
