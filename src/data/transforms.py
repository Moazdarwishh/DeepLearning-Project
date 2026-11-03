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
    """Undo Normalize for plotting."""
    import torch

    shape = (-1, 1, 1) if tensor.dim() == 3 else (1, -1, 1, 1)
    m = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(shape)
    s = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(shape)
    return (tensor * s + m).clamp(0, 1)
