"""Optimizer and learning-rate schedule from the config.

* Two parameter groups: encoder ("backbone") and head. For pretrained models the
  backbone gets a smaller learning rate (train.backbone_lr) so the useful ImageNet
  features are only gently adjusted, while the new head learns fast.
* No weight decay on biases and BatchNorm parameters (standard practice: decaying
  them only hurts).
* Schedulers step every batch: linear warm-up for ``warmup_epochs`` (avoids large,
  unstable updates at the start when weights are random), then cosine decay to 0
  or step decay.
"""
from __future__ import annotations

import math

import torch


def _split_decay(params_with_names):
    decay, no_decay = [], []
    for name, p in params_with_names:
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    return decay, no_decay


def build_optimizer(model, train_cfg) -> torch.optim.Optimizer:
    head_lr = train_cfg.lr
    backbone_lr = train_cfg.backbone_lr if train_cfg.backbone_lr is not None else train_cfg.lr
    groups = []
    for prefix, lr in (("encoder.", backbone_lr), ("head.", head_lr)):
        named = [(n, p) for n, p in model.named_parameters() if n.startswith(prefix)]
        decay, no_decay = _split_decay(named)
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": train_cfg.weight_decay, "name": prefix + "decay"})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0, "name": prefix + "no_decay"})
    other = [(n, p) for n, p in model.named_parameters()
             if not n.startswith(("encoder.", "head.")) and p.requires_grad]
    if other:
        decay, no_decay = _split_decay(other)
        groups += [{"params": decay, "lr": head_lr, "weight_decay": train_cfg.weight_decay, "name": "other.decay"},
                   {"params": no_decay, "lr": head_lr, "weight_decay": 0.0, "name": "other.no_decay"}]
        groups = [g for g in groups if g["params"]]

    name = train_cfg.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(groups)
    if name == "sgd":
        return torch.optim.SGD(groups, momentum=train_cfg.momentum, nesterov=True)
    raise ValueError(f"Unknown optimizer '{train_cfg.optimizer}' (adamw | sgd)")


def build_scheduler(optimizer, train_cfg, steps_per_epoch: int):
    """Per-batch LambdaLR: warm-up then cosine / step / constant."""
    total = max(1, train_cfg.epochs * steps_per_epoch)
    warmup = min(train_cfg.warmup_epochs * steps_per_epoch, total - 1)
    kind = train_cfg.scheduler.lower()

    def factor(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        if kind == "cosine":
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        if kind == "step":
            return train_cfg.gamma ** ((step // steps_per_epoch) // max(1, train_cfg.step_size))
        if kind == "none":
            return 1.0
        raise ValueError(f"Unknown scheduler '{train_cfg.scheduler}' (cosine | step | none)")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
