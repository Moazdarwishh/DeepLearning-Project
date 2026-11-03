from __future__ import annotations

from .base import Classifier
from .registry import build_encoder


def build_classifier(model_cfg) -> Classifier:
    encoder = build_encoder(model_cfg.name, pretrained=model_cfg.pretrained, **model_cfg.params)
    if model_cfg.freeze_backbone:
        encoder.freeze()
    return Classifier(encoder, num_classes=model_cfg.num_classes, dropout=model_cfg.dropout)


def count_parameters(model) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
