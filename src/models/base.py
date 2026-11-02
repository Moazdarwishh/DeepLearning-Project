"""Shared encoder interface.

Every encoder (from-scratch CNN, ResNet-50, EfficientNet, ...) implements
``forward_features`` returning multi-scale feature maps. That single contract
lets the same encoder be reused in all three tasks:

* classification  -> pooled vector of the last stage  (``forward``)
* colorization    -> all stages as U-Net skip connections
* captioning      -> last-stage spatial grid fed to the text decoder
"""
from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn


class Encoder(nn.Module):
    #: channels of each stage returned by ``forward_features`` (low -> high level)
    out_channels: list[int]
    #: spatial downsampling factor of each stage relative to the input
    reductions: list[int]

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return feature maps of every stage, shallow to deep."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Global-average-pooled embedding of the deepest stage: (B, out_channels[-1])."""
        return self.forward_features(x)[-1].mean(dim=(2, 3))

    @property
    def feature_dim(self) -> int:
        return self.out_channels[-1]

    def freeze(self, trainable: bool = False) -> None:
        for p in self.parameters():
            p.requires_grad = trainable

    def last_conv_layer(self) -> nn.Module:
        """Layer used as Grad-CAM target. Override in concrete encoders."""
        raise NotImplementedError


class Classifier(nn.Module):
    """Encoder + linear head for the 200 bird species."""

    def __init__(self, encoder: Encoder, num_classes: int = 200, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(encoder.feature_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
