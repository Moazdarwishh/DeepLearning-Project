from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn


class Encoder(nn.Module):
    out_channels: list[int]
    reductions: list[int]

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return feature maps of every stage, shallow to deep."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Global-average-pooled embedding of the deepest stage."""
        return self.forward_features(x)[-1].mean(dim=(2, 3))

    @property
    def feature_dim(self) -> int:
        return self.out_channels[-1]

    frozen: bool = False

    def freeze(self, trainable: bool = False) -> None:
        """Freeze (or unfreeze) all encoder weights."""
        for p in self.parameters():
            p.requires_grad = trainable
        self.frozen = not trainable

    def last_conv_layer(self) -> nn.Module:
        """Layer used as Grad-CAM target."""
        raise NotImplementedError


class Classifier(nn.Module):
    """Encoder + linear head for the 200 bird species."""

    def __init__(self, encoder: Encoder, num_classes: int = 200, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(encoder.feature_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self.encoder, "frozen", False):
            self.encoder.eval()
        return self
