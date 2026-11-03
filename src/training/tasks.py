"""Task strategies: everything task-specific the generic Trainer needs.

The Trainer only knows "run batches, call task.loss, collect outputs, ask the task
for metrics". Classification, colorization and captioning each implement this
small interface, so the training loop is written once (Open/Closed + Dependency
Inversion principles).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Task(ABC):
    @abstractmethod
    def forward(self, model: nn.Module, batch, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (loss, outputs, targets) for one batch."""

    @abstractmethod
    def metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """Metrics over a whole epoch (outputs/targets concatenated on CPU)."""


class ClassificationTask(Task):
    def __init__(self, num_classes: int, label_smoothing: float = 0.0):
        self.num_classes = num_classes
        # Label smoothing: the target is 0.9 for the true class instead of 1.0, which
        # discourages over-confident predictions - helpful with only ~25 images/class.
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, model, batch, device):
        images, labels = batch
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        return self.criterion(logits, labels), logits, labels

    def metrics(self, outputs, targets):
        from src.evaluation.metrics import classification_metrics
        return classification_metrics(outputs, targets, self.num_classes)
