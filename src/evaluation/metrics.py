from __future__ import annotations

import numpy as np


def topk_correct(logits, targets, k: int) -> int:
    k = min(k, logits.shape[1])
    topk = logits.topk(k, dim=1).indices
    return int((topk == targets.view(-1, 1)).any(dim=1).sum())


def classification_metrics(logits, targets, num_classes: int | None = None) -> dict[str, float]:
    """Accuracy, top-5 accuracy and macro precision/recall/F1 from logits."""
    from sklearn.metrics import precision_recall_fscore_support

    preds = logits.argmax(dim=1).cpu().numpy()
    y = targets.cpu().numpy()
    labels = np.arange(num_classes or logits.shape[1])
    p, r, f1, _ = precision_recall_fscore_support(y, preds, labels=labels, average="macro", zero_division=0)
    n = len(y)
    return {
        "accuracy": float((preds == y).mean()),
        "top5_accuracy": topk_correct(logits, targets, 5) / n,
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
    }
