"""Generic, task-agnostic training loop.

The Trainer receives everything it needs (model, task strategy, optimizer,
scheduler, tracker) from outside - it contains no classification-specific code,
so Milestone 2 (colorization, captioning) reuses it unchanged.

Features: per-batch LR scheduling, mixed precision on CUDA, gradient clipping,
early stopping on a monitored validation metric, best/last checkpoints, resume,
per-epoch history (JSON) and experiment-tracker logging.
"""
from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from .tasks import Task


@torch.no_grad()
def evaluate_model(model, task: Task, loader, device, limit: int | None = None, prefix: str = "val",
                   use_amp: bool = False):
    """Run the model over a loader -> (metrics with prefix, outputs, targets) on CPU."""
    model.eval()
    total_loss = seen = 0
    outs, tgts = [], []
    amp = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
    for i, batch in enumerate(loader):
        if limit is not None and i >= limit:
            break
        with amp:
            loss, outputs, targets = task.forward(model, batch, device)
        n = targets.shape[0]
        total_loss += loss.item() * n
        seen += n
        outs.append(outputs.float().cpu())
        tgts.append(targets.cpu())
    outputs, targets = torch.cat(outs), torch.cat(tgts)
    metrics = {f"{prefix}/loss": total_loss / max(seen, 1)}
    metrics.update({f"{prefix}/{k}": v for k, v in task.metrics(outputs, targets).items()})
    return metrics, outputs, targets


class EarlyStopping:
    """Tracks the monitored metric. Ties are broken by a secondary value (lower is better,
    e.g. validation loss), so between two epochs with equal accuracy the better-calibrated
    one is kept."""

    def __init__(self, patience: int, mode: str = "max"):
        self.patience, self.mode = patience, mode
        self.best = -math.inf if mode == "max" else math.inf
        self.best_tiebreak = math.inf
        self.bad_epochs = 0

    def improved(self, value: float, tiebreak: float | None = None) -> bool:
        better = value > self.best if self.mode == "max" else value < self.best
        if not better and value == self.best and tiebreak is not None:
            better = tiebreak < self.best_tiebreak
        if better:
            self.best, self.bad_epochs = value, 0
            self.best_tiebreak = tiebreak if tiebreak is not None else math.inf
        else:
            self.bad_epochs += 1
        return better

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.bad_epochs >= self.patience


class Trainer:
    def __init__(self, model, task: Task, optimizer, scheduler, device, tracker, train_cfg,
                 output_dir: str | Path, run_info: dict[str, Any] | None = None):
        self.model = model.to(device)
        self.task = task
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.tracker = tracker
        self.cfg = train_cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_info = run_info or {}          # config, class names... stored in checkpoints
        self.history: list[dict[str, float]] = []
        self.start_epoch = 0
        self.stopper = EarlyStopping(train_cfg.early_stopping_patience, train_cfg.monitor_mode)
        self.use_amp = bool(train_cfg.amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    # ------------------------------------------------------------------ steps
    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.float16) if self.use_amp else nullcontext()

    def train_one_epoch(self, loader) -> dict[str, float]:
        self.model.train()
        total_loss = correct = seen = 0
        limit = self.cfg.limit_train_batches
        for i, batch in enumerate(loader):
            if limit is not None and i >= limit:
                break
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                loss, outputs, targets = self.task.forward(self.model, batch, self.device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss became {loss.item()} - try a lower learning rate.")
            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            n = targets.shape[0]
            total_loss += loss.item() * n
            seen += n
            if outputs.dim() == 2 and targets.dim() == 1:          # cheap running accuracy
                correct += int((outputs.argmax(1) == targets).sum())
        out = {"train/loss": total_loss / max(seen, 1)}
        if correct:
            out["train/accuracy"] = correct / seen
        return out

    def evaluate(self, loader, limit: int | None = None, prefix: str = "val"):
        """Returns (metrics with prefix, all outputs, all targets) - outputs kept on CPU."""
        return evaluate_model(self.model, self.task, loader, self.device, limit, prefix, self.use_amp)

    # ------------------------------------------------------------------- loop
    def fit(self, train_loader, val_loader) -> dict[str, Any]:
        best_epoch = None
        for epoch in range(self.start_epoch, self.cfg.epochs):
            start = time.perf_counter()
            row = {"epoch": epoch + 1, "lr": self.optimizer.param_groups[-1]["lr"]}
            row.update(self.train_one_epoch(train_loader))
            val_metrics, _, _ = self.evaluate(val_loader, self.cfg.limit_val_batches)
            row.update(val_metrics)
            row["epoch_time_s"] = time.perf_counter() - start
            self.history.append(row)
            self.tracker.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch + 1)

            score = row[self.cfg.monitor]
            tiebreak = row.get("val/loss") if self.cfg.monitor != "val/loss" else None
            if self.stopper.improved(score, tiebreak):
                best_epoch = epoch + 1
                self.save_checkpoint(self.output_dir / "best.pt", epoch + 1)
            if self.cfg.save_every_epoch:
                self.save_checkpoint(self.output_dir / "last.pt", epoch + 1)
            self._write_history()
            print(self._format(row, improved=best_epoch == epoch + 1), flush=True)

            if self.stopper.should_stop:
                print(f"Early stopping: no improvement in {self.cfg.monitor} for "
                      f"{self.stopper.patience} epochs.", flush=True)
                break
        if best_epoch is None and self.history:      # resumed run that never improved
            best_epoch = max(self.history, key=lambda r: r[self.cfg.monitor])["epoch"]
        return {"best_epoch": best_epoch, "best_score": self.stopper.best,
                "monitor": self.cfg.monitor, "epochs_run": len(self.history)}

    # ------------------------------------------------------------ persistence
    def save_checkpoint(self, path: Path, epoch: int) -> None:
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_score": self.stopper.best,
            "history": self.history,
            **self.run_info,
        }, path)

    def resume(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.history = ckpt.get("history", [])
        self.start_epoch = ckpt["epoch"]
        self.stopper.best = ckpt.get("best_score", self.stopper.best)
        print(f"Resumed from {path} at epoch {self.start_epoch}")

    def _write_history(self) -> None:
        (self.output_dir / "history.json").write_text(json.dumps(self.history, indent=1))

    def _format(self, row: dict, improved: bool) -> str:
        keys = ["train/loss", "train/accuracy", "val/loss", "val/accuracy", "val/top5_accuracy", "val/macro_f1"]
        parts = [f"epoch {row['epoch']:>3}/{self.cfg.epochs}"]
        parts += [f"{k.replace('val/', 'val_').replace('train/', 'tr_')}={row[k]:.4f}" for k in keys if k in row]
        parts.append(f"lr={row['lr']:.2e}")
        parts.append(f"{row['epoch_time_s']:.0f}s")
        return "  ".join(parts) + ("  *best*" if improved else "")
