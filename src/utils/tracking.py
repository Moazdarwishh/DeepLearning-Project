"""Experiment-tracking abstraction.

The trainer only talks to the ``Tracker`` interface, so switching between
MLflow, Weights & Biases or no tracking is a one-line config change
(Dependency Inversion principle).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Tracker(ABC):
    """Minimal interface every tracking backend implements."""

    @abstractmethod
    def start_run(self, run_name: str, config: dict[str, Any], tags: list[str] | None = None) -> None: ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...

    @abstractmethod
    def log_artifact(self, path: str | Path) -> None: ...

    @abstractmethod
    def end_run(self) -> None: ...

    # Allow ``with build_tracker(cfg) as t:`` usage
    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, *exc) -> None:
        self.end_run()


class NoOpTracker(Tracker):
    """Prints metrics; used for quick debugging or when tracking is disabled."""

    def start_run(self, run_name, config, tags=None):
        print(f"[tracking:none] run '{run_name}' started")

    def log_metrics(self, metrics, step=None):
        shown = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"[tracking:none] step={step} {shown}")

    def log_artifact(self, path):
        print(f"[tracking:none] artifact {path}")

    def end_run(self):
        pass


class MLflowTracker(Tracker):
    def __init__(self, project: str, tracking_uri: str | None = None):
        import mlflow
        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(project)

    def start_run(self, run_name, config, tags=None):
        self._mlflow.start_run(run_name=run_name)
        self._mlflow.log_params({k: str(v) for k, v in config.items()})
        if tags:
            self._mlflow.set_tags({t: "true" for t in tags})

    def log_metrics(self, metrics, step=None):
        # MLflow metric names cannot contain '/' on some backends -> keep as-is, it is supported locally
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path):
        self._mlflow.log_artifact(str(path))

    def end_run(self):
        if self._mlflow.active_run():
            self._mlflow.end_run()


class WandbTracker(Tracker):
    def __init__(self, project: str, entity: str | None = None):
        import wandb
        self._wandb = wandb
        self._project, self._entity = project, entity
        self._run = None

    def start_run(self, run_name, config, tags=None):
        self._run = self._wandb.init(project=self._project, entity=self._entity,
                                     name=run_name, config=config, tags=tags, reinit=True)

    def log_metrics(self, metrics, step=None):
        self._wandb.log(metrics, step=step)

    def log_artifact(self, path):
        self._wandb.save(str(path), policy="now")

    def end_run(self):
        if self._run is not None:
            self._run.finish()
            self._run = None


def build_tracker(backend: str, project: str, tracking_uri: str | None = None,
                  entity: str | None = None) -> Tracker:
    backend = backend.lower()
    if backend == "mlflow":
        return MLflowTracker(project, tracking_uri)
    if backend == "wandb":
        return WandbTracker(project, entity)
    if backend in {"none", "noop", ""}:
        return NoOpTracker()
    raise ValueError(f"Unknown tracking backend '{backend}' (use mlflow | wandb | none)")
