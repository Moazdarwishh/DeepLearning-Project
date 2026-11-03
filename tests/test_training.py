"""Trainer, optimizer, scheduler and the classification pipeline on tiny synthetic data."""
import json
from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import Dataset

from src.models import build_classifier, build_encoder, count_parameters
from src.training.optim import build_optimizer, build_scheduler
from src.utils.config import load_config


# ------------------------------------------------------------------ helpers
class PatternDataset(Dataset):
    """Each class = a different bright square position; easy to learn."""

    def __init__(self, n_per_class=12, num_classes=4, size=32, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.items = []
        for c in range(num_classes):
            for _ in range(n_per_class):
                x = torch.randn(3, size, size, generator=g) * 0.3
                r, col = divmod(c, 2)
                x[:, r * 16:(r + 1) * 16, col * 16:(col + 1) * 16] += 2.0
                self.items.append((x, c))
        self.indices = list(range(len(self.items)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


@dataclass
class FakeData:
    train: Dataset
    val: Dataset
    test: Dataset
    class_names: list

    def sizes(self):
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def tiny_cfg(tmp_path, *extra):
    return load_config("configs/cls_scratch.yaml", [
        "model.num_classes=4", "model.dropout=0.0",
        'model.params={"widths": [8, 16, 32, 32], "blocks_per_stage": 1, "residual": True}',
        "data.batch_size=8", "data.num_workers=0",
        "train.epochs=12", "train.lr=3e-3", "train.warmup_epochs=1", "train.early_stopping_patience=0",
        "tracking.backend=none", f"experiment.output_dir={tmp_path}", "experiment.name=tiny", *extra])


# ------------------------------------------------------------------ models
@pytest.mark.parametrize("residual", [True, False])
def test_scratch_cnn_stages(residual):
    enc = build_encoder("scratch_cnn", widths=[16, 32, 64, 128], blocks_per_stage=2, residual=residual)
    feats = enc.forward_features(torch.randn(2, 3, 224, 224))
    assert [f.shape[1] for f in feats] == [16, 32, 64, 128]
    assert [224 // f.shape[-1] for f in feats] == enc.reductions == [4, 8, 16, 32]
    assert enc(torch.randn(2, 3, 224, 224)).shape == (2, 128)


def test_default_scratch_model_size():
    cfg = load_config("configs/cls_scratch.yaml")
    params = count_parameters(build_classifier(cfg.model))
    assert 10e6 < params["total"] < 13e6


def test_freeze_backbone_leaves_only_head_trainable():
    cfg = load_config("configs/cls_scratch.yaml", ["model.freeze_backbone=true"])
    params = count_parameters(build_classifier(cfg.model))
    assert params["trainable"] == 512 * 200 + 200


# --------------------------------------------------------------- optimizer
def test_param_groups_and_backbone_lr(tmp_path):
    cfg = tiny_cfg(tmp_path, "train.backbone_lr=1e-4")
    model = build_classifier(cfg.model)
    opt = build_optimizer(model, cfg.train)
    by_name = {g["name"]: g for g in opt.param_groups}
    assert by_name["encoder.decay"]["lr"] == 1e-4 and by_name["head.decay"]["lr"] == 3e-3
    assert by_name["encoder.no_decay"]["weight_decay"] == 0.0     # biases / BatchNorm


def test_warmup_then_cosine_to_zero(tmp_path):
    cfg = tiny_cfg(tmp_path, "train.epochs=10", "train.warmup_epochs=2")
    opt = build_optimizer(build_classifier(cfg.model), cfg.train)
    sched = build_scheduler(opt, cfg.train, steps_per_epoch=5)
    lrs = []
    for _ in range(50):
        lrs.append(opt.param_groups[-1]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[9]                       # warm-up increases
    assert lrs[9] == pytest.approx(3e-3)         # peak after 2 epochs
    assert lrs[-1] < 1e-4                        # cosine decays to ~0


# ----------------------------------------------------------------- training
def test_training_pipeline_learns_and_saves(tmp_path):
    from src.pipelines.classification import load_model, train
    from src.training.tasks import ClassificationTask
    from src.training.trainer import evaluate_model

    data = FakeData(PatternDataset(seed=0), PatternDataset(4, seed=1), PatternDataset(4, seed=2),
                    ["a", "b", "c", "d"])
    summary = train(tiny_cfg(tmp_path), data=data)
    run = tmp_path / "runs" / "tiny"
    for f in ("best.pt", "config.yaml", "history.json", "curves.png", "summary.json"):
        assert (run / f).exists(), f
    assert summary["best_val"]["val/accuracy"] >= 0.9           # easy patterns are learned
    history = json.loads((run / "history.json").read_text())
    assert history[-1]["train/loss"] < history[0]["train/loss"]

    model, cfg, ckpt, device = load_model(run / "best.pt")      # checkpoint is self-contained
    assert ckpt["class_names"] == ["a", "b", "c", "d"] and cfg.model.num_classes == 4
    loader = torch.utils.data.DataLoader(data.val, batch_size=8)
    metrics, logits, _ = evaluate_model(model, ClassificationTask(4), loader, device, prefix="val")
    assert metrics["val/accuracy"] == pytest.approx(summary["best_val"]["val/accuracy"])
    assert logits.shape == (16, 4)


def test_early_stopping_and_new_run_dir(tmp_path):
    from src.pipelines.classification import train

    data = FakeData(PatternDataset(seed=0), PatternDataset(4, seed=1), PatternDataset(4, seed=2), list("abcd"))
    # lr=0 -> the model cannot get better -> early stopping ends the run well before 12 epochs
    s = train(tiny_cfg(tmp_path, "train.lr=0", "train.early_stopping_patience=2"), data=data)
    assert s["epochs_run"] < 12
    s2 = train(tiny_cfg(tmp_path, "train.epochs=1"), data=data)  # same name -> new dir
    assert s2["run"].startswith("tiny-")


def test_resume_continues_from_checkpoint(tmp_path):
    from src.pipelines.classification import train

    data = FakeData(PatternDataset(seed=0), PatternDataset(4, seed=1), PatternDataset(4, seed=2), list("abcd"))
    train(tiny_cfg(tmp_path, "train.epochs=3", "train.save_every_epoch=true"), data=data)
    last = tmp_path / "runs" / "tiny" / "last.pt"
    s = train(tiny_cfg(tmp_path, "train.epochs=5", "train.save_every_epoch=true"), resume=str(last), data=data)
    history = json.loads((tmp_path / "runs" / "tiny" / "history.json").read_text())
    assert [r["epoch"] for r in history] == [1, 2, 3, 4, 5] and s["epochs_run"] == 5


def test_mlflow_tracking_records_run(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    from src.pipelines.classification import train

    data = FakeData(PatternDataset(seed=0), PatternDataset(4, seed=1), PatternDataset(4, seed=2), list("abcd"))
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    train(tiny_cfg(tmp_path, "train.epochs=2", "tracking.backend=mlflow",
                   f"tracking.tracking_uri={uri}", "tracking.project=test-exp"), data=data)
    client = mlflow.tracking.MlflowClient(uri)
    run = client.search_runs([client.get_experiment_by_name("test-exp").experiment_id])[0]
    assert "val/accuracy" in run.data.metrics and run.data.params["model.name"] == "scratch_cnn"
    assert len(client.get_metric_history(run.info.run_id, "val/loss")) == 2


def test_cli_train_command_is_registered():
    from src.cli import _load_pipelines
    from src.utils.commands import HANDLERS
    _load_pipelines()
    assert {("train", "classification"), ("evaluate", "classification"),
            ("predict", "classification")} <= set(HANDLERS)


def test_early_stopping_tie_break_on_loss():
    from src.training.trainer import EarlyStopping
    es = EarlyStopping(patience=2, mode="max")
    assert es.improved(0.8, 0.5)
    assert es.improved(0.8, 0.4)          # same accuracy, lower loss -> better
    assert not es.improved(0.8, 0.45)     # same accuracy, higher loss -> not better
    assert not es.improved(0.7, 0.1)
    assert es.should_stop
