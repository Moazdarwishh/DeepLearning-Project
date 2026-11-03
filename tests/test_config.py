from pathlib import Path

import pytest

from src.utils.config import apply_overrides, load_config

CONFIGS = sorted(Path("configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_every_config_loads(path):
    cfg = load_config(path)
    assert cfg.model.num_classes == 200


def test_overrides_are_typed():
    cfg = load_config("configs/cls_resnet50.yaml", ["train.lr=3e-4", "model.freeze_backbone=true"])
    assert cfg.train.lr == pytest.approx(3e-4)
    assert cfg.model.freeze_backbone is True


def test_unknown_key_rejected():
    cfg = load_config()
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["train.not_a_key=1"])


def test_base_inheritance():
    cfg = load_config("configs/cls_resnet50.yaml")
    assert cfg.data.val_fraction == 0.15
    assert cfg.data.batch_size == 32
    assert cfg.experiment.name == "cls-resnet50-ft"
