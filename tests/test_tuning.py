import json
import math

import pytest

from src.training.tuning import (SearchRunner, compare_searches, grid_trials, random_trials, summarize,
                                 validate_space)
from src.utils.config import load_config


def test_project_configs_have_equal_budgets():
    grid = load_config("configs/tune_grid.yaml")
    rnd = load_config("configs/tune_random.yaml")
    assert len(grid_trials(grid.tuning.search_space)) == rnd.tuning.n_trials == 8
    assert grid.tuning.epochs_per_trial == rnd.tuning.epochs_per_trial
    assert set(grid.tuning.search_space) == set(rnd.tuning.search_space)
    assert grid.model.freeze_backbone and rnd.model.freeze_backbone


def test_grid_is_cartesian_product():
    trials = grid_trials({"train.lr": [1e-3, 1e-2], "model.dropout": [0.2, 0.3, 0.5]})
    assert len(trials) == 6
    assert {(t["train.lr"], t["model.dropout"]) for t in trials} == {(a, b) for a in (1e-3, 1e-2) for b in (0.2, 0.3, 0.5)}


def test_random_respects_ranges_and_seed():
    space = {"train.lr": {"dist": "loguniform", "low": 1e-4, "high": 1e-1},
             "model.dropout": {"dist": "uniform", "low": 0.1, "high": 0.6},
             "data.batch_size": [32, 64]}
    a, b = random_trials(space, 50, seed=3), random_trials(space, 50, seed=3)
    assert a == b
    assert all(1e-4 <= t["train.lr"] <= 1e-1 and 0.1 <= t["model.dropout"] <= 0.6 for t in a)
    assert {t["data.batch_size"] for t in a} <= {32, 64}
    logs = [math.log10(t["train.lr"]) for t in a]
    assert min(logs) < -3 and max(logs) > -2
    assert len({t["train.lr"] for t in a}) == 50


@pytest.mark.parametrize("space,method", [
    ({"train.lr": {"dist": "loguniform", "low": 1e-3, "high": 1e-2}}, "grid"),
    ({"lr": [1, 2]}, "random"),
    ({"train.lr": {"dist": "loguniform", "low": 0, "high": 1}}, "random"),
    ({}, "grid"),
])
def test_invalid_spaces_rejected(space, method):
    with pytest.raises(ValueError):
        validate_space(space, method)


def _fake_train(scores_by_lr):
    calls = []

    def train(cfg, data=None):
        calls.append(cfg)
        lr = cfg.train.lr
        if lr > 1:
            raise FloatingPointError("Loss became nan")
        score = scores_by_lr(lr, cfg.model.dropout)
        return {"run": cfg.experiment.name, "best_epoch": cfg.train.epochs,
                "best_val": {"val/accuracy": score, "val/macro_f1": score - 0.01}}
    return train, calls


def _cfg(tmp_path, name, method, space, n=4):
    cfg = load_config("configs/tune_grid.yaml", [f"experiment.output_dir={tmp_path}", f"experiment.name={name}",
                                                 f"tuning.method={method}", f"tuning.n_trials={n}",
                                                 "tuning.epochs_per_trial=2", "tracking.backend=none"])
    cfg.tuning.search_space = space
    return cfg


def test_search_runs_all_trials_and_picks_best(tmp_path):
    train, calls = _fake_train(lambda lr, d: 0.5 + lr - d / 10)
    space = {"train.lr": [0.01, 0.1], "model.dropout": [0.2, 0.5]}
    summary = SearchRunner(_cfg(tmp_path, "g", "grid", space), data=object(), train_fn=train).run()
    assert len(calls) == 4 and all(c.train.epochs == 2 for c in calls)
    assert summary["best_params"] == {"train.lr": 0.1, "model.dropout": 0.2}
    assert summary["trials_to_best"] == 3
    assert "--set train.lr=0.1 model.dropout=0.2" in summary["retrain_command"]
    d = tmp_path / "tuning" / "g"
    assert (d / "results.json").exists() and (d / "summary.json").exists() and (d / "search.png").exists()


def test_search_resumes_and_survives_failed_trial(tmp_path):
    space = {"train.lr": [0.01, 5.0, 0.1]}
    train, calls = _fake_train(lambda lr, d: lr)
    cfg = _cfg(tmp_path, "r", "grid", space)
    runner = SearchRunner(cfg, data=object(), train_fn=train)
    runner.run_trial(0, {"train.lr": 0.01})
    train2, calls2 = _fake_train(lambda lr, d: lr)
    summary = SearchRunner(cfg, data=object(), train_fn=train2).run()
    assert [c.train.lr for c in calls2] == [5.0, 0.1]
    assert summary["n_trials"] == 3 and summary["n_failed"] == 1 and summary["best_score"] == 0.1


def test_random_search_and_comparison(tmp_path, capsys):
    rng_space = {"train.lr": {"dist": "loguniform", "low": 1e-3, "high": 1e-1},
                 "model.dropout": {"dist": "uniform", "low": 0.2, "high": 0.5}}
    train, _ = _fake_train(lambda lr, d: 0.5 + lr)
    SearchRunner(_cfg(tmp_path, "grid", "grid", {"train.lr": [1e-3, 1e-1], "model.dropout": [0.2, 0.5]}),
                 data=object(), train_fn=train).run()
    s = SearchRunner(_cfg(tmp_path, "random", "random", rng_space, n=4), data=object(), train_fn=train).run()
    assert s["distinct_values"] == {"train.lr": 4, "model.dropout": 4}
    rows = compare_searches([tmp_path / "tuning" / "grid", tmp_path / "tuning" / "random"], tmp_path / "tuning")
    assert [r["method"] for r in rows] == ["grid", "random"]
    assert (tmp_path / "tuning" / "comparison.png").exists()
    assert "distinct values" in capsys.readouterr().out


def test_summarize_handles_all_failed():
    s = summarize([{"trial": 0, "params": {}, "status": "failed", "score": None, "minutes": 1.0}], "grid", "val/accuracy")
    assert s["n_failed"] == 1 and "best_params" not in s


def test_tune_runs_real_training_end_to_end(tmp_path):
    pytest.importorskip("torch")
    import sys
    sys.path.insert(0, "tests")
    from test_training import FakeData, PatternDataset, tiny_cfg

    cfg = tiny_cfg(tmp_path, "train.epochs=1")
    cfg.experiment.name = "tiny-search"
    cfg.tuning.method, cfg.tuning.epochs_per_trial = "grid", 1
    cfg.tuning.search_space = {"train.lr": [1e-3, 3e-3]}
    data = FakeData(PatternDataset(seed=0), PatternDataset(4, seed=1), PatternDataset(4, seed=2), list("abcd"))
    s = SearchRunner(cfg, data=data).run()
    assert s["n_trials"] == 2 and s["n_failed"] == 0
    runs = tmp_path / "tuning" / "tiny-search" / "runs"
    assert (runs / "trial-00" / "history.json").exists()
    assert not (runs / "trial-00" / "best.pt").exists()
    assert json.loads((tmp_path / "tuning" / "tiny-search" / "results.json").read_text())[0]["status"] == "ok"
