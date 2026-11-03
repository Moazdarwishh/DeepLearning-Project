"""Hyperparameter search: Grid Search vs Random Search (and optional Optuna/TPE).

How a search runs (``python -m src.cli tune --task classification --config configs/tune_grid.yaml``):

1. The search space comes from ``tuning.search_space`` in the config::

       train.lr: [1.0e-3, 1.0e-2]                              # list  -> grid values / random choice
       train.lr: {dist: loguniform, low: 1.0e-3, high: 1.0e-2} # range -> random / optuna only

2. Each trial = the base config + that trial's values, trained for
   ``tuning.epochs_per_trial`` epochs. Trials are scored on the VALIDATION set
   (``tuning.metric``, default val/accuracy). The test set is never used.
3. After every trial, ``outputs/tuning/<name>/results.json`` is rewritten, so a
   disconnected Colab session can simply re-run the command: finished trials are skipped.
4. At the end: ``summary.json`` (best trial, budget, time-to-best, the exact command to
   retrain the best setting) and ``search.png``.

``compare_searches`` puts several searches side by side (best score, trials and minutes
needed to reach it) - the comparison the project asks for.

Why compare them: with the same budget, grid search tests only a few values per
hyperparameter (2 x 2 x 2 here), while random search tests a new value in every trial.
When only some hyperparameters matter (often the learning rate), random search explores
the important ones more densely (Bergstra & Bengio, 2012).
"""
from __future__ import annotations

import itertools
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

from src.utils.config import config_from_dict

RANGE_KINDS = {"loguniform", "uniform", "int"}


# --------------------------------------------------------------- search space
def _is_range(spec) -> bool:
    return isinstance(spec, dict) and spec.get("dist") in RANGE_KINDS


def validate_space(space: dict[str, Any], method: str) -> None:
    if not space:
        raise ValueError("tuning.search_space is empty")
    for key, spec in space.items():
        if "." not in key:
            raise ValueError(f"Search-space key '{key}' must look like section.key (e.g. train.lr)")
        if _is_range(spec):
            if method == "grid":
                raise ValueError(f"Grid search needs a list of values for '{key}', got a range {spec}")
            if not spec["low"] < spec["high"]:
                raise ValueError(f"'{key}': low must be < high")
            if spec["dist"] == "loguniform" and spec["low"] <= 0:
                raise ValueError(f"'{key}': loguniform needs low > 0")
        elif not isinstance(spec, list) or not spec:
            raise ValueError(f"'{key}': expected a non-empty list or {{dist, low, high}}, got {spec!r}")


def grid_trials(space: dict[str, Any]) -> list[dict[str, Any]]:
    """Every combination (cartesian product) of the listed values."""
    validate_space(space, "grid")
    keys = list(space)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(space[k] for k in keys))]


def _sample(spec, rng: random.Random):
    if isinstance(spec, list):
        return rng.choice(spec)
    low, high = spec["low"], spec["high"]
    if spec["dist"] == "loguniform":          # learning rates / weight decay: spread over orders of magnitude
        return float(math.exp(rng.uniform(math.log(low), math.log(high))))
    if spec["dist"] == "uniform":
        return float(rng.uniform(low, high))
    return int(rng.randint(int(low), int(high)))


def random_trials(space: dict[str, Any], n_trials: int, seed: int = 0) -> list[dict[str, Any]]:
    """n_trials independent random draws (reproducible for a given seed)."""
    validate_space(space, "random")
    rng = random.Random(seed)
    return [{k: _sample(spec, rng) for k, spec in space.items()} for _ in range(n_trials)]


def set_dotted(cfg, key: str, value) -> None:
    section, attr = key.split(".", 1)
    target = getattr(cfg, section)
    if not hasattr(target, attr):
        raise ValueError(f"Unknown config key '{key}'")
    setattr(target, attr, value)


# -------------------------------------------------------------------- running
class SearchRunner:
    def __init__(self, cfg, data=None, base_config_path: str | None = None, train_fn=None):
        if train_fn is None:
            from src.pipelines.classification import train as train_fn
        self.cfg = cfg
        self.tcfg = cfg.tuning
        self.method = self.tcfg.method.lower()
        self.space = self.tcfg.search_space
        validate_space(self.space, self.method)
        self.train_fn = train_fn
        self.data = data
        self.base_config_path = base_config_path
        self.dir = Path(cfg.experiment.output_dir) / "tuning" / cfg.experiment.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.dir / "results.json"
        self.results: list[dict] = json.loads(self.results_path.read_text()) if self.results_path.exists() else []
        if self.results:
            print(f"Resuming search '{cfg.experiment.name}': {len(self.results)} trial(s) already done.")

    # one trial ---------------------------------------------------------------
    def _data_for(self, trial_cfg, params):
        if any(k.startswith("data.") for k in params):
            return None                      # data settings change -> let train() rebuild datasets
        if self.data is None:
            from src.data.classification import build_classification_data
            self.data = build_classification_data(trial_cfg)
        return self.data

    def run_trial(self, index: int, params: dict[str, Any]) -> dict:
        trial_cfg = config_from_dict(self.cfg.to_dict())
        for key, value in params.items():
            set_dotted(trial_cfg, key, value)
        trial_cfg.train.epochs = self.tcfg.epochs_per_trial
        trial_cfg.experiment.name = f"trial-{index:02d}"
        trial_cfg.experiment.output_dir = str(self.dir)
        trial_cfg.experiment.tags = list(self.cfg.experiment.tags) + [f"search:{self.cfg.experiment.name}"]

        shown = ", ".join(f"{k}={_fmt(v)}" for k, v in params.items())
        print(f"\n===== {self.cfg.experiment.name} | trial {index + 1} | {shown} =====", flush=True)
        start = time.perf_counter()
        record = {"trial": index, "params": params}
        try:
            summary = self.train_fn(trial_cfg, data=self._data_for(trial_cfg, params))
            record.update(status="ok", score=summary["best_val"][self.tcfg.metric],
                          best_val=summary["best_val"], best_epoch=summary["best_epoch"],
                          run_dir=str(self.dir / "runs" / summary["run"]))
            if not self.tcfg.keep_checkpoints:
                for ckpt in ("best.pt", "last.pt"):
                    (self.dir / "runs" / summary["run"] / ckpt).unlink(missing_ok=True)
        except (FloatingPointError, RuntimeError) as exc:   # e.g. loss exploded with a huge lr
            print(f"Trial {index + 1} failed: {exc}", flush=True)
            record.update(status="failed", score=None, error=str(exc)[:300])
        record["minutes"] = round((time.perf_counter() - start) / 60, 3)
        self.results.append(record)
        self.results_path.write_text(json.dumps(self.results, indent=1))
        return record

    def _done(self, index: int, params: dict) -> bool:
        return any(r["trial"] == index and _same(r["params"], params) for r in self.results)

    # whole search ------------------------------------------------------------
    def run(self) -> dict:
        start = time.perf_counter()
        if self.method == "grid":
            planned = grid_trials(self.space)
        elif self.method == "random":
            planned = random_trials(self.space, self.tcfg.n_trials, self.tcfg.seed)
        elif self.method == "optuna":
            self._run_optuna()
            planned = None
        else:
            raise ValueError(f"Unknown tuning.method '{self.method}' (grid | random | optuna)")

        if planned is not None:
            print(f"{self.method} search: {len(planned)} trials x {self.tcfg.epochs_per_trial} epochs")
            for i, params in enumerate(planned):
                if not self._done(i, params):
                    self.run_trial(i, params)
        return self.finish(time.perf_counter() - start)

    def _run_optuna(self) -> None:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        dists = {}
        for key, spec in self.space.items():
            if isinstance(spec, list):
                dists[key] = optuna.distributions.CategoricalDistribution(spec)
            elif spec["dist"] == "int":
                dists[key] = optuna.distributions.IntDistribution(int(spec["low"]), int(spec["high"]))
            else:
                dists[key] = optuna.distributions.FloatDistribution(spec["low"], spec["high"],
                                                                    log=spec["dist"] == "loguniform")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.tcfg.seed))
        for r in self.results:                      # resume: tell Optuna what was already tried
            if r["status"] == "ok":
                study.add_trial(optuna.trial.create_trial(params=r["params"], distributions=dists, value=r["score"]))
        print(f"optuna (TPE) search: {self.tcfg.n_trials} trials x {self.tcfg.epochs_per_trial} epochs")
        for i in range(len(self.results), self.tcfg.n_trials):
            trial = study.ask(dists)
            record = self.run_trial(i, dict(trial.params))
            if record["status"] == "ok":
                study.tell(trial, record["score"])
            else:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)

    def finish(self, elapsed_s: float) -> dict:
        from src.visualization.tuning_plots import plot_search

        summary = summarize(self.results, self.method, self.tcfg.metric)
        summary.update(name=self.cfg.experiment.name, epochs_per_trial=self.tcfg.epochs_per_trial,
                       search_space=self.space, session_minutes=round(elapsed_s / 60, 2))
        if summary.get("best_params"):
            sets = " ".join(f"{k}={_fmt(v)}" for k, v in summary["best_params"].items())
            cfg_path = self.base_config_path or f"configs/{self.cfg.experiment.name}.yaml"
            summary["retrain_command"] = (
                f"python -m src.cli train --task classification --config {cfg_path} --set {sets} "
                f"train.epochs=30 train.early_stopping_patience=8 train.save_every_epoch=true "
                f"experiment.name={self.cfg.experiment.name}-best")
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        plot_search(self.results, self.space, self.dir / "search.png", self.cfg.experiment.name, self.tcfg.metric)

        print(f"\n{'=' * 70}\nSearch '{self.cfg.experiment.name}' ({self.method}) finished")
        if summary.get("best_params"):
            print(f"best {self.tcfg.metric} = {summary['best_score']:.4f} at trial {summary['best_trial'] + 1} "
                  f"of {summary['n_trials']} ({summary['minutes_to_best']:.1f} of {summary['total_minutes']:.1f} min)")
            print("best params: " + ", ".join(f"{k}={_fmt(v)}" for k, v in summary["best_params"].items()))
            print(f"retrain    : {summary['retrain_command']}")
        print(f"saved      : {self.dir}")
        return summary


# ------------------------------------------------------------------ summaries
def summarize(results: list[dict], method: str, metric: str) -> dict:
    ok = [r for r in sorted(results, key=lambda r: r["trial"]) if r["status"] == "ok"]
    out = {"method": method, "metric": metric, "n_trials": len(results),
           "n_failed": sum(r["status"] != "ok" for r in results),
           "total_minutes": round(sum(r["minutes"] for r in results), 2)}
    if not ok:
        return out
    best = max(ok, key=lambda r: r["score"])
    ordered = sorted(results, key=lambda r: r["trial"])
    minutes_to_best = sum(r["minutes"] for r in ordered if r["trial"] <= best["trial"])
    out.update(best_trial=best["trial"], best_score=best["score"], best_params=best["params"],
               best_val=best["best_val"], trials_to_best=best["trial"] + 1,
               minutes_to_best=round(minutes_to_best, 2),
               mean_score=sum(r["score"] for r in ok) / len(ok),
               worst_score=min(r["score"] for r in ok),
               distinct_values={k: len({_fmt(r["params"][k]) for r in results}) for k in best["params"]})
    return out


def compare_searches(dirs: list[str | Path], out_dir: str | Path = "outputs/tuning") -> list[dict]:
    """Side-by-side table + figure of several finished searches."""
    from src.visualization.tuning_plots import plot_comparison

    rows, all_results = [], {}
    for d in map(Path, dirs):
        s = json.loads((d / "summary.json").read_text())
        all_results[s["name"]] = json.loads((d / "results.json").read_text())
        rows.append({"search": s["name"], "method": s["method"], "trials": s["n_trials"], "failed": s["n_failed"],
                     "best_score": s.get("best_score"), "mean_score": s.get("mean_score"),
                     "worst_score": s.get("worst_score"), "trials_to_best": s.get("trials_to_best"),
                     "minutes_to_best": s.get("minutes_to_best"), "total_minutes": s["total_minutes"],
                     "distinct_values": s.get("distinct_values"), "best_params": s.get("best_params")})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(rows, indent=2))
    plot_comparison(all_results, out_dir / "comparison.png")

    header = f"{'search':<14}{'method':<8}{'trials':>7}{'best':>9}{'mean':>9}{'worst':>9}{'to best':>9}{'min->best':>11}{'total min':>11}"
    print(header + "\n" + "-" * len(header))
    for r in rows:
        f = lambda v, p=4: "-" if v is None else f"{v:.{p}f}"   # noqa: E731
        print(f"{r['search']:<14}{r['method']:<8}{r['trials']:>7}{f(r['best_score']):>9}{f(r['mean_score']):>9}"
              f"{f(r['worst_score']):>9}{str(r['trials_to_best']):>9}{f(r['minutes_to_best'], 1):>11}"
              f"{f(r['total_minutes'], 1):>11}")
    for r in rows:
        print(f"{r['search']}: distinct values tried per hyperparameter = {r['distinct_values']}")
    print(f"\nsaved -> {out_dir / 'comparison.json'}, {out_dir / 'comparison.png'}")
    return rows


def _fmt(v) -> str:
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def _same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(_fmt(a[k]) == _fmt(b[k]) for k in a)
