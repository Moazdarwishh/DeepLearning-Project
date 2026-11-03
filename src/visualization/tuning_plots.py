from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

COLORS = ["#3b6ea8", "#e08a2e", "#4a9a5b", "#c0504d"]


def _best_so_far(scores):
    out, best = [], None
    for s in scores:
        if s is not None and (best is None or s > best):
            best = s
        out.append(best)
    return out


def plot_search(results: list[dict], space: dict, out_path: Path, title: str, metric: str) -> Path:
    """Score per trial and score against each hyperparameter."""
    results = sorted(results, key=lambda r: r["trial"])
    keys = list(space)
    fig, axes = plt.subplots(1, 1 + len(keys), figsize=(4.6 * (1 + len(keys)), 3.8))
    x = [r["trial"] + 1 for r in results]
    scores = [r["score"] for r in results]
    axes[0].scatter(x, [s if s is not None else float("nan") for s in scores], color=COLORS[0], label="trial")
    axes[0].step(x, _best_so_far(scores), where="post", color=COLORS[1], label="best so far")
    axes[0].set_xlabel("trial")
    axes[0].set_ylabel(metric)
    axes[0].set_title("Score per trial")
    axes[0].legend(frameon=False)
    for ax, key in zip(axes[1:], keys):
        vals = [r["params"][key] for r in results if r["score"] is not None]
        ax.scatter(vals, [r["score"] for r in results if r["score"] is not None], color=COLORS[0])
        spec = space[key]
        if isinstance(spec, dict) and spec.get("dist") == "loguniform" or key.endswith(("lr", "weight_decay")):
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(key)
        ax.set_title(f"{metric} vs {key.split('.')[-1]}")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_comparison(results_by_search: dict[str, list[dict]], out_path: Path) -> Path:
    """Best-so-far score against trial number and against cumulative minutes."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for color, (name, results) in zip(COLORS, results_by_search.items()):
        results = sorted(results, key=lambda r: r["trial"])
        scores = [r["score"] for r in results]
        best = _best_so_far(scores)
        x = list(range(1, len(results) + 1))
        minutes, total = [], 0.0
        for r in results:
            total += r["minutes"]
            minutes.append(total)
        axes[0].step(x, best, where="post", color=color, label=name)
        axes[0].scatter(x, [s if s is not None else float("nan") for s in scores], color=color, alpha=0.35, s=18)
        axes[1].step(minutes, best, where="post", color=color, label=name)
    axes[0].set_xlabel("trial")
    axes[0].set_title("Best validation score so far (dots = each trial)")
    axes[1].set_xlabel("cumulative minutes")
    axes[1].set_title("Best score vs search time")
    for ax in axes:
        ax.set_ylabel("val accuracy")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path
