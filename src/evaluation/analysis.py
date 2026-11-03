from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

METRIC_EXPLANATIONS = {
    "accuracy": "Share of images whose top-1 prediction is the correct species. Easy to read, but "
                "dominated by large classes when classes are imbalanced (test: 11-30 images per class).",
    "top5_accuracy": "Share of images whose correct species is among the 5 highest-scoring predictions. "
                     "Shows whether the model is 'close' on fine-grained look-alike species.",
    "macro_precision": "For each species: of the images predicted as this species, how many really are it; "
                       "then averaged over species with equal weight. Low = many false alarms.",
    "macro_recall": "For each species: of its images, how many were found; averaged with equal weight. "
                    "Equal to the mean per-class accuracy.",
    "macro_f1": "Harmonic mean of precision and recall per species, averaged with equal weight - the "
                "main 'fair to every species' summary number, robust to the small test classes.",
    "confusion_matrix": "Row = true species, column = predicted species. Off-diagonal cells show which "
                        "species are mistaken for which (e.g. similar gulls, sparrows, terns).",
}


def confusion_matrix(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (targets, preds), 1)
    return cm


def per_class_table(cm: np.ndarray, class_names: list[str]) -> list[dict]:
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    correct = np.diag(cm)
    rows = []
    for c, name in enumerate(class_names):
        recall = correct[c] / support[c] if support[c] else 0.0
        precision = correct[c] / predicted[c] if predicted[c] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"class_id": c, "class": name, "support": int(support[c]), "correct": int(correct[c]),
                     "accuracy": round(float(recall), 4), "precision": round(float(precision), 4),
                     "f1": round(float(f1), 4)})
    return rows


def top_confusions(cm: np.ndarray, class_names: list[str], k: int = 20) -> list[dict]:
    off = cm.copy()
    np.fill_diagonal(off, 0)
    support = cm.sum(axis=1)
    flat = np.argsort(off, axis=None)[::-1][:k]
    pairs = []
    for idx in flat:
        t, p = np.unravel_index(idx, off.shape)
        if off[t, p] == 0:
            break
        pairs.append({"true_id": int(t), "true": class_names[t], "pred_id": int(p), "predicted": class_names[p],
                      "count": int(off[t, p]), "share_of_true_class": round(float(off[t, p] / support[t]), 4),
                      "reverse_count": int(off[p, t])})
    return pairs


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(logits, targets, class_names: list[str], out_dir: str | Path, dataset=None, reference=None,
            norm=None, top_k_pairs: int = 20) -> dict:
    """Write per-class tables, confused pairs and figures for one evaluated split."""
    from src.visualization import analysis_plots as P

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logits = np.asarray(logits)
    targets = np.asarray(targets).astype(int)
    preds = logits.argmax(axis=1)
    num_classes = len(class_names)

    cm = confusion_matrix(preds, targets, num_classes)
    per_class = per_class_table(cm, class_names)
    pairs = top_confusions(cm, class_names, top_k_pairs)
    _write_csv(per_class, out_dir / "per_class.csv")
    _write_csv(pairs, out_dir / "confused_pairs.csv")
    np.save(out_dir / "confusion_matrix.npy", cm)

    ranked = sorted((r for r in per_class if r["support"]), key=lambda r: (r["accuracy"], r["f1"]))
    accs = np.array([r["accuracy"] for r in per_class if r["support"]])
    summary = {
        "images": int(len(targets)),
        "accuracy": float((preds == targets).mean()),
        "mean_per_class_accuracy": float(accs.mean()),
        "per_class_accuracy_std": float(accs.std()),
        "classes_with_0_accuracy": int((accs == 0).sum()),
        "classes_with_100_accuracy": int((accs == 1).sum()),
        "worst_classes": [{k: r[k] for k in ("class", "support", "accuracy", "f1")} for r in ranked[:10]],
        "best_classes": [{k: r[k] for k in ("class", "support", "accuracy", "f1")} for r in ranked[::-1][:10]],
        "top_confused_pairs": pairs[:10],
        "metric_explanations": METRIC_EXPLANATIONS,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    P.per_class_figure(per_class, out_dir / "per_class.png")
    P.confusion_figure(cm, class_names, pairs, out_dir / "confusion_matrix.png")
    if dataset is not None and norm is not None:
        P.confused_pairs_figure(pairs[:6], logits, targets, dataset, reference, class_names, norm,
                                out_dir / "confused_pairs.png")
    return summary


def print_summary(summary: dict, n: int = 5) -> None:
    print(f"\nMean per-class accuracy {summary['mean_per_class_accuracy']:.3f} "
          f"(std {summary['per_class_accuracy_std']:.3f}); "
          f"{summary['classes_with_0_accuracy']} species never correct, "
          f"{summary['classes_with_100_accuracy']} always correct.")
    print("Hardest species : " + ", ".join(f"{r['class']} ({r['accuracy']:.0%})" for r in summary["worst_classes"][:n]))
    print("Easiest species : " + ", ".join(f"{r['class']} ({r['accuracy']:.0%})" for r in summary["best_classes"][:n]))
    print("Most confused   :")
    for p in summary["top_confused_pairs"][:n]:
        print(f"   {p['true']:<28} -> {p['predicted']:<28} {p['count']}x"
              + (f"  (and {p['reverse_count']}x the other way)" if p["reverse_count"] else ""))


def compare_runs(run_dirs: list[str | Path], split: str, out_dir: str | Path = "outputs") -> list[dict]:
    """Table + bar chart of several evaluated runs (needs eval_<split>.json in each run dir)."""
    from src.visualization.analysis_plots import comparison_figure

    rows = []
    for d in map(Path, run_dirs):
        ev_path = d / f"eval_{split}.json"
        if not ev_path.exists():
            raise FileNotFoundError(f"{ev_path} missing - run: python -m src.cli evaluate --task classification "
                                    f"--checkpoint {d / 'best.pt'} --split {split}")
        ev = json.loads(ev_path.read_text())
        sm = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
        rows.append({"run": d.name,
                     "accuracy": ev[f"{split}/accuracy"], "top5_accuracy": ev[f"{split}/top5_accuracy"],
                     "macro_precision": ev[f"{split}/macro_precision"], "macro_recall": ev[f"{split}/macro_recall"],
                     "macro_f1": ev[f"{split}/macro_f1"],
                     "trainable_params": sm.get("params", {}).get("trainable"),
                     "epochs_run": sm.get("epochs_run"), "train_minutes": sm.get("train_time_min")})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, out_dir / f"comparison_{split}.csv")
    lines = [f"| run | {split} accuracy | top-5 | macro precision | macro recall | macro F1 | trainable params | epochs | minutes |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        params = f"{r['trainable_params'] / 1e6:.2f}M" if r["trainable_params"] else "-"
        lines.append(f"| {r['run']} | {r['accuracy']:.4f} | {r['top5_accuracy']:.4f} | {r['macro_precision']:.4f} | "
                     f"{r['macro_recall']:.4f} | {r['macro_f1']:.4f} | {params} | {r['epochs_run'] or '-'} | "
                     f"{r['train_minutes'] or '-'} |")
    table = "\n".join(lines)
    (out_dir / f"comparison_{split}.md").write_text(table + "\n")
    comparison_figure(rows, split, out_dir / f"comparison_{split}.png")
    print(table)
    print(f"\nsaved -> {out_dir / f'comparison_{split}.md'}, .csv, .png")
    return rows
