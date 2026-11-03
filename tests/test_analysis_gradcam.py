import json
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, "tests")
from test_training import FakeData, PatternDataset, tiny_cfg  # noqa: E402

from src.evaluation.analysis import (analyze, compare_runs, confusion_matrix, per_class_table,  # noqa: E402
                                     top_confusions)


def test_confusion_per_class_and_pairs():
    targets = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    preds = np.array([0, 1, 1, 1, 1, 2, 0, 2])
    cm = confusion_matrix(preds, targets, 3)
    assert cm.tolist() == [[1, 2, 0], [0, 2, 0], [1, 0, 2]]
    rows = per_class_table(cm, ["a", "b", "c"])
    assert rows[0]["accuracy"] == pytest.approx(1 / 3) or rows[0]["accuracy"] == 0.3333
    assert rows[1]["precision"] == 0.5 and rows[1]["accuracy"] == 1.0
    pairs = top_confusions(cm, ["a", "b", "c"])
    assert (pairs[0]["true"], pairs[0]["predicted"], pairs[0]["count"]) == ("a", "b", 2)
    assert {(p["true"], p["predicted"]) for p in pairs} == {("a", "b"), ("c", "a")}


def test_analyze_writes_tables_and_figures(tmp_path):
    rng = np.random.default_rng(0)
    targets = np.repeat(np.arange(4), 4)
    logits = rng.normal(size=(16, 4))
    logits[np.arange(16), targets] += 2.0
    ds = PatternDataset(4, seed=1)
    summary = analyze(logits, targets, list("abcd"), tmp_path, dataset=ds, reference=None,
                      norm=((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)))
    for f in ("per_class.csv", "confused_pairs.csv", "summary.json", "per_class.png", "confusion_matrix.png"):
        assert (tmp_path / f).exists(), f
    assert 0 <= summary["accuracy"] <= 1 and "macro_f1" in summary["metric_explanations"]
    if summary["top_confused_pairs"]:
        assert (tmp_path / "confused_pairs.png").exists()


def test_compare_runs_table(tmp_path, capsys):
    for name, acc in (("cls-scratch", 0.45), ("cls-resnet50-ft", 0.80)):
        d = tmp_path / name
        d.mkdir()
        (d / "eval_test.json").write_text(json.dumps({f"test/{k}": acc for k in
                                                      ("accuracy", "top5_accuracy", "macro_precision", "macro_recall", "macro_f1")}))
        (d / "summary.json").write_text(json.dumps({"params": {"trainable": 11_000_000}, "epochs_run": 30, "train_time_min": 40}))
    rows = compare_runs([tmp_path / "cls-scratch", tmp_path / "cls-resnet50-ft"], "test", tmp_path)
    assert [r["accuracy"] for r in rows] == [0.45, 0.80]
    assert (tmp_path / "comparison_test.md").exists() and (tmp_path / "comparison_test.png").exists()
    assert "| cls-resnet50-ft | 0.8000" in capsys.readouterr().out


def test_compare_runs_explains_missing_eval(tmp_path):
    (tmp_path / "run").mkdir()
    with pytest.raises(FileNotFoundError, match="evaluate"):
        compare_runs([tmp_path / "run"], "test", tmp_path)


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    from src.pipelines.classification import train
    tmp = tmp_path_factory.mktemp("gc")
    data = FakeData(PatternDataset(size=64, seed=0), PatternDataset(4, size=64, seed=1),
                    PatternDataset(4, size=64, seed=2), list("abcd"))
    summary = train(tiny_cfg(tmp, "train.epochs=12"), data=data)
    assert summary["best_val"]["val/accuracy"] >= 0.9
    return tmp / "runs" / "tiny", data


def test_gradcam_equals_analytic_cam_for_gap_linear_head(trained_run):
    import torch.nn.functional as F

    from src.pipelines.classification import load_model
    from src.visualization.gradcam import GradCAM

    run, data = trained_run
    model, *_ = load_model(run / "best.pt", device=torch.device("cpu"))
    cam = GradCAM(model)
    w = model.head[1].weight
    for i in range(0, len(data.val), 3):
        x, _ = data.val[i]
        heat, logits, idx = cam(x.unsqueeze(0))
        A = cam.activations.detach()[0]
        expected = F.relu((w[idx[0]].view(-1, 1, 1) * A).sum(0, keepdim=True)).unsqueeze(0)
        expected = F.interpolate(expected, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        expected = (expected - expected.min()) / (expected.max() - expected.min() + 1e-8)
        assert torch.allclose(heat[0], expected, atol=1e-4)
        assert heat.shape == (1, 64, 64) and 0 <= heat.min() and heat.max() <= 1
        assert int(idx[0]) == int(logits.argmax())
    cam.remove()


def test_gradcam_for_a_chosen_class_differs(trained_run):
    from src.pipelines.classification import load_model
    from src.visualization.gradcam import GradCAM

    run, data = trained_run
    model, *_ = load_model(run / "best.pt", device=torch.device("cpu"))
    x, y = data.val[0]
    cam = GradCAM(model)
    h_pred, _, _ = cam(x.unsqueeze(0))
    h_other, _, idx = cam(x.unsqueeze(0), class_idx=(y + 1) % 4)
    assert int(idx[0]) == (y + 1) % 4 and not torch.allclose(h_pred, h_other)
    cam.remove()


def test_gradcam_pipeline_correct_wrong_and_two_models(trained_run, tmp_path):
    from src.pipelines.classification import gradcam
    run, data = trained_run
    paths = gradcam([run / "best.pt", run / "best.pt"], split="val", num_samples=3, output=tmp_path, data=data)
    assert any("correct" in p.name for p in paths)
    assert all(p.exists() for p in paths)


def test_evaluate_with_analysis(trained_run):
    from src.pipelines.classification import evaluate
    run, data = trained_run
    metrics = evaluate(run / "best.pt", "test", ["data.num_workers=0"], data=data)
    assert metrics["test/accuracy"] >= 0.75
    assert (run / "eval_test.json").exists() and (run / "analysis_test" / "per_class.csv").exists()


def test_gradcam_on_frozen_pretrained_backbone():
    pytest.importorskip("torchvision")
    from src.models import build_classifier
    from src.utils.config import load_config
    from src.visualization.gradcam import GradCAM

    cfg = load_config("configs/cls_resnet50_probe.yaml", ["model.pretrained=false", "model.name=resnet18"])
    model = build_classifier(cfg.model)
    heat, logits, idx = GradCAM(model)(torch.randn(2, 3, 64, 64))
    assert heat.shape == (2, 64, 64) and torch.isfinite(heat).all()
