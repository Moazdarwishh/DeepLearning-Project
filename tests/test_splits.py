import pytest

pytest.importorskip("sklearn")

from src.data.datasets import split_captions
from src.data.splits import load_split, make_stratified_split, save_split


def _labels(num_classes=200, per_class=30):
    return [c for c in range(num_classes) for _ in range(per_class)]


def test_stratified_split_properties(tmp_path):
    labels = _labels()
    split = make_stratified_split(labels, val_fraction=0.15, seed=42)
    s = split.summary(labels)
    assert s["train_size"] + s["val_size"] == len(labels)
    assert s["classes_in_val"] == 200 and s["classes_in_train"] == 200
    assert 4 <= s["val_per_class_min"] <= s["val_per_class_max"] <= 5
    assert not set(split.train_indices) & set(split.val_indices)

    path = save_split(split, tmp_path)
    loaded = load_split(tmp_path, expected_rows=len(labels))
    assert loaded.val_indices == split.val_indices and path.exists()


def test_split_is_deterministic():
    labels = _labels(20, 10)
    a = make_stratified_split(labels, 0.2, seed=1)
    b = make_stratified_split(labels, 0.2, seed=1)
    assert a.val_indices == b.val_indices


def test_load_rejects_wrong_size(tmp_path):
    save_split(make_stratified_split(_labels(10, 10), 0.2, 0), tmp_path)
    with pytest.raises(ValueError):
        load_split(tmp_path, expected_rows=999)


def test_split_captions():
    assert split_captions("a bird.\n\n  another bird \n") == ["a bird.", "another bird"]
