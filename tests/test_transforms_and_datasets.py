import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
datasets = pytest.importorskip("datasets")
from PIL import Image

from src.data.classification import (build_classification_data, build_dataloaders,
                                     load_exclusions)
from src.data.splits import make_stratified_split, save_split
from src.data.transforms import (IMAGENET_MEAN, IMAGENET_STD, build_eval_transform,
                                 build_train_transform, denormalize, normalization_stats)
from src.utils.config import load_config


def _image(w=300, h=200, mode="RGB", seed=0):
    rng = np.random.default_rng(seed)
    arr = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    return img.convert("L") if mode == "L" else img


def test_eval_transform_shape_and_determinism():
    cfg = load_config("configs/cls_resnet50.yaml")
    tf = build_eval_transform(cfg.data)
    img = _image(500, 333)
    a, b = tf(img), tf(img)
    assert a.shape == (3, 224, 224) and a.dtype == torch.float32
    assert torch.equal(a, b)


def test_train_transform_is_random_but_same_shape():
    cfg = load_config("configs/cls_resnet50.yaml")
    tf = build_train_transform(cfg.data)
    img = _image(500, 333)
    torch.manual_seed(0)
    a, b = tf(img), tf(img)
    assert a.shape == b.shape == (3, 224, 224)
    assert not torch.equal(a, b)


def test_augment_false_gives_eval_pipeline():
    cfg = load_config("configs/cls_resnet50.yaml", ["data.augment=false"])
    tf = build_train_transform(cfg.data)
    img = _image()
    assert torch.equal(tf(img), tf(img))


def test_normalization_choice_and_denormalize_roundtrip():
    scratch = load_config("configs/cls_scratch.yaml")
    assert normalization_stats(scratch.data)[0] == tuple(scratch.data.dataset_mean)
    resnet = load_config("configs/cls_resnet50.yaml")
    mean, std = normalization_stats(resnet.data)
    assert mean == IMAGENET_MEAN and std == IMAGENET_STD

    tf = build_eval_transform(resnet.data)
    img = Image.new("RGB", (256, 256), (200, 100, 50))
    back = denormalize(tf(img), mean, std)
    assert torch.allclose(back[:, 100, 100], torch.tensor([200, 100, 50]) / 255.0, atol=1e-2)


def _fake_cub(n, num_classes, offset=0, gray_every=7):
    from datasets import ClassLabel, Dataset, Features, Value
    from datasets import Image as HFImage

    names = [f"Bird {i}" for i in range(num_classes)]
    images = [_image(120, 90, "L" if i % gray_every == 0 else "RGB", seed=i) for i in range(n)]
    return Dataset.from_dict(
        {"image": images, "label": [i % num_classes for i in range(n)],
         "description": ["a bird"] * n, "id": [offset + i for i in range(n)]},
        features=Features({"image": HFImage(), "label": ClassLabel(names=names),
                           "description": Value("string"), "id": Value("int64")}),
    )


@pytest.fixture()
def fake_setup(tmp_path):
    train_hf, test_hf = _fake_cub(100, 10), _fake_cub(30, 10, offset=1000)
    split = make_stratified_split([int(x) for x in train_hf["label"]], 0.2, seed=0)
    save_split(split, tmp_path)
    dropped = split.val_indices[0]
    excl = tmp_path / "exclusions.json"
    excl.write_text(json.dumps({"exclude": [{"split": "val", "index": dropped}]}))
    cfg = load_config("configs/cls_resnet50.yaml", [
        f"data.splits_dir={tmp_path}", f"data.exclusions_file={excl}",
        "data.batch_size=8", "data.num_workers=0", "model.num_classes=10"])
    return cfg, train_hf, test_hf, split, dropped


def test_datasets_respect_split_and_exclusions(fake_setup):
    cfg, train_hf, test_hf, split, dropped = fake_setup
    data = build_classification_data(cfg, train_hf, test_hf)
    assert data.sizes() == {"train": 80, "val": 19, "test": 30}
    assert dropped not in data.val.indices
    assert not set(data.train.indices) & set(data.val.indices)
    x, y = data.train[0]
    assert x.shape == (3, 224, 224) and isinstance(y, int)
    assert data.class_names[0] == "Bird 0"


def test_grayscale_images_become_rgb(fake_setup):
    cfg, train_hf, test_hf, *_ = fake_setup
    data = build_classification_data(cfg, train_hf, test_hf)
    gray_pos = data.test.indices.index(0)
    x, _ = data.test[gray_pos]
    assert x.shape[0] == 3


def test_dataloaders_batch_and_shuffle(fake_setup):
    cfg, train_hf, test_hf, *_ = fake_setup
    loaders = build_dataloaders(cfg, build_classification_data(cfg, train_hf, test_hf))
    x, y = next(iter(loaders["train"]))
    assert x.shape == (8, 3, 224, 224) and y.shape == (8,)
    assert len(loaders["train"]) == 80 // 8
    assert len(loaders["val"]) == 3


def test_load_exclusions_missing_file(tmp_path):
    assert load_exclusions(tmp_path / "nope.json") == {"train": [], "val": [], "test": []}


def test_repo_exclusions_file_is_valid():
    excl = load_exclusions("data/splits/exclusions.json")
    assert sorted(excl["val"]) == [774, 5277]
    from src.data.splits import load_split
    split = load_split("data/splits")
    assert set(excl["val"]) <= set(split.val_indices)
