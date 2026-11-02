import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from src.models import Classifier, Encoder, available_encoders, build_encoder, register_encoder


class _TinyEncoder(Encoder):
    out_channels = [8, 16]
    reductions = [2, 4]

    def __init__(self):
        super().__init__()
        self.s1 = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.s2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)

    def forward_features(self, x):
        f1 = self.s1(x)
        return [f1, self.s2(f1)]


def test_encoder_contract_and_classifier():
    enc = _TinyEncoder()
    feats = enc.forward_features(torch.randn(2, 3, 32, 32))
    assert [f.shape[1] for f in feats] == enc.out_channels
    assert [32 // f.shape[-1] for f in feats] == enc.reductions
    logits = Classifier(enc, num_classes=200)(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 200)


def test_registry():
    register_encoder("_tiny_test")(_TinyEncoder)
    assert "_tiny_test" in available_encoders()
    assert isinstance(build_encoder("_tiny_test"), Encoder)
