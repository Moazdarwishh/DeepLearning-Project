"""Model layer: encoders (scratch + pretrained), task heads and a registry."""
from .base import Classifier, Encoder
from .registry import available_encoders, build_encoder, register_encoder

__all__ = ["Encoder", "Classifier", "build_encoder", "register_encoder", "available_encoders"]
