"""Model layer: encoders (scratch + pretrained), task heads and a registry.

Importing this package registers every encoder in the registry.
"""
from . import scratch_cnn  # noqa: F401  (registers "scratch_cnn")
from .base import Classifier, Encoder
from .factory import build_classifier, count_parameters
from .registry import available_encoders, build_encoder, register_encoder

__all__ = ["Encoder", "Classifier", "build_encoder", "register_encoder", "available_encoders",
           "build_classifier", "count_parameters"]
