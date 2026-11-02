"""Name -> encoder factory, so configs select models by name (Open/Closed principle).

Add a new encoder without touching the trainer::

    @register_encoder("my_cnn")
    class MyCNN(Encoder): ...
"""
from __future__ import annotations

from typing import Callable

_ENCODERS: dict[str, Callable] = {}


def register_encoder(name: str):
    def decorator(cls_or_fn):
        if name in _ENCODERS:
            raise KeyError(f"Encoder '{name}' is already registered")
        _ENCODERS[name] = cls_or_fn
        return cls_or_fn
    return decorator


def build_encoder(name: str, **kwargs):
    if name not in _ENCODERS:
        raise KeyError(f"Unknown encoder '{name}'. Available: {available_encoders()}")
    return _ENCODERS[name](**kwargs)


def available_encoders() -> list[str]:
    return sorted(_ENCODERS)
