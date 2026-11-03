from __future__ import annotations

import argparse
from typing import Callable

HANDLERS: dict[tuple[str, str], Callable[[argparse.Namespace], int]] = {}


def register(command: str, task: str):
    def decorator(fn):
        HANDLERS[(command, task)] = fn
        return fn
    return decorator
