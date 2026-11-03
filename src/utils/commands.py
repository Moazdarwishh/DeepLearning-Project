"""Registry of CLI command handlers: (command, task) -> function.

Kept in its own module (not in src/cli.py) because ``python -m src.cli`` runs the CLI
as ``__main__``; a registry defined there would be duplicated when pipelines import it.
"""
from __future__ import annotations

import argparse
from typing import Callable

HANDLERS: dict[tuple[str, str], Callable[[argparse.Namespace], int]] = {}


def register(command: str, task: str):
    def decorator(fn):
        HANDLERS[(command, task)] = fn
        return fn
    return decorator
