"""Deprecated alias for :mod:`leggedsnake.world_visualizer`.

Importing from ``leggedsnake.worldvisualizer`` still works but emits a
:class:`DeprecationWarning`. Update callers to
``from leggedsnake.world_visualizer import ...``.
"""

from __future__ import annotations

import warnings as _warnings
from typing import Any

from . import world_visualizer as _target

_warnings.warn(
    "leggedsnake.worldvisualizer is deprecated; "
    "import from leggedsnake.world_visualizer instead.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str) -> Any:
    return getattr(_target, name)


def __dir__() -> list[str]:
    return dir(_target)
