"""Deprecated alias for :mod:`leggedsnake.physics_engine`.

Importing from ``leggedsnake.physicsengine`` still works but emits a
:class:`DeprecationWarning`. Update callers to
``from leggedsnake.physics_engine import ...``.
"""

from __future__ import annotations

import warnings as _warnings
from typing import Any

from . import physics_engine as _target

_warnings.warn(
    "leggedsnake.physicsengine is deprecated; "
    "import from leggedsnake.physics_engine instead.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str) -> Any:
    return getattr(_target, name)


def __dir__() -> list[str]:
    return dir(_target)
