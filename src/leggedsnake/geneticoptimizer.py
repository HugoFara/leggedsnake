"""Deprecated alias for :mod:`leggedsnake.genetic_optimizer`.

Importing from ``leggedsnake.geneticoptimizer`` still works but emits a
:class:`DeprecationWarning`. Update callers to
``from leggedsnake.genetic_optimizer import ...``.
"""

from __future__ import annotations

import warnings as _warnings
from typing import Any

from . import genetic_optimizer as _target

_warnings.warn(
    "leggedsnake.geneticoptimizer is deprecated; "
    "import from leggedsnake.genetic_optimizer instead.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str) -> Any:
    return getattr(_target, name)


def __dir__() -> list[str]:
    return dir(_target)
