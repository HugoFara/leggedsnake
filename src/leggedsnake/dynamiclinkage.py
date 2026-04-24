"""Deprecated alias for :mod:`leggedsnake.dynamic_linkage`.

Importing from ``leggedsnake.dynamiclinkage`` still works but emits a
:class:`DeprecationWarning`. Update callers to
``from leggedsnake.dynamic_linkage import ...``.
"""

from __future__ import annotations

import warnings as _warnings
from typing import Any

from . import dynamic_linkage as _target

_warnings.warn(
    "leggedsnake.dynamiclinkage is deprecated; "
    "import from leggedsnake.dynamic_linkage instead.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str) -> Any:
    return getattr(_target, name)


def __dir__() -> list[str]:
    return dir(_target)
