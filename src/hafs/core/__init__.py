import importlib
import warnings
from typing import Any

_DEPRECATION_MESSAGE = "core is deprecated. Import from 'core' instead."

warnings.warn(
    _DEPRECATION_MESSAGE,
    DeprecationWarning,
    stacklevel=2,
)

# Core exports sessions by default
_EXPORTS = [
    "sessions",
]

def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        module = importlib.import_module("core")
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")

def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_EXPORTS))

__all__ = _EXPORTS
