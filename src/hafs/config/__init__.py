import importlib
import warnings
from typing import Any

_DEPRECATION_MESSAGE = "config is deprecated. Import from 'config' instead."

warnings.warn(
    _DEPRECATION_MESSAGE,
    DeprecationWarning,
    stacklevel=2,
)

_EXPORTS = [
    "AFSDirectoryConfig",
    "GeneralConfig",
    "HafsConfig",
    "ParsersConfig",
    "ParserConfig",
    "PolicyType",
    "ThemeConfig",
    "load_config",
]

def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        module = importlib.import_module("config")
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")

def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_EXPORTS))

__all__ = _EXPORTS
