"""Domain-specific validators for training samples.

Provides validation beyond generic quality scoring. Zelda-specific validators
live in plugins like hafs_scawful.
"""

from __future__ import annotations

import os
from pathlib import Path

from agents.training.validators.base import CompositeValidator, ValidationResult, Validator


def _ensure_scawful_on_path() -> None:
    import sys

    env_root = os.environ.get("HAFS_SCAWFUL_ROOT")
    candidates = [
        Path(env_root).expanduser() if env_root else None,
        Path("~/.config/hafs/plugins/hafs_scawful").expanduser(),
        Path("~/Code/hafs_scawful").expanduser(),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            parent = candidate.parent
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            break


def _lazy_import_validators() -> dict[str, type]:
    try:
        _ensure_scawful_on_path()
        from hafs_scawful.validators.asm_validator import AsmValidator
        from hafs_scawful.validators.cpp_validator import CppValidator
        from hafs_scawful.validators.kg_validator import KGValidator
        return {
            "AsmValidator": AsmValidator,
            "CppValidator": CppValidator,
            "KGValidator": KGValidator,
        }
    except ImportError:
        return {}


_plugin_validators: dict[str, type] | None = None


def __getattr__(name: str):
    global _plugin_validators
    validator_names = {"AsmValidator", "CppValidator", "KGValidator"}
    if name in validator_names:
        if _plugin_validators is None:
            _plugin_validators = _lazy_import_validators()
        if name in _plugin_validators:
            return _plugin_validators[name]
        raise ImportError(
            f"{name} is plugin-only. Install hafs_scawful or configure HAFS_SCAWFUL_ROOT."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CompositeValidator",
    "ValidationResult",
    "Validator",
    "AsmValidator",
    "CppValidator",
    "KGValidator",
]
