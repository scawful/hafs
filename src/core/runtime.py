"""Runtime helpers for locating executables and environment details."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

try:
    from config.loader import load_config
    from config.schema import HafsConfig
except Exception:  # pragma: no cover - config may be unavailable in minimal installs
    load_config = None
    HafsConfig = None  # type: ignore[assignment]


def _expand_executable(value: str) -> str:
    """Expand ~ in executable paths while preserving bare commands."""
    if not value:
        return value
    return os.path.expanduser(value)


def _find_repo_root() -> Optional[Path]:
    """Best-effort repo root detection for source checkouts."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def resolve_python_executable(config: Optional["HafsConfig"] = None) -> str:
    """Resolve the Python executable for background services.

    Priority:
    1. HAFS_PYTHON environment variable
    2. general.python_executable in config
    3. VIRTUAL_ENV/bin/python
    4. .venv/bin/python from repo root or cwd
    5. sys.executable
    """
    env_override = os.environ.get("HAFS_PYTHON")
    if env_override:
        return _expand_executable(env_override)

    if config is None and load_config:
        try:
            config = load_config()
        except Exception:
            config = None

    if config is not None:
        python_override = getattr(config.general, "python_executable", None)
        if python_override:
            return _expand_executable(str(python_override))

    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path:
        candidate = Path(venv_path) / "bin" / "python"
        if candidate.exists():
            return str(candidate)

    repo_root = _find_repo_root()
    for root in [Path.cwd(), repo_root]:
        if not root:
            continue
        candidate = root / ".venv" / "bin" / "python"
        if candidate.exists():
            return str(candidate)

    return sys.executable
