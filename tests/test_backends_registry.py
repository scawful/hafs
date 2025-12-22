from __future__ import annotations

import importlib

from backends.base import BackendRegistry


def test_backends_register_oneshot_backends() -> None:
    BackendRegistry.clear()

    import backends as backends

    importlib.reload(backends)

    names = set(BackendRegistry.list_backends())
    assert "gemini_oneshot" in names
    assert "claude_oneshot" in names

    BackendRegistry.clear()

