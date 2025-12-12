"""Plugin system for extending hafs functionality.

Keep imports light here to avoid circular dependencies.
"""

from __future__ import annotations

from typing import Any

from hafs.plugins.protocol import (
    BackendPlugin,
    HafsPlugin,
    ParserPlugin,
    ToolPlugin,
    WidgetPlugin,
)

__all__ = [
    "BackendPlugin",
    "HafsPlugin",
    "ParserPlugin",
    "ToolPlugin",
    "WidgetPlugin",
    "PluginLoader",
]


def __getattr__(name: str) -> Any:
    if name == "PluginLoader":
        from hafs.plugins.loader import PluginLoader

        return PluginLoader
    raise AttributeError(name)
