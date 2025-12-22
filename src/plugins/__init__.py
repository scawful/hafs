"""Plugin system for extending hafs functionality.

Keep imports light here to avoid circular dependencies.
"""

from __future__ import annotations

from typing import Any

from plugins.protocol import (
    BackendPlugin,
    HafsPlugin,
    IntegrationPlugin,
    ParserPlugin,
    ToolPlugin,
    WidgetPlugin,
)

__all__ = [
    "BackendPlugin",
    "HafsPlugin",
    "HeadlessPluginHost",
    "IntegrationPlugin",
    "ParserPlugin",
    "ToolPlugin",
    "WidgetPlugin",
    "PluginLoader",
]


def __getattr__(name: str) -> Any:
    if name == "PluginLoader":
        from plugins.loader import PluginLoader

        return PluginLoader
    if name == "HeadlessPluginHost":
        from plugins.host import HeadlessPluginHost

        return HeadlessPluginHost
    raise AttributeError(name)
