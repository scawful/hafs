"""Plugin system for extending hafs functionality."""

from __future__ import annotations

from hafs.plugins.loader import PluginLoader
from hafs.plugins.protocol import (
    BackendPlugin,
    HafsPlugin,
    ParserPlugin,
    WidgetPlugin,
)

__all__ = [
    "BackendPlugin",
    "HafsPlugin",
    "ParserPlugin",
    "PluginLoader",
    "WidgetPlugin",
]
