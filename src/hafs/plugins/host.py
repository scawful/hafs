"""Headless plugin host for non-UI contexts."""

from __future__ import annotations

from typing import Any


class HeadlessPluginHost:
    """Minimal app-like host for plugin activation in headless flows."""

    def __init__(self, config: Any | None = None) -> None:
        self.config = config
        self.widget_plugins: list[Any] = []

    def register_widget_plugin(self, plugin: Any) -> None:
        """Capture widget plugins without attaching them to a UI."""
        self.widget_plugins.append(plugin)
