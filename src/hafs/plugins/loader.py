"""Plugin loader for discovering and loading hafs plugins."""

from __future__ import annotations

import importlib.util
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from hafs.core.registry import agent_registry
from hafs.core.tools import ToolRegistry
from hafs.plugins.protocol import (
    BackendPlugin,
    HafsPlugin,
    IntegrationPlugin,
    ParserPlugin,
    ToolPlugin,
    WidgetPlugin,
)

logger = logging.getLogger(__name__)


@dataclass
class LegacyRegisterPlugin:
    """Compatibility wrapper for legacy plugins with register(registry) functions."""

    name: str
    register_fn: Callable[..., Any]
    version: str = "legacy"

    def activate(self, app: Any = None) -> None:
        from hafs.core.registry import agent_registry

        try:
            params = list(inspect.signature(self.register_fn).parameters.values())
        except (TypeError, ValueError):
            params = []

        if not params:
            self.register_fn()
            return

        if len(params) == 1:
            if params[0].name in {"app", "application"}:
                self.register_fn(app)
            else:
                self.register_fn(agent_registry)
            return

        self.register_fn(agent_registry, app)

    def deactivate(self) -> None:
        return None


class PluginLoader:
    """Loads plugins from entry points and directories.

    Plugins can be discovered from:
    1. Python entry points (hafs.plugins group)
    2. Plugin directories (Python files)

    Example:
        loader = PluginLoader(plugin_dirs=[Path("~/.config/hafs/plugins")])
        available = loader.discover_plugins()
        plugin = loader.load_plugin("my-plugin")
        if plugin:
            plugin.activate(app)
    """

    ENTRY_POINT_GROUP = "hafs.plugins"

    def __init__(self, plugin_dirs: list[Path] | None = None):
        """Initialize plugin loader.

        Args:
            plugin_dirs: Directories to search for plugins.
        """
        self._plugin_dirs = [Path(p).expanduser() for p in (plugin_dirs or [])]
        self._loaded: dict[str, HafsPlugin] = {}
        self._activated: set[str] = set()

    def discover_plugins(self) -> list[str]:
        """Discover available plugins.

        Returns:
            List of plugin names.
        """
        plugins: list[str] = []

        # From entry points
        try:
            from importlib.metadata import entry_points

            eps = entry_points(group=self.ENTRY_POINT_GROUP)
            for ep in eps:
                plugins.append(ep.name)
        except ImportError:
            pass

        # From plugin directories
        for plugin_dir in self._plugin_dirs:
            if plugin_dir.exists():
                for item in plugin_dir.iterdir():
                    if item.suffix == ".py" and not item.name.startswith("_"):
                        plugins.append(item.stem)
                    elif item.is_dir() and (item / "__init__.py").exists():
                        plugins.append(item.name)

        return list(set(plugins))

    def load_plugin(self, name: str) -> HafsPlugin | None:
        """Load a plugin by name.

        Args:
            name: Plugin name to load.

        Returns:
            Plugin instance, or None if not found.
        """
        if name in self._loaded:
            return self._loaded[name]

        # Try entry point first
        plugin = self._load_from_entry_point(name)
        if plugin:
            self._loaded[name] = plugin
            return plugin

        # Try plugin directories
        plugin = self._load_from_directory(name)
        if plugin:
            self._loaded[name] = plugin
            return plugin

        return None

    def _load_from_entry_point(self, name: str) -> HafsPlugin | None:
        """Load plugin from entry point.

        Args:
            name: Plugin name.

        Returns:
            Plugin instance or None.
        """
        try:
            from importlib.metadata import entry_points

            eps = entry_points(group=self.ENTRY_POINT_GROUP)
            for ep in eps:
                if ep.name == name:
                    plugin_obj = ep.load()
                    if isinstance(plugin_obj, type):
                        return plugin_obj()
                    if isinstance(plugin_obj, HafsPlugin):
                        return plugin_obj
                    if callable(plugin_obj):
                        return LegacyRegisterPlugin(name=name, register_fn=plugin_obj)
        except (ImportError, Exception):
            pass
        return None

    def _load_from_directory(self, name: str) -> HafsPlugin | None:
        """Load plugin from plugin directory.

        Args:
            name: Plugin name.

        Returns:
            Plugin instance or None.
        """
        import sys
        
        for plugin_dir in self._plugin_dirs:
            # Try as single file
            plugin_path = plugin_dir / f"{name}.py"
            if plugin_path.exists():
                return self._load_from_file(name, plugin_path)

            # Try as package
            package_path = plugin_dir / name / "__init__.py"
            if package_path.exists():
                # Add parent directory to path so relative imports work
                if str(plugin_dir) not in sys.path:
                    sys.path.insert(0, str(plugin_dir))
                return self._load_from_file(name, package_path)

        return None

    def _load_from_file(self, name: str, path: Path) -> HafsPlugin | None:
        """Load plugin from a Python file.

        Args:
            name: Plugin name.
            path: Path to the Python file.

        Returns:
            Plugin instance or None.
        """
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Look for Plugin class
                if hasattr(module, "Plugin"):
                    return module.Plugin()

                # Look for class matching plugin name
                class_name = "".join(word.capitalize() for word in name.split("-"))
                if hasattr(module, class_name):
                    return getattr(module, class_name)()

                # Look for any class implementing HafsPlugin
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and isinstance(attr(), HafsPlugin)
                        and attr_name not in ("HafsPlugin", "Protocol")
                    ):
                        return attr()

                if hasattr(module, "register") and callable(module.register):
                    return LegacyRegisterPlugin(name=name, register_fn=module.register)

        except Exception:
            pass
        return None

    def _register_plugin_components(self, plugin: HafsPlugin, app: Any | None) -> None:
        # Register backend if it's a backend plugin
        if isinstance(plugin, BackendPlugin):
            from hafs.backends.base import BackendRegistry

            BackendRegistry.register(plugin.get_backend_class())

        # Register parser if it's a parser plugin
        if isinstance(plugin, ParserPlugin):
            from hafs.core.parsers.registry import ParserRegistry

            parser_class = plugin.get_parser_class()
            # Prefer explicit name attribute, fall back to class name
            parser_name = getattr(parser_class, "name", None) or getattr(
                parser_class, "__name__", "custom_parser"
            )
            ParserRegistry.register(str(parser_name), parser_class)

        # Register widget if it's a widget plugin
        if isinstance(plugin, WidgetPlugin) and app is not None:
            if hasattr(app, "register_widget_plugin"):
                app.register_widget_plugin(plugin)

        # Register tool providers
        if isinstance(plugin, ToolPlugin):
            search_provider = plugin.get_search_provider()
            if search_provider:
                ToolRegistry.register_search_provider(search_provider)

            review_provider = plugin.get_review_provider()
            if review_provider:
                ToolRegistry.register_review_provider(review_provider)

        # Register external provider adapters for background agents
        if isinstance(plugin, IntegrationPlugin):
            issue_adapter = plugin.get_issue_tracker()
            if issue_adapter:
                agent_registry.register_adapter("issue_tracker", issue_adapter)

            review_adapter = plugin.get_code_review()
            if review_adapter:
                agent_registry.register_adapter("code_review", review_adapter)

            search_adapter = plugin.get_code_search()
            if search_adapter:
                agent_registry.register_adapter("code_search", search_adapter)

    def activate_plugin(self, name: str, app: Any | None = None) -> bool:
        """Activate a plugin.

        Args:
            name: Plugin name.
            app: HafsApp instance (optional for headless contexts).

        Returns:
            True if activated successfully.
        """
        if name in self._activated:
            return True

        plugin = self.load_plugin(name)
        if not plugin:
            return False

        if app is not None or isinstance(plugin, LegacyRegisterPlugin):
            try:
                plugin.activate(app)
            except Exception as exc:
                logger.warning("Plugin activation failed for %s: %s", name, exc)

        try:
            self._register_plugin_components(plugin, app)
        except Exception as exc:
            logger.warning("Plugin registration failed for %s: %s", name, exc)
            return False

        self._activated.add(name)
        return True

    def deactivate_plugin(self, name: str) -> bool:
        """Deactivate a plugin.

        Args:
            name: Plugin name.

        Returns:
            True if deactivated successfully.
        """
        if name not in self._activated:
            return True

        plugin = self._loaded.get(name)
        if not plugin:
            return False

        try:
            plugin.deactivate()
            self._activated.discard(name)
            return True
        except Exception:
            return False

    def get_plugin(self, name: str) -> HafsPlugin | None:
        """Get a loaded plugin by name.

        Args:
            name: Plugin name.

        Returns:
            Plugin instance or None.
        """
        return self._loaded.get(name)

    @property
    def loaded_plugins(self) -> list[str]:
        """Get list of loaded plugin names."""
        return list(self._loaded.keys())

    @property
    def activated_plugins(self) -> list[str]:
        """Get list of activated plugin names."""
        return list(self._activated)
