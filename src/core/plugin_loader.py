"""Plugin Loader for HAFS."""
import importlib
import logging
import pkgutil

from config.loader import load_config
from core.registry import agent_registry
from plugins.host import HeadlessPluginHost
from plugins.loader import PluginLoader

logger = logging.getLogger(__name__)

def load_plugins():
    """
    Discovers and loads all HAFS plugins.
    """
    config = load_config()
    plugin_loader = PluginLoader(plugin_dirs=list(config.plugins.plugin_dirs))
    host = HeadlessPluginHost(config)

    enabled = list(config.plugins.enabled_plugins)
    loaded = set()
    if enabled:
        logger.info("Loading configured plugins: %s", ", ".join(enabled))
        for plugin_name in enabled:
            if plugin_loader.activate_plugin(plugin_name, app=host):
                loaded.add(plugin_name)

    logger.info("Scanning for auto-discoverable plugins (hafs_plugin*).")
    for _, name, _ in pkgutil.iter_modules():
        if name.startswith("hafs_plugin") and name not in loaded:
            plugin_loader.activate_plugin(name, app=host)

def load_all_agents_from_package(package):
    """Dynamically loads all agent classes from a given package (recursively)."""
    if not hasattr(package, '__path__'):
        return

    print(f"[PluginLoader] Scanning package {package.__name__} for agents...")
    
    # We use a set to avoid double-processing
    for _, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            module = importlib.import_module(name)
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                # We import BaseAgent here to ensure we compare against the canonical one
                from agents.core.base import BaseAgent
                if isinstance(attribute, type) and issubclass(attribute, BaseAgent) and attribute is not BaseAgent:
                    print(f"[PluginLoader]   Found agent: {attribute.__name__}")
                    agent_registry.register_agent(attribute)
        except Exception as e:
            print(f"[PluginLoader]   Failed to scan {name}: {e}")
