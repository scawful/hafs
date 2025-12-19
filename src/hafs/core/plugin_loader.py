"""Plugin Loader for HAFS.

Discovers and loads agent and adapter plugins from installed packages.
"""
import importlib
import pkgutil
import sys
from hafs.core.registry import agent_registry
from hafs.agents.base import BaseAgent
from hafs.core.config import hafs_config

def load_plugins():
    """
    Discovers and loads all HAFS plugins.
    """
    # Add plugin directories to sys.path
    for plugin_dir in hafs_config.plugin_dirs:
        plugin_path = str(plugin_dir.expanduser())
        if plugin_path not in sys.path:
            sys.path.append(plugin_path)
            print(f"[PluginLoader] Added plugin dir: {plugin_path}")

    print(f"[PluginLoader] Configured plugins: {hafs_config.plugins}")
    print("[PluginLoader] Loading configured plugins...")
    
    # 1. Load explicitly configured plugins
    for plugin_name in hafs_config.plugins:
        try:
            # Try importing the package
            module = importlib.import_module(plugin_name)
            
            # Check for hafs_plugin submodule (standard convention)
            try:
                entry_point = importlib.import_module(f"{plugin_name}.hafs_plugin")
                if hasattr(entry_point, "register"):
                    entry_point.register(agent_registry)
                    print(f"[PluginLoader] Registered: {plugin_name}")
                    continue
            except ImportError as e:
                print(f"[PluginLoader] Failed to import {plugin_name}.hafs_plugin: {e}")
                pass

            # Check if top-level module has register
            if hasattr(module, "register"):
                module.register(agent_registry)
                print(f"[PluginLoader] Registered: {plugin_name}")
            else:
                print(f"[PluginLoader] {plugin_name} has no 'register' or 'hafs_plugin.register'")
                
        except Exception as e:
            print(f"[PluginLoader] Failed to load {plugin_name}: {e}")

    # 2. Dynamic Discovery (auto-discovery)
    print("[PluginLoader] scanning for 'hafs_plugin_*'...")
    for _, name, _ in pkgutil.iter_modules():
        if name.startswith("hafs_plugin"):
            try:
                module = importlib.import_module(name)
                if hasattr(module, "register"):
                    module.register(agent_registry)
                    print(f"[PluginLoader] Auto-registered: {name}")
            except Exception as e:
                print(f"[PluginLoader] Error auto-loading {name}: {e}")

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
                from hafs.agents.base import BaseAgent
                if isinstance(attribute, type) and issubclass(attribute, BaseAgent) and attribute is not BaseAgent:
                    print(f"[PluginLoader]   Found agent: {attribute.__name__}")
                    agent_registry.register_agent(attribute)
        except Exception as e:
            print(f"[PluginLoader]   Failed to scan {name}: {e}")
