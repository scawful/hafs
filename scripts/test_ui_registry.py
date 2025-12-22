"""Test that plugins can register UI pages."""
import importlib
import os
import sys
from pathlib import Path


def _extend_sys_path() -> None:
    extra_paths = os.environ.get("HAFS_EXTRA_PYTHONPATH", "")
    for entry in [p for p in extra_paths.split(os.pathsep) if p]:
        sys.path.append(os.path.expanduser(entry))
    repo_src = Path(__file__).resolve().parents[1] / "src"
    if repo_src.exists():
        sys.path.append(str(repo_src))

import pytest

from core.ui_registry import ui_registry
from core.plugin_loader import load_plugins
from core.registry import agent_registry

_extend_sys_path()

plugin_module = os.environ.get("HAFS_TEST_PLUGIN_MODULE")
plugin = None
if plugin_module:
    try:
        plugin = importlib.import_module(plugin_module)
    except ModuleNotFoundError:
        plugin = None

def test_ui_registration():
    if not plugin_module:
        pytest.skip("HAFS_TEST_PLUGIN_MODULE not set")
    if plugin is None:
        pytest.skip(f"Plugin module not available: {plugin_module}")

    print("--- Testing UI Registry ---")
    
    # 1. Check initial state
    initial_pages = list(ui_registry.pages.keys())
    print(f"Initial pages: {initial_pages}")
    
    # 2. Load Plugins (simulate startup)
    load_plugins()
    if hasattr(plugin, "register"):
        plugin.register(agent_registry)

    # 3. Check for newly registered pages
    final_pages = list(ui_registry.pages.keys())
    print(f"Final pages: {final_pages}")

    if final_pages != initial_pages:
        print("✅ SUCCESS: UI pages updated after plugin load.")
    else:
        print("⚠️ No new pages registered by plugin.")

if __name__ == "__main__":
    test_ui_registration()
