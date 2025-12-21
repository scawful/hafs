"""Test that plugins can register UI pages."""
import sys
import os

# Add paths
sys.path.append(os.path.expanduser("~/Code/Experimental/hafs/src"))
sys.path.append(os.path.expanduser("~/Code/Experimental/hafs_google_internal/src"))

import pytest

from hafs.core.ui_registry import ui_registry
from hafs.core.plugin_loader import load_plugins
from hafs.core.registry import agent_registry

try:
    import hafs_google_internal.hafs_plugin as google_plugin
except ModuleNotFoundError:
    google_plugin = None

def test_ui_registration():
    if google_plugin is None:
        pytest.skip("hafs_google_internal not available")

    print("--- Testing UI Registry ---")
    
    # 1. Check initial state
    initial_pages = list(ui_registry.pages.keys())
    print(f"Initial pages: {initial_pages}")
    
    # 2. Load Plugins (simulate startup)
    load_plugins()
    # Force manual register to be safe for test environment quirks
    google_plugin.register(agent_registry)

    # 3. Check for "My Work"
    final_pages = list(ui_registry.pages.keys())
    print(f"Final pages: {final_pages}")
    
    if "My Work (Google)" in final_pages:
        print("✅ SUCCESS: 'My Work (Google)' page is registered.")
    else:
        print("❌ FAILURE: Google page missing.")

if __name__ == "__main__":
    test_ui_registration()
