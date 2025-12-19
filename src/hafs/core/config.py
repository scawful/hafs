"""Configuration loader for HAFS.

Reads settings from the central `~/.context/hafs_config.toml` file.
"""
import os
from pathlib import Path
try:
    import tomllib as toml
except ImportError:
    class MockToml:
        def load(self, f): return {}
    toml = MockToml()

from typing import Dict, Any, List

# --- Core Paths ---
HOME = Path.home()

# --- Main Config Class ---

class HAFSConfig:
    """A singleton class to hold all HAFS configuration."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HAFSConfig, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load settings from TOML file, with fallbacks."""
        config_path = HOME / ".context" / "hafs_config.toml"
        self._data: Dict[str, Any] = {}
        if config_path.exists():
            try:
                with open(config_path, "rb") as f:
                    self._data = toml.load(f)
            except Exception as e:
                print(f"Warning: Could not parse hafs_config.toml: {e}")
        
        # Helper for nested gets
        self._get = lambda key, default: self._data.get(key, default)

    # --- Core Properties ---
    @property
    def context_root(self) -> Path:
        path_str = self._get("core", {}).get("context_root", "~/.context")
        return Path(path_str).expanduser()
    
    @property
    def agent_workspaces_dir(self) -> Path:
        path_str = self._get("core", {}).get("agent_workspaces", "~/AgentWorkspaces")
        return Path(path_str).expanduser()

    # --- LLM Properties ---
    @property
    def aistudio_api_key(self) -> str:
        return os.environ.get("AISTUDIO_API_KEY") or self._get("llm", {}).get("aistudio_api_key", "")

    @property
    def plugins(self) -> List[str]:
        return self._get("core", {}).get("plugins", [])

# Create a global instance for easy access
hafs_config = HAFSConfig()
swarm_config = hafs_config # Alias


# --- Legacy/Convenience Top-Level Constants ---
HOME = Path.home()
CONTEXT_ROOT = hafs_config.context_root
KNOWLEDGE_DIR = CONTEXT_ROOT / "knowledge"
VERIFIED_DIR = KNOWLEDGE_DIR / "verified"
DISCOVERED_DIR = KNOWLEDGE_DIR / "discovered"
MEMORY_DIR = CONTEXT_ROOT / "memory"
METRICS_DIR = CONTEXT_ROOT / "metrics"
LOGS_DIR = CONTEXT_ROOT / "background_agent" / "reports"
REPORTS_DIR = LOGS_DIR # Alias
SCRATCHPAD_DIR = CONTEXT_ROOT / "scratchpad"
BRIEFINGS_DIR = CONTEXT_ROOT / "background_agent" / "briefings"
KNOWLEDGE_GRAPH_FILE = MEMORY_DIR / "knowledge_graph.json"

QUOTA_USAGE_FILE = METRICS_DIR / "quota_usage.json"
COGNITIVE_STATE_FILE = MEMORY_DIR / "cognitive_state.json"
AGENT_WORKSPACES_DIR = hafs_config.agent_workspaces_dir