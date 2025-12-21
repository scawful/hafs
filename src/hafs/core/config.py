"""Configuration loader for HAFS.

Wraps the unified `hafs.config.loader` configuration while keeping legacy
properties for older call sites.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List

from hafs.config.loader import load_config

# --- Core Paths ---
HOME = Path.home()


class HAFSConfig:
    """A singleton wrapper for unified HAFS configuration."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HAFSConfig, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> None:
        self._config = load_config()

    def reload(self) -> None:
        self._load_config()

    @property
    def context_root(self) -> Path:
        return self._config.general.context_root

    @property
    def agent_workspaces_dir(self) -> Path:
        return self._config.general.agent_workspaces_dir

    @property
    def aistudio_api_key(self) -> str:
        env_key = os.environ.get("AISTUDIO_API_KEY")
        if env_key:
            return env_key
        legacy_path = Path.home() / ".context" / "hafs_config.toml"
        if legacy_path.exists():
            try:
                import tomllib

                data = tomllib.loads(legacy_path.read_text())
                return data.get("llm", {}).get("aistudio_api_key", "")
            except Exception:
                return ""
        return ""

    @property
    def plugins(self) -> List[str]:
        return list(self._config.plugins.enabled_plugins)

    @property
    def plugin_dirs(self) -> List[Path]:
        return list(self._config.plugins.plugin_dirs)

    @property
    def llamacpp(self):
        return self._config.llamacpp

    @property
    def username(self) -> str:
        return os.environ.get("USER", "unknown")

    @property
    def context_agents(self):
        return self._config.context_agents

    @property
    def embedding_daemon(self):
        return self._config.embedding_daemon

    def get_backend_config(self, name: str) -> Any:
        """Get configuration for a specific backend."""
        return self._config.get_backend_config(name)


# Create a global instance for easy access
hafs_config = HAFSConfig()
swarm_config = hafs_config


# --- Legacy/Convenience Top-Level Constants ---
HOME = Path.home()
CONTEXT_ROOT = hafs_config.context_root
KNOWLEDGE_DIR = CONTEXT_ROOT / "knowledge"
VERIFIED_DIR = KNOWLEDGE_DIR / "verified"
DISCOVERED_DIR = KNOWLEDGE_DIR / "discovered"
MEMORY_DIR = CONTEXT_ROOT / "memory"
METRICS_DIR = CONTEXT_ROOT / "metrics"
LOGS_DIR = CONTEXT_ROOT / "background_agent" / "reports"
REPORTS_DIR = LOGS_DIR
SCRATCHPAD_DIR = CONTEXT_ROOT / "scratchpad"
BRIEFINGS_DIR = CONTEXT_ROOT / "background_agent" / "briefings"
KNOWLEDGE_GRAPH_FILE = MEMORY_DIR / "knowledge_graph.json"

QUOTA_USAGE_FILE = METRICS_DIR / "quota_usage.json"
COGNITIVE_STATE_FILE = MEMORY_DIR / "cognitive_state.json"
AGENT_WORKSPACES_DIR = hafs_config.agent_workspaces_dir
