"""Configuration loader with TOML support and merge capability."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from hafs.config.schema import HafsConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _expand_path(path: str | Path) -> Path:
    """Expand ~ and environment variables in path."""
    return Path(path).expanduser().resolve()


def load_config(
    config_path: Path | None = None,
    merge_user: bool = True,
) -> HafsConfig:
    """Load configuration with precedence.

    Priority (highest to lowest):
    1. Provided config_path
    2. ~/.config/hafs/config.toml (user)
    3. ./hafs.toml (project-local)
    4. Built-in defaults

    Args:
        config_path: Explicit path to config file.
        merge_user: Whether to merge user config from ~/.config/hafs/.

    Returns:
        Merged HafsConfig instance.
    """
    config_data: dict[str, Any] = {}

    # Project-local config
    local_path = Path("hafs.toml")
    if local_path.exists():
        with open(local_path, "rb") as f:
            config_data = _deep_merge(config_data, tomllib.load(f))

    # User config
    if merge_user:
        user_path = Path.home() / ".config" / "hafs" / "config.toml"
        if user_path.exists():
            with open(user_path, "rb") as f:
                config_data = _deep_merge(config_data, tomllib.load(f))

    # Explicit config path (highest precedence)
    if config_path and config_path.exists():
        with open(config_path, "rb") as f:
            config_data = _deep_merge(config_data, tomllib.load(f))

    # Expand paths in tracked_projects
    if "tracked_projects" in config_data:
        config_data["tracked_projects"] = [
            _expand_path(p) for p in config_data["tracked_projects"]
        ]

    # Expand paths in parser configs
    if "parsers" in config_data:
        for parser_name in ["gemini", "claude", "antigravity"]:
            if parser_name in config_data["parsers"]:
                parser_config = config_data["parsers"][parser_name]
                if "base_path" in parser_config and parser_config["base_path"]:
                    parser_config["base_path"] = _expand_path(parser_config["base_path"])

    return HafsConfig(**config_data)
