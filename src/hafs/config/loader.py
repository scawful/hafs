"""Configuration loader with TOML support and merge capability."""

from __future__ import annotations

import os
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
    2. ./hafs.toml (project-local)
    3. ~/.config/hafs/config.toml (user)
    4. Built-in defaults

    Note: project lists are merged with user config taking precedence so that
    local project definitions override repo templates.

    Args:
        config_path: Explicit path to config file.
        merge_user: Whether to merge user config from ~/.config/hafs/.

    Returns:
        Merged HafsConfig instance.
    """
    env_config = os.environ.get("HAFS_CONFIG_PATH")
    if config_path is None and env_config:
        config_path = Path(env_config).expanduser()

    prefer_user = os.environ.get("HAFS_PREFER_USER_CONFIG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    config_data: dict[str, Any] = {}
    legacy_mapped: dict[str, Any] = {}
    user_raw: dict[str, Any] = {}
    local_raw: dict[str, Any] = {}
    explicit_raw: dict[str, Any] = {}

    # Legacy config (lowest precedence)
    legacy_path = Path.home() / ".context" / "hafs_config.toml"
    if legacy_path.exists():
        try:
            with open(legacy_path, "rb") as f:
                legacy_raw = tomllib.load(f)
            legacy_mapped = {}
            core_section = legacy_raw.get("core", {})
            if isinstance(core_section, dict):
                general = legacy_mapped.setdefault("general", {})
                if "context_root" in core_section:
                    general["context_root"] = core_section["context_root"]
                if "agent_workspaces" in core_section:
                    general["agent_workspaces_dir"] = core_section["agent_workspaces"]

                if "plugins" in core_section:
                    plugins = legacy_mapped.setdefault("plugins", {})
                    plugins["enabled_plugins"] = core_section.get("plugins", [])

            config_data = _deep_merge(config_data, legacy_mapped)
        except Exception:
            pass

    # User config (default precedence below project-local)
    if merge_user:
        user_path = Path.home() / ".config" / "hafs" / "config.toml"
        if user_path.exists():
            with open(user_path, "rb") as f:
                user_raw = tomllib.load(f)

    # Project-local config (default precedence above user)
    local_path = Path("hafs.toml")
    if local_path.exists():
        with open(local_path, "rb") as f:
            local_raw = tomllib.load(f)

    if prefer_user:
        config_data = _deep_merge(config_data, local_raw)
        config_data = _deep_merge(config_data, user_raw)
    else:
        config_data = _deep_merge(config_data, user_raw)
        config_data = _deep_merge(config_data, local_raw)

    # Explicit config path (highest precedence)
    if config_path and config_path.exists():
        with open(config_path, "rb") as f:
            explicit_raw = tomllib.load(f)
        config_data = _deep_merge(config_data, explicit_raw)

    # Merge project lists with user config overriding project-local entries.
    project_sources = [
        legacy_mapped.get("projects", []),
        local_raw.get("projects", []),
        user_raw.get("projects", []),
        explicit_raw.get("projects", []),
    ]
    merged_projects = []
    seen: dict[str, dict[str, Any]] = {}
    for source in project_sources:
        for project in source or []:
            name = project.get("name")
            if not name:
                continue
            seen[name] = project
    if seen:
        merged_projects = list(seen.values())
        config_data["projects"] = merged_projects

    # Expand paths in tracked_projects
    if "tracked_projects" in config_data:
        config_data["tracked_projects"] = [
            _expand_path(p) for p in config_data["tracked_projects"]
        ]
    elif "plugins" in config_data and "tracked_projects" in config_data["plugins"]:
        # Legacy location for tracked_projects in older config files.
        config_data["tracked_projects"] = [
            _expand_path(p) for p in config_data["plugins"]["tracked_projects"]
        ]

    # Expand paths in parser configs
    if "parsers" in config_data:
        for parser_name in ["gemini", "claude", "antigravity"]:
            if parser_name in config_data["parsers"]:
                parser_config = config_data["parsers"][parser_name]
                if "base_path" in parser_config and parser_config["base_path"]:
                    parser_config["base_path"] = _expand_path(parser_config["base_path"])

    # Expand paths in plugins config
    if "plugins" in config_data:
        if "plugin_dirs" in config_data["plugins"]:
            config_data["plugins"]["plugin_dirs"] = [
                _expand_path(p) for p in config_data["plugins"]["plugin_dirs"]
            ]

    # Expand paths in synergy config
    if "synergy" in config_data:
        if "profile_storage" in config_data["synergy"]:
            config_data["synergy"]["profile_storage"] = _expand_path(
                config_data["synergy"]["profile_storage"]
            )

    # Expand paths in workspace directories
    if "general" in config_data:
        if "context_root" in config_data["general"]:
            config_data["general"]["context_root"] = _expand_path(
                config_data["general"]["context_root"]
            )
        if "agent_workspaces_dir" in config_data["general"]:
            config_data["general"]["agent_workspaces_dir"] = _expand_path(
                config_data["general"]["agent_workspaces_dir"]
            )
        if "python_executable" in config_data["general"]:
            python_exec = config_data["general"]["python_executable"]
            if isinstance(python_exec, str) and python_exec.startswith("~"):
                config_data["general"]["python_executable"] = _expand_path(python_exec)
        if "workspace_directories" in config_data["general"]:
            for ws_dir in config_data["general"]["workspace_directories"]:
                if "path" in ws_dir:
                    ws_dir["path"] = _expand_path(ws_dir["path"])

    # Expand paths in project configs
    if "projects" in config_data:
        for project in config_data["projects"]:
            if "path" in project:
                project["path"] = _expand_path(project["path"])
            if "knowledge_roots" in project:
                project["knowledge_roots"] = [
                    _expand_path(p) for p in project["knowledge_roots"]
                ]

    return HafsConfig(**config_data)
