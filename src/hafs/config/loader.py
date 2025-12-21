"""Configuration loader with TOML support and merge capability."""

from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path
from typing import Any

from hafs.config.schema import HafsConfig

logger = logging.getLogger(__name__)


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


def _parse_bool(value: str | None, default: bool = False) -> bool:
    """Parse a string to bool with a default."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _humanize_path(path: Path) -> str:
    """Return a user-friendly path string (use ~ when under home)."""
    try:
        resolved = path.expanduser().resolve()
    except Exception:
        return str(path)
    home = Path.home().resolve()
    try:
        relative = resolved.relative_to(home)
    except ValueError:
        return str(resolved)
    return f"~/{relative}"


def _escape_toml_string(value: str) -> str:
    """Escape a string for inline TOML usage."""
    compact = " ".join(value.splitlines()).strip()
    return compact.replace("\\", "\\\\").replace('"', '\\"')


def _discover_afs_projects() -> list[dict[str, str]]:
    """Discover AFS contexts and return project entries."""
    try:
        from hafs.core.afs.discovery import discover_projects
    except Exception:
        return []

    projects = []
    seen: set[str] = set()
    for context in discover_projects():
        project_root = context.path.parent
        name = context.project_name or project_root.name
        if not name or name in seen:
            continue
        entry = {"name": name, "path": _humanize_path(project_root)}
        description = getattr(context.metadata, "description", "").strip()
        if description:
            entry["description"] = _escape_toml_string(description)
        projects.append(entry)
        seen.add(name)

    projects.sort(key=lambda item: item["name"].lower())
    return projects


def _render_local_config(projects: list[dict[str, str]]) -> str:
    """Render a minimal local config with auto-discovered projects."""
    lines = [
        "# Local HAFS configuration (auto-created).",
        "# This file overrides repo defaults in ./hafs.toml.",
        "# Safe to edit on this machine.",
        "",
        "[plugins]",
        "# enabled_plugins = [\"my_hafs_plugin\"]",
        "# plugin_dirs = [\"~/Code/hafs-plugins/src\"]",
        "",
    ]

    if projects:
        lines.append("# Auto-discovered AFS projects")
        for project in projects:
            lines.append("")
            lines.append("[[projects]]")
            lines.append(f"name = \"{_escape_toml_string(project['name'])}\"")
            lines.append(f"path = \"{_escape_toml_string(project['path'])}\"")
            if "description" in project:
                lines.append(f"description = \"{project['description']}\"")
    else:
        lines.append("# No AFS projects discovered yet.")
        lines.append("# Run `hafs afs init` in a repo to create one.")

    return "\n".join(lines) + "\n"


def _ensure_local_config(
    config_path: Path | None,
    env_config: str | None,
    local_path: Path,
) -> None:
    """Create a local config file with discovered projects if missing."""
    if config_path is not None or env_config:
        return
    if local_path.exists():
        return

    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        projects = _discover_afs_projects()
        local_path.write_text(_render_local_config(projects), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to create local config: %s", exc)


def load_config(
    config_path: Path | None = None,
    merge_user: bool = True,
) -> HafsConfig:
    """Load configuration with precedence.

    Priority (highest to lowest):
    1. Provided config_path
    2. ~/.config/hafs/config.toml (local user overrides)
    3. ./hafs.toml (project defaults)
    4. ~/.context/hafs_config.toml (legacy fallback)
    5. Built-in defaults (schema)

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

    prefer_user = _parse_bool(os.environ.get("HAFS_PREFER_USER_CONFIG"), default=True)
    prefer_repo = _parse_bool(os.environ.get("HAFS_PREFER_REPO_CONFIG"))
    if prefer_repo:
        prefer_user = False

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

    # User config (preferred by default)
    if merge_user:
        user_path = Path.home() / ".config" / "hafs" / "config.toml"
        _ensure_local_config(config_path, env_config, user_path)
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
