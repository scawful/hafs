"""Prompt loader for configurable templates."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

_PROMPT_CACHE: Optional[dict[str, Any]] = None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _get_plugin_prompt_paths() -> list[Path]:
    paths: list[Path] = []

    env_paths = os.environ.get("HAFS_PLUGIN_PROMPT_PATHS", "")
    if env_paths:
        for raw in env_paths.split(os.pathsep):
            raw = raw.strip()
            if raw:
                paths.append(Path(raw).expanduser())

    scawful_root = os.environ.get("HAFS_SCAWFUL_ROOT")
    if scawful_root:
        paths.append(Path(scawful_root).expanduser() / "config" / "prompts.toml")

    plugins_dir = Path.home() / ".config" / "hafs" / "plugins"
    if plugins_dir.exists():
        for plugin_dir in plugins_dir.iterdir():
            candidate = plugin_dir / "config" / "prompts.toml"
            if candidate.exists():
                paths.append(candidate)

    # Preserve order, dedupe
    unique_paths = []
    seen = set()
    for path in paths:
        if path not in seen:
            unique_paths.append(path)
            seen.add(path)
    return unique_paths


def load_prompts() -> dict[str, Any]:
    """Load prompt templates with user overrides."""
    global _PROMPT_CACHE
    if _PROMPT_CACHE is not None:
        return _PROMPT_CACHE

    repo_path = Path("config/prompts.toml")
    user_path = Path.home() / ".config" / "hafs" / "prompts.toml"

    prompts = _load_toml(repo_path)
    for plugin_path in _get_plugin_prompt_paths():
        plugin_prompts = _load_toml(plugin_path)
        if plugin_prompts:
            prompts = _deep_merge(prompts, plugin_prompts)
    overrides = _load_toml(user_path)
    if overrides:
        prompts = _deep_merge(prompts, overrides)

    _PROMPT_CACHE = prompts
    return prompts


def get_prompt(key: str, default: Optional[str] = None) -> str:
    """Return a prompt template by dotted key."""
    data = load_prompts()
    current: Any = data
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default or ""
        current = current[part]
    if isinstance(current, str):
        return current
    return default or ""
