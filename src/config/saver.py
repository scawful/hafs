"""Configuration saver for persisting HAFS config changes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomli_w
except ImportError:
    tomli_w = None  # type: ignore[assignment]

from config.schema import HafsConfig


def _config_to_dict(config: HafsConfig) -> dict[str, Any]:
    """Convert HafsConfig to a TOML-compatible dictionary.

    Args:
        config: HafsConfig instance to convert.

    Returns:
        Dictionary suitable for TOML serialization.
    """
    data = config.model_dump(mode="python")

    # Convert Path objects to strings for TOML serialization
    if "tracked_projects" in data:
        data["tracked_projects"] = [str(p) for p in data["tracked_projects"]]

    if "parsers" in data:
        for parser_name in ["gemini", "claude", "antigravity"]:
            if parser_name in data["parsers"]:
                parser = data["parsers"][parser_name]
                if "base_path" in parser and parser["base_path"]:
                    parser["base_path"] = str(parser["base_path"])

    if "plugins" in data:
        if "plugin_dirs" in data["plugins"]:
            data["plugins"]["plugin_dirs"] = [
                str(p) for p in data["plugins"]["plugin_dirs"]
            ]

    if "synergy" in data:
        if "profile_storage" in data["synergy"]:
            data["synergy"]["profile_storage"] = str(data["synergy"]["profile_storage"])

    # Convert workspace directory paths
    if "general" in data:
        if "context_root" in data["general"]:
            data["general"]["context_root"] = str(data["general"]["context_root"])
        if "agent_workspaces_dir" in data["general"]:
            data["general"]["agent_workspaces_dir"] = str(
                data["general"]["agent_workspaces_dir"]
            )
        if "workspace_directories" in data["general"]:
            data["general"]["workspace_directories"] = [
                {
                    "path": str(ws["path"]),
                    "name": ws.get("name"),
                    "recursive": ws.get("recursive", True),
                }
                for ws in data["general"]["workspace_directories"]
            ]

    if "plugins" in data:
        plugin_dirs = data["plugins"].get("plugin_dirs", [])
        data["plugins"]["plugin_dirs"] = [str(p) for p in plugin_dirs]

    if "projects" in data:
        data["projects"] = [
            {
                "name": project["name"],
                "path": str(project["path"]),
                "kind": project.get("kind", "general"),
                "tags": project.get("tags", []),
                "tooling_profile": project.get("tooling_profile"),
                "knowledge_roots": [str(p) for p in project.get("knowledge_roots", [])],
                "enabled": project.get("enabled", True),
                "description": project.get("description", ""),
            }
            for project in data["projects"]
        ]

    # Convert AFS directory configs to dicts (convert enum to string)
    if "afs_directories" in data:
        data["afs_directories"] = [
            {
                "name": d["name"],
                "policy": d["policy"].value if hasattr(d["policy"], "value") else str(d["policy"]),
                "description": d.get("description", ""),
            }
            for d in data["afs_directories"]
        ]

    # Convert backend configs (exclude None values)
    if "backends" in data:
        new_backends = []
        for b in data["backends"]:
            backend_dict = {
                "name": b["name"],
                "enabled": b.get("enabled", True),
                "command": b.get("command", []),
                "env": b.get("env", {}),
            }
            # Only include working_dir if it's not None
            if b.get("working_dir"):
                backend_dict["working_dir"] = str(b["working_dir"])
            new_backends.append(backend_dict)
        data["backends"] = new_backends

    return data


def save_config(config: HafsConfig, path: Path | None = None) -> None:
    """Save configuration to a TOML file.

    Args:
        config: HafsConfig instance to save.
        path: Path to save to. Defaults to ~/.config/hafs/config.toml.

    Raises:
        ImportError: If tomli_w is not installed.
        IOError: If unable to write the config file.
    """
    if tomli_w is None:
        raise ImportError(
            "tomli_w is required to save config. Install with: pip install tomli-w"
        )

    if path is None:
        path = Path.home() / ".config" / "hafs" / "config.toml"

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert config to dict
    config_dict = _config_to_dict(config)

    # Write to TOML file
    with open(path, "wb") as f:
        tomli_w.dump(config_dict, f)


def save_afs_policies(
    config: HafsConfig,
    context_path: Path | None = None,
) -> None:
    """Save AFS policies to project metadata.json.

    Updates the policy field in the project's .context/metadata.json file
    to reflect the current AFS directory configurations.

    Args:
        config: HafsConfig with current AFS directory policies.
        context_path: Path to .context directory. Defaults to ./.context.

    Raises:
        FileNotFoundError: If .context/metadata.json doesn't exist.
        IOError: If unable to write metadata file.
    """
    import json

    if context_path is None:
        context_path = Path(".") / ".context"

    metadata_path = context_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata.json found at {metadata_path}")

    # Load existing metadata
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    # Build policy dict from config
    policy_dict: dict[str, list[str]] = {
        "read_only": [],
        "writable": [],
        "executable": [],
    }

    for dir_config in config.afs_directories:
        policy_key = dir_config.policy.value
        if policy_key in policy_dict:
            policy_dict[policy_key].append(dir_config.name)

    # Update metadata
    metadata["policy"] = policy_dict

    # Write back to file
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
