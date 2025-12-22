"""AFS project discovery utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterator

from models.afs import ContextRoot, MountPoint, MountType, ProjectMetadata

ProjectProvider = Callable[[], list[ContextRoot]]


class DiscoveryRegistry:
    """Registry for additional project discovery providers."""

    _providers: list[ProjectProvider] = []

    @classmethod
    def register(cls, provider: ProjectProvider) -> None:
        """Register a new project provider function."""
        cls._providers.append(provider)

    @classmethod
    def get_providers(cls) -> list[ProjectProvider]:
        """Get all registered providers."""
        return cls._providers


def find_context_root(start_path: Path = Path(".")) -> Path | None:
    """Find the nearest .context directory.

    Searches from start_path upward through parent directories. If not found,
    also checks for a .hafs_context_link file to resolve the path.

    Args:
        start_path: Directory to start searching from.

    Returns:
        Path to .context directory, or None if not found.
    """
    current = start_path.resolve()

    # 1. Standard upward search for .context
    search_path = current
    while search_path != search_path.parent:
        context_path = search_path / ".context"
        if context_path.exists() and context_path.is_dir():
            return context_path
        search_path = search_path.parent

    # 2. If not found, upward search for .hafs_context_link
    search_path = current
    while search_path != search_path.parent:
        link_file = search_path / ".hafs_context_link"
        if link_file.is_file():
            try:
                linked_path_str = link_file.read_text().strip()
                linked_path = Path(linked_path_str).resolve()
                if linked_path.is_dir():
                    return linked_path
                else:
                    # The path in the link is invalid, stop here.
                    return None
            except Exception:
                # Failed to read or resolve the path, stop here.
                return None
        search_path = search_path.parent

    return None


def discover_projects(
    search_paths: list[Path] | None = None,
    max_depth: int = 3,
) -> list[ContextRoot]:
    """Discover AFS-enabled projects.

    Searches through specified paths for directories containing .context.
    Also queries any registered discovery providers.

    Args:
        search_paths: Paths to search. Defaults to home directory subdirs.
        max_depth: Maximum directory depth to search.

    Returns:
        List of ContextRoot objects for discovered projects.
    """
    if search_paths is None:
        home = Path.home()
        search_paths = [
            home / "Code",
            home / "Projects",
            home / "Developer",
            home / "dev",
        ]

    projects: list[ContextRoot] = []
    seen_paths: set[Path] = set()

    # 1. Standard directory search
    for search_path in search_paths:
        if not search_path.exists():
            continue

        for context_path in _find_context_dirs(search_path, max_depth):
            resolved_path = context_path.resolve()
            if resolved_path in seen_paths:
                continue
            seen_paths.add(resolved_path)

            project = _load_context_root(context_path)
            if project:
                projects.append(project)

    # 2. Query registered providers
    for provider in DiscoveryRegistry.get_providers():
        try:
            extra_projects = provider()
            for project in extra_projects:
                resolved_path = project.path.resolve()
                if resolved_path in seen_paths:
                    continue
                seen_paths.add(resolved_path)
                projects.append(project)
        except Exception:
            # Ignore provider errors to prevent crashing the UI
            pass

    # Sort by project name
    projects.sort(key=lambda p: p.project_name.lower())
    return projects


def _find_context_dirs(root: Path, max_depth: int, current_depth: int = 0) -> Iterator[Path]:
    """Recursively find .context directories.

    Args:
        root: Directory to search.
        max_depth: Maximum depth to search.
        current_depth: Current recursion depth.

    Yields:
        Paths to .context directories.
    """
    if current_depth > max_depth:
        return

    try:
        for entry in root.iterdir():
            if entry.name == ".context" and entry.is_dir():
                yield entry
            elif entry.is_dir() and not entry.name.startswith("."):
                yield from _find_context_dirs(entry, max_depth, current_depth + 1)
    except PermissionError:
        pass


def _load_context_root(context_path: Path) -> ContextRoot | None:
    """Load a ContextRoot from a .context directory.

    Args:
        context_path: Path to .context directory.

    Returns:
        ContextRoot object or None on error.
    """
    return load_context_root(context_path)


def load_context_root(context_path: Path) -> ContextRoot | None:
    """Load a ContextRoot from a .context directory.

    Args:
        context_path: Path to .context directory.

    Returns:
        ContextRoot object or None on error.
    """
    try:
        # Load metadata
        metadata_path = context_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                data = json.load(f)
                # Handle legacy metadata with empty created_at
                if "created_at" in data and not data["created_at"]:
                    del data["created_at"]
                metadata = ProjectMetadata(**data)
        else:
            metadata = ProjectMetadata()

        # Scan mounts
        mounts: dict[MountType, list[MountPoint]] = {}

        for mt in MountType:
            subdir = context_path / mt.value
            if not subdir.exists():
                continue

            mount_list: list[MountPoint] = []
            for item in subdir.iterdir():
                if item.name == ".keep":
                    continue

                source = item.resolve() if item.is_symlink() else item
                mount_list.append(
                    MountPoint(
                        name=item.name,
                        source=source,
                        mount_type=mt,
                        is_symlink=item.is_symlink(),
                    )
                )

            if mount_list:
                mounts[mt] = mount_list

        return ContextRoot(
            path=context_path,
            project_name=context_path.parent.name,
            metadata=metadata,
            mounts=mounts,
        )
    except (OSError, json.JSONDecodeError):
        return None


def get_project_stats(projects: list[ContextRoot]) -> dict[str, int]:
    """Get aggregate statistics for projects.

    Args:
        projects: List of ContextRoot objects.

    Returns:
        Dictionary with stats like total_projects, total_mounts, etc.
    """
    total_mounts = 0
    mounts_by_type: dict[str, int] = {mt.value: 0 for mt in MountType}

    for project in projects:
        for mt, mount_list in project.mounts.items():
            total_mounts += len(mount_list)
            mounts_by_type[mt.value] += len(mount_list)

    return {
        "total_projects": len(projects),
        "total_mounts": total_mounts,
        **mounts_by_type,
    }
