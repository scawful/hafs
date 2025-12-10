"""AFS (Agentic File System) manager for context directories."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from hafs.config.schema import HafsConfig
from hafs.models.afs import ContextRoot, MountPoint, MountType, ProjectMetadata


class AFSManager:
    """Manages AFS (.context) directories.

    Provides operations for initializing, mounting, listing, and cleaning
    AFS context directories.
    """

    CONTEXT_ROOT = ".context"
    METADATA_FILE = "metadata.json"

    def __init__(self, config: HafsConfig):
        """Initialize manager with configuration.

        Args:
            config: HAFS configuration object.
        """
        self.config = config
        self._directories = {d.name: d for d in config.afs_directories}

    def init(self, path: Path = Path("."), force: bool = False) -> ContextRoot:
        """Initialize AFS in the given directory.

        Creates the .context directory structure with all subdirectories,
        .keep files for Git compatibility, and metadata.json.

        Args:
            path: Directory to initialize AFS in.
            force: If True, overwrite existing AFS.

        Returns:
            ContextRoot object representing the initialized AFS.

        Raises:
            FileExistsError: If AFS already exists and force is False.
        """
        context_path = path.resolve() / self.CONTEXT_ROOT

        if context_path.exists() and not force:
            raise FileExistsError(f"AFS already exists at {context_path}")

        # Create root
        context_path.mkdir(exist_ok=True)

        # Create subdirectories
        for dir_config in self.config.afs_directories:
            subdir = context_path / dir_config.name
            subdir.mkdir(exist_ok=True)
            (subdir / ".keep").touch()

        # Create metadata
        metadata = ProjectMetadata(
            created_at=datetime.now(),
            agents=[],
            description=f"AFS for {path.resolve().name}",
        )

        metadata_path = context_path / self.METADATA_FILE
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                metadata.model_dump(mode="json"),
                f,
                indent=2,
                default=str,
            )

        return ContextRoot(
            path=context_path,
            project_name=path.resolve().name,
            metadata=metadata,
        )

    def mount(
        self,
        source: Path,
        mount_type: MountType,
        alias: Optional[str] = None,
        context_path: Optional[Path] = None,
    ) -> MountPoint:
        """Mount a resource into the AFS.

        Creates a symlink from the AFS directory to the source.

        Args:
            source: Path to the file or directory to mount.
            mount_type: Type of mount (memory, knowledge, tools, etc.).
            alias: Optional custom name for the mount point.
            context_path: Optional custom .context path.

        Returns:
            MountPoint object representing the mount.

        Raises:
            FileNotFoundError: If source doesn't exist.
            FileExistsError: If mount point already exists.
        """
        if context_path is None:
            context_path = Path(".") / self.CONTEXT_ROOT

        source = source.resolve()
        if not source.exists():
            raise FileNotFoundError(f"Source {source} does not exist")

        alias = alias or source.name
        dest = context_path / mount_type.value / alias

        if dest.exists():
            raise FileExistsError(
                f"Mount point '{alias}' already exists in {mount_type.value}"
            )

        # Create symlink
        dest.symlink_to(source)

        return MountPoint(
            name=alias,
            source=source,
            mount_type=mount_type,
            is_symlink=True,
        )

    def unmount(
        self,
        alias: str,
        mount_type: MountType,
        context_path: Optional[Path] = None,
    ) -> bool:
        """Remove a mount point.

        Args:
            alias: Name of the mount point to remove.
            mount_type: Type of mount.
            context_path: Optional custom .context path.

        Returns:
            True if mount was removed, False if it didn't exist.
        """
        if context_path is None:
            context_path = Path(".") / self.CONTEXT_ROOT

        mount_path = context_path / mount_type.value / alias
        if mount_path.exists() or mount_path.is_symlink():
            mount_path.unlink()
            return True
        return False

    def list(self, context_path: Optional[Path] = None) -> ContextRoot:
        """List current AFS structure.

        Args:
            context_path: Optional custom .context path.

        Returns:
            ContextRoot object with all mounts.

        Raises:
            FileNotFoundError: If no AFS is initialized.
        """
        if context_path is None:
            context_path = Path(".") / self.CONTEXT_ROOT

        context_path = context_path.resolve()
        if not context_path.exists():
            raise FileNotFoundError("No AFS initialized")

        # Load metadata
        metadata_path = context_path / self.METADATA_FILE
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                metadata = ProjectMetadata(**json.load(f))
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

            mounts[mt] = mount_list

        return ContextRoot(
            path=context_path,
            project_name=context_path.parent.name,
            metadata=metadata,
            mounts=mounts,
        )

    def clean(self, context_path: Optional[Path] = None) -> None:
        """Remove the AFS directory.

        This only removes symlinks, not the actual files they point to.

        Args:
            context_path: Optional custom .context path.
        """
        if context_path is None:
            context_path = Path(".") / self.CONTEXT_ROOT

        if context_path.exists():
            shutil.rmtree(context_path)

    def update_metadata(
        self,
        context_path: Optional[Path] = None,
        description: Optional[str] = None,
        agents: Optional[list[str]] = None,
    ) -> ProjectMetadata:
        """Update AFS metadata.

        Args:
            context_path: Optional custom .context path.
            description: New description (if provided).
            agents: New agents list (if provided).

        Returns:
            Updated ProjectMetadata object.

        Raises:
            FileNotFoundError: If no AFS is initialized.
        """
        if context_path is None:
            context_path = Path(".") / self.CONTEXT_ROOT

        metadata_path = context_path / self.METADATA_FILE
        if not metadata_path.exists():
            raise FileNotFoundError("No AFS initialized")

        with open(metadata_path, encoding="utf-8") as f:
            metadata = ProjectMetadata(**json.load(f))

        if description is not None:
            metadata = metadata.model_copy(update={"description": description})
        if agents is not None:
            metadata = metadata.model_copy(update={"agents": agents})

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata.model_dump(mode="json"), f, indent=2, default=str)

        return metadata
