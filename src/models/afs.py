"""AFS (Agentic File System) data models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class MountType(str, Enum):
    """Type of mount point in AFS."""

    MEMORY = "memory"
    KNOWLEDGE = "knowledge"
    TOOLS = "tools"
    SCRATCHPAD = "scratchpad"
    HISTORY = "history"


class MountPoint(BaseModel):
    """A mounted resource in an AFS directory."""

    name: str
    source: Path
    mount_type: MountType
    is_symlink: bool = True

    model_config = ConfigDict(frozen=True)


class ProjectMetadata(BaseModel):
    """Metadata for an AFS-enabled project."""

    created_at: datetime = Field(default_factory=datetime.now)
    agents: list[str] = Field(default_factory=list)
    description: str = ""
    directories: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of directory roles (memory, tools, etc.) to on-disk names.",
    )
    policy: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "read_only": ["knowledge", "memory"],
            "writable": ["scratchpad"],
            "executable": ["tools"],
        }
    )


class ContextRoot(BaseModel):
    """An AFS .context directory."""

    path: Path
    project_name: str
    metadata: ProjectMetadata = Field(default_factory=ProjectMetadata)
    mounts: dict[MountType, list[MountPoint]] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if all required directories exist."""
        required = [mt.value for mt in MountType]
        directory_map = self.metadata.directories if self.metadata else {}
        if directory_map:
            return all(
                (self.path / directory_map.get(role, role)).exists() for role in required
            )
        return all((self.path / d).exists() for d in required)

    @property
    def total_mounts(self) -> int:
        """Count total number of mounts across all types."""
        return sum(len(mounts) for mounts in self.mounts.values())

    def get_mounts(self, mount_type: MountType) -> list[MountPoint]:
        """Get mounts for a specific type."""
        return self.mounts.get(mount_type, [])
