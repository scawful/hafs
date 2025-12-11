from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional


class VCSStatusType(Enum):
    MODIFIED = "modified"
    ADDED = "added"
    DELETED = "deleted"
    RENAMED = "renamed"
    UNTRACKED = "untracked"
    IGNORED = "ignored"
    UNKNOWN = "unknown"

@dataclass
class VCSFileStatus:
    path: Path
    status: VCSStatusType
    staged: bool = False

class SourceControlProvider(ABC):
    """Abstract base class for source control providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the provider (e.g., 'git', 'svn')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the VCS tool is installed and available."""
        pass

    @abstractmethod
    def is_repository(self, path: Path) -> bool:
        """Check if the given path is inside a valid repository."""
        pass

    @abstractmethod
    def get_root(self, path: Path) -> Optional[Path]:
        """Get the root directory of the repository."""
        pass

    @abstractmethod
    def get_branch(self, path: Path) -> Optional[str]:
        """Get the current branch name."""
        pass

    @abstractmethod
    def get_status(self, path: Path) -> List[VCSFileStatus]:
        """Get the status of files in the repository."""
        pass

    @abstractmethod
    def stage_file(self, file_path: Path) -> bool:
        """Stage a file."""
        pass

    @abstractmethod
    def unstage_file(self, file_path: Path) -> bool:
        """Unstage a file."""
        pass

    @abstractmethod
    def commit(self, path: Path, message: str) -> bool:
        """Commit staged changes."""
        pass

    @abstractmethod
    def get_diff(self, file_path: Path, staged: bool = False) -> str:
        """Get the diff of a file."""
        pass
