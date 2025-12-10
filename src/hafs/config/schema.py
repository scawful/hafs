"""Pydantic configuration models for HAFS."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class PolicyType(str, Enum):
    """AFS directory permission policy."""

    READ_ONLY = "read_only"
    WRITABLE = "writable"
    EXECUTABLE = "executable"


class AFSDirectoryConfig(BaseModel):
    """Configuration for a single AFS directory type."""

    name: str
    policy: PolicyType
    description: str = ""


class ParserConfig(BaseModel):
    """Configuration for a log parser."""

    enabled: bool = True
    base_path: Optional[Path] = None
    max_items: int = 50


class ThemeConfig(BaseModel):
    """UI theme configuration."""

    primary: str = "#4C3B52"
    secondary: str = "#9B59B6"
    accent: str = "#E74C3C"
    gradient_start: str = "#4C3B52"
    gradient_end: str = "#000000"


class GeneralConfig(BaseModel):
    """General application settings."""

    refresh_interval: int = 5
    show_hidden_files: bool = False
    default_editor: str = "nvim"


class ParsersConfig(BaseModel):
    """All parser configurations."""

    gemini: ParserConfig = Field(default_factory=ParserConfig)
    claude: ParserConfig = Field(default_factory=ParserConfig)
    antigravity: ParserConfig = Field(default_factory=ParserConfig)


class HafsConfig(BaseModel):
    """Root configuration model."""

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    theme: ThemeConfig = Field(default_factory=ThemeConfig)
    parsers: ParsersConfig = Field(default_factory=ParsersConfig)
    tracked_projects: list[Path] = Field(default_factory=list)
    afs_directories: list[AFSDirectoryConfig] = Field(
        default_factory=lambda: [
            AFSDirectoryConfig(
                name="memory",
                policy=PolicyType.READ_ONLY,
                description="Long-term docs and specs",
            ),
            AFSDirectoryConfig(
                name="knowledge",
                policy=PolicyType.READ_ONLY,
                description="Reference materials",
            ),
            AFSDirectoryConfig(
                name="tools",
                policy=PolicyType.EXECUTABLE,
                description="Executable scripts",
            ),
            AFSDirectoryConfig(
                name="scratchpad",
                policy=PolicyType.WRITABLE,
                description="AI reasoning space",
            ),
            AFSDirectoryConfig(
                name="history",
                policy=PolicyType.READ_ONLY,
                description="Archived scratchpads",
            ),
        ]
    )

    def get_directory_config(self, name: str) -> Optional[AFSDirectoryConfig]:
        """Get configuration for a specific AFS directory."""
        for dir_config in self.afs_directories:
            if dir_config.name == name:
                return dir_config
        return None
