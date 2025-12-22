"""Project registry for background agents and knowledge builders."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config.loader import load_config
from core.search import fuzzy_filter_multi
from core.tooling import ToolProfile


@dataclass
class Project:
    """Represents a tracked project in the workspace."""

    name: str
    path: Path
    kind: str = "general"
    tags: list[str] = field(default_factory=list)
    tooling_profile: Optional[str] = None
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    knowledge_roots: list[Path] = field(default_factory=list)
    enabled: bool = True
    description: str = ""

    @property
    def slug(self) -> str:
        safe = re.sub(r"[^a-z0-9_-]+", "_", self.name.lower())
        return safe.strip("_")


class ProjectRegistry:
    """Loads and manages project configuration for background agents."""

    def __init__(
        self,
        projects: list[Project],
        tool_profiles: dict[str, ToolProfile],
        default_tool_profile: str = "read_only",
    ) -> None:
        self._projects = [p for p in projects if p.enabled]
        self._tool_profiles = tool_profiles
        self._default_tool_profile = default_tool_profile

    @classmethod
    def load(cls) -> "ProjectRegistry":
        config = load_config()

        projects: list[Project] = []
        if config.projects:
            for project in config.projects:
                projects.append(
                    Project(
                        name=project.name,
                        path=project.path,
                        kind=project.kind,
                        tags=list(project.tags),
                        tooling_profile=project.tooling_profile,
                        embedding_provider=project.embedding_provider,
                        embedding_model=project.embedding_model,
                        knowledge_roots=list(project.knowledge_roots),
                        enabled=project.enabled,
                        description=project.description,
                    )
                )
        elif config.tracked_projects:
            for path in config.tracked_projects:
                projects.append(
                    Project(
                        name=path.name,
                        path=path,
                        kind="general",
                        tags=[],
                    )
                )

        tool_profiles: dict[str, ToolProfile] = {}
        for profile in config.tool_profiles:
            tool_profiles[profile.name] = ToolProfile(
                name=profile.name,
                allow=set(profile.allow),
                deny=set(profile.deny),
            )

        return cls(
            projects=projects,
            tool_profiles=tool_profiles,
            default_tool_profile=config.default_tool_profile,
        )

    def list(self) -> list[Project]:
        return list(self._projects)

    def get(self, name: str) -> Optional[Project]:
        for project in self._projects:
            if project.name.lower() == name.lower():
                return project
        return None

    def match_path(self, path: Path) -> Optional[Project]:
        """Match a project by filesystem path."""
        try:
            resolved = path.expanduser().resolve()
        except Exception:
            resolved = path
        for project in self._projects:
            try:
                project_path = project.path.expanduser().resolve()
            except Exception:
                project_path = project.path
            if resolved == project_path or resolved.is_relative_to(project_path):
                return project
        return None

    def match(self, query: str) -> list[Project]:
        if not query:
            return self.list()
        results = fuzzy_filter_multi(
            query,
            self._projects,
            keys={
                "name": lambda p: p.name,
                "tags": lambda p: " ".join(p.tags),
                "kind": lambda p: p.kind,
            },
            threshold=55,
            weights={"name": 1.0, "tags": 0.7, "kind": 0.6},
        )
        return [result.item for result in results]

    def resolve_tool_profile(self, project: Project) -> ToolProfile:
        profile_name = project.tooling_profile or self._default_tool_profile
        if profile_name in self._tool_profiles:
            return self._tool_profiles[profile_name]
        if self._default_tool_profile in self._tool_profiles:
            return self._tool_profiles[self._default_tool_profile]
        return ToolProfile(name=profile_name, allow=set(), deny=set())

    def get_tool_profile(self, name: str) -> Optional[ToolProfile]:
        return self._tool_profiles.get(name)
