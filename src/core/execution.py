"""Execution policy resolution for tool profiles and modes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from config.loader import load_config
from core.projects import Project, ProjectRegistry
from core.personas import PersonaRegistry
from core.tooling import ToolProfile


@dataclass(frozen=True)
class ExecutionMode:
    """Execution mode definition."""

    name: str
    tool_profile: str
    description: str = ""


class ExecutionPolicy:
    """Resolve tool profiles based on project + execution mode."""

    def __init__(
        self,
        registry: Optional[ProjectRegistry] = None,
        execution_mode: Optional[str] = None,
        persona: Optional[str] = None,
        persona_registry: Optional[PersonaRegistry] = None,
    ) -> None:
        self._config = load_config()
        self._registry = registry or ProjectRegistry.load()
        self._execution_modes = {
            mode.name: ExecutionMode(
                name=mode.name,
                tool_profile=mode.tool_profile,
                description=mode.description,
            )
            for mode in self._config.execution_modes
        }
        self._execution_mode = (
            execution_mode
            or os.environ.get("HAFS_EXEC_MODE")
            or self._config.default_execution_mode
        )
        self._persona_name = persona
        self._persona_registry = persona_registry or (
            PersonaRegistry.load() if persona else None
        )

    @property
    def execution_mode(self) -> str:
        return self._execution_mode

    def _resolve_mode_profile(self, mode_name: Optional[str]) -> Optional[ToolProfile]:
        if not mode_name:
            return None
        mode = self._execution_modes.get(mode_name)
        if not mode:
            return None
        return self._registry.get_tool_profile(mode.tool_profile)

    def resolve_tool_profile(self, project: Optional[Project]) -> ToolProfile:
        if project and project.tooling_profile:
            profile = self._registry.get_tool_profile(project.tooling_profile)
            if profile:
                return profile

        if self._persona_name and self._persona_registry:
            persona = self._persona_registry.get(self._persona_name)
            if persona:
                profile = self._registry.get_tool_profile(persona.tool_profile or "")
                if profile:
                    return profile
                profile = self._resolve_mode_profile(persona.execution_mode)
                if profile:
                    return profile

        profile = self._resolve_mode_profile(self._execution_mode)
        if profile:
            return profile

        profile = self._registry.get_tool_profile(self._config.default_tool_profile)
        if profile:
            return profile

        return ToolProfile(name="empty", allow=set(), deny=set())
