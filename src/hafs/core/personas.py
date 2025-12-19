"""Persona and skill registry for agent orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from hafs.config.loader import load_config
from hafs.models.agent import AgentRole


@dataclass(frozen=True)
class Skill:
    """Normalized skill definition."""

    name: str
    description: str = ""
    tools: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Persona:
    """Normalized persona definition."""

    name: str
    role: AgentRole
    system_prompt: str = ""
    skills: list[Skill] = field(default_factory=list)
    tool_profile: Optional[str] = None
    execution_mode: Optional[str] = None
    constraints: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    description: str = ""
    default_for_role: bool = False
    enabled: bool = True


class PersonaRegistry:
    """Registry for personas and skills."""

    def __init__(self, personas: list[Persona], skills: dict[str, Skill]) -> None:
        self._personas = [p for p in personas if p.enabled]
        self._skills = skills

    @classmethod
    def load(cls) -> "PersonaRegistry":
        config = load_config()

        skills: dict[str, Skill] = {}
        for skill in config.skills:
            skills[skill.name.lower()] = Skill(
                name=skill.name,
                description=skill.description,
                tools=list(skill.tools),
                constraints=list(skill.constraints),
                goals=list(skill.goals),
            )

        personas: list[Persona] = []
        for persona in config.personas:
            try:
                role = AgentRole(persona.role)
            except Exception:
                role = AgentRole.GENERAL
            persona_skills = [
                skills[name.lower()]
                for name in persona.skills
                if name.lower() in skills
            ]
            personas.append(
                Persona(
                    name=persona.name,
                    role=role,
                    system_prompt=persona.system_prompt,
                    skills=persona_skills,
                    tool_profile=persona.tool_profile,
                    execution_mode=persona.execution_mode,
                    constraints=list(persona.constraints),
                    goals=list(persona.goals),
                    description=persona.description,
                    default_for_role=persona.default_for_role,
                    enabled=persona.enabled,
                )
            )

        return cls(personas=personas, skills=skills)

    def list(self) -> list[Persona]:
        return list(self._personas)

    def get(self, name: str) -> Optional[Persona]:
        needle = name.lower()
        for persona in self._personas:
            if persona.name.lower() == needle:
                return persona
        return None

    def default_for_role(self, role: AgentRole) -> Optional[Persona]:
        for persona in self._personas:
            if persona.role == role and persona.default_for_role:
                return persona
        return None

    def skill(self, name: str) -> Optional[Skill]:
        return self._skills.get(name.lower())
