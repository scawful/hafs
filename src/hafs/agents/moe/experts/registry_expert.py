"""Registry-backed expert for planned/active models."""

from __future__ import annotations

from typing import Optional

from agents.moe.expert import BaseExpert, ExpertConfig
from agents.moe.registry import ModelRecord, RoutingTable
from core.orchestrator_v2 import Provider, TaskTier, UnifiedOrchestrator


def _provider_from_string(value: Optional[str]) -> Optional[Provider]:
    if not value:
        return None
    try:
        return Provider(value)
    except ValueError:
        return None


class RegistryExpert(BaseExpert):
    """Expert constructed from model registry metadata."""

    def __init__(
        self,
        model: ModelRecord,
        orchestrator: Optional[UnifiedOrchestrator] = None,
        routing_table: Optional[RoutingTable] = None,
    ):
        self.model = model
        keywords = routing_table.keywords_for_expert(model.name) if routing_table else []

        config = ExpertConfig(
            name=model.name,
            display_name=model.display_name,
            specialization=model.role or model.group or "general",
            keywords=keywords,
            confidence_threshold=0.0,
            model_name=model.inference_model,
            tier=TaskTier.CODING,
            temperature=0.6,
            max_tokens=2048,
            provider=_provider_from_string(model.default_provider),
        )

        super().__init__(config, orchestrator)

    def get_system_prompt(self) -> str:
        """Return a generic system prompt for registry experts."""
        if self.model.system_prompt:
            return self.model.system_prompt

        role = self.model.role or "specialist"
        group = self.model.group or "HAFS"
        tags = ", ".join(self.model.tags) if self.model.tags else ""
        notes = self.model.notes or ""

        prompt_lines = [
            f"You are {self.model.display_name}, a {role} expert for {group}.",
            "Focus on precise, practical guidance for ROM hacking workflows.",
        ]

        if tags:
            prompt_lines.append(f"Expert tags: {tags}.")
        if notes:
            prompt_lines.append(notes)

        return "\n".join(prompt_lines)

