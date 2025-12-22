"""Base classes and models for autonomy agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agents.core.base import BaseAgent
from core.history import AgentMemoryManager


@dataclass
class LoopReport:
    """Structured report output from an autonomy loop."""

    title: str
    body: str
    tags: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class MemoryAwareAgent(BaseAgent):
    """Base agent that writes summaries into AgentMemory."""

    def __init__(self, name: str, role_description: str):
        super().__init__(name, role_description)
        self._memory_manager: Optional[AgentMemoryManager] = None

    def _get_memory(self):
        if self._memory_manager is None:
            self._memory_manager = AgentMemoryManager(self.context_root)
        return self._memory_manager.get_agent_memory(self.name)

    async def remember(
        self,
        content: str,
        memory_type: str,
        context: Optional[dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> None:
        try:
            memory = self._get_memory()
            await memory.remember(
                content=content,
                memory_type=memory_type,
                context=context or {},
                importance=importance,
            )
        except Exception:
            return
