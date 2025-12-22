"""Multi-agent orchestration data models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class AgentRole(str, Enum):
    """Role of an agent in the multi-agent system."""

    GENERAL = "general"
    PLANNER = "planner"
    CODER = "coder"
    CRITIC = "critic"
    RESEARCHER = "researcher"


class AgentMessage(BaseModel):
    """A message exchanged between agents."""

    id: UUID = Field(default_factory=uuid4)
    content: str
    sender: str
    recipient: Optional[str] = None
    mentions: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    is_delegation: bool = False
    priority: int = Field(default=0, ge=0, le=10)

    model_config = ConfigDict(frozen=True)

    @property
    def has_mentions(self) -> bool:
        """Check if message has any mentions."""
        return len(self.mentions) > 0

    @property
    def is_broadcast(self) -> bool:
        """Check if message is a broadcast (no specific recipient)."""
        return self.recipient is None

    # Backwards compatibility properties
    @property
    def from_agent(self) -> str:
        """Alias for sender (backwards compatibility)."""
        return self.sender

    @property
    def to_agent(self) -> Optional[str]:
        """Alias for recipient (backwards compatibility)."""
        return self.recipient


class SharedContext(BaseModel):
    """Shared context state across all agents."""

    active_task: Optional[str] = None
    plan: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    context_items: list[Path] = Field(default_factory=list)

    def add_finding(self, finding: str) -> None:
        """Add a finding to shared context, maintaining max 50 entries."""
        self.findings.append(finding)
        if len(self.findings) > 50:
            self.findings.pop(0)

    def add_decision(self, decision: str) -> None:
        """Add a decision to shared context, maintaining max 20 entries."""
        self.decisions.append(decision)
        if len(self.decisions) > 20:
            self.decisions.pop(0)

    def set_context_items(self, paths: list[Path]) -> list[Path]:
        """Replace context items with a de-duplicated, limited list.

        Args:
            paths: Paths selected for context.

        Returns:
            The normalized list that was stored.
        """
        unique_paths: list[Path] = []
        for path in paths:
            normalized = Path(path)
            if normalized not in unique_paths:
                unique_paths.append(normalized)

        # Keep the list manageable for prompts/UX
        self.context_items = unique_paths[:20]
        return self.context_items

    @property
    def has_active_task(self) -> bool:
        """Check if there is an active task."""
        return self.active_task is not None

    @property
    def finding_count(self) -> int:
        """Count of findings in context."""
        return len(self.findings)

    @property
    def decision_count(self) -> int:
        """Count of decisions in context."""
        return len(self.decisions)

    @property
    def plan_step_count(self) -> int:
        """Count of steps in the plan."""
        return len(self.plan)

    @property
    def context_item_count(self) -> int:
        """Count of paths pinned to context."""
        return len(self.context_items)

    def to_prompt_text(self) -> str:
        """Convert shared context to text for prompt injection.

        Returns:
            Formatted context text suitable for inclusion in agent prompts.
        """
        lines = ["=== Shared Context ==="]

        if self.active_task:
            lines.append(f"\nActive Task: {self.active_task}")

        if self.context_items:
            lines.append("\nContext Items:")
            for path in self.context_items[:10]:
                lines.append(f"  - {path}")
            if len(self.context_items) > 10:
                lines.append(f"  [and {len(self.context_items) - 10} more...]")

        if self.plan:
            lines.append("\nPlan:")
            for i, step in enumerate(self.plan, 1):
                lines.append(f"  {i}. {step}")

        if self.findings:
            lines.append("\nKey Findings:")
            for finding in self.findings[-10:]:
                lines.append(f"  - {finding}")

        if self.decisions:
            lines.append("\nDecisions Made:")
            for decision in self.decisions[-5:]:
                lines.append(f"  - {decision}")

        lines.append("\n=== End Shared Context ===\n")
        return "\n".join(lines)


class Agent(BaseModel):
    """An agent in the multi-agent orchestration system."""

    name: str
    role: AgentRole
    backend_name: str
    system_prompt: str
    is_active: bool = True

    model_config = ConfigDict(frozen=True)

    @property
    def is_specialist(self) -> bool:
        """Check if agent has a specialized role."""
        return self.role != AgentRole.GENERAL

    @property
    def can_delegate(self) -> bool:
        """Check if agent can delegate to other agents."""
        return self.role in {AgentRole.GENERAL, AgentRole.PLANNER}
