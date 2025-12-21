"""Pipeline scaffolding for multi-step orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional


class StepStatus(str, Enum):
    """Status values for pipeline steps."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineContext:
    """Shared context for pipeline execution."""

    topic: str
    research: str = ""
    plan: Any = None
    data: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    critique: str = ""
    summary: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineStep:
    """Single step in an orchestration pipeline."""

    name: str
    kind: str
    run: Callable[[PipelineContext], Awaitable[Any]]
    description: str = ""
    required: bool = True


@dataclass
class PipelineStepResult:
    """Result for a single pipeline step."""

    name: str
    kind: str
    status: StepStatus
    output: Any = None
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Result of a full pipeline run."""

    context: PipelineContext
    steps: list[PipelineStepResult] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        """True when no pipeline steps failed."""
        return not any(step.status == StepStatus.FAILED for step in self.steps)


class OrchestrationPipeline:
    """Runs a sequence of orchestration steps with shared context."""

    def __init__(self, steps: list[PipelineStep]) -> None:
        self._steps = steps

    async def run(self, context: PipelineContext) -> PipelineResult:
        results: list[PipelineStepResult] = []

        for step in self._steps:
            try:
                output = await step.run(context)
                results.append(
                    PipelineStepResult(
                        name=step.name,
                        kind=step.kind,
                        status=StepStatus.SUCCESS,
                        output=output,
                    )
                )
            except Exception as exc:
                results.append(
                    PipelineStepResult(
                        name=step.name,
                        kind=step.kind,
                        status=StepStatus.FAILED,
                        error=str(exc),
                    )
                )
                if step.required:
                    break

        return PipelineResult(context=context, steps=results)
