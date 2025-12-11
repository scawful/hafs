"""Claude plan data models (ported from halext-code C++)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class TaskStatus(str, Enum):
    """Status of a plan task."""

    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class PlanTask(BaseModel):
    """A task from a Claude plan document."""

    description: str
    status: TaskStatus

    model_config = ConfigDict(frozen=True)

    @property
    def is_done(self) -> bool:
        """Check if task is completed."""
        return self.status == TaskStatus.DONE

    @property
    def is_in_progress(self) -> bool:
        """Check if task is in progress."""
        return self.status == TaskStatus.IN_PROGRESS

    @property
    def is_todo(self) -> bool:
        """Check if task is pending."""
        return self.status == TaskStatus.TODO


class PlanDocument(BaseModel):
    """A Claude plan markdown document."""

    title: str
    path: Path
    tasks: list[PlanTask] = Field(default_factory=list)
    modified_at: datetime | None = None

    @property
    def progress(self) -> tuple[int, int]:
        """Returns (completed, total) task counts."""
        done = sum(1 for t in self.tasks if t.is_done)
        return (done, len(self.tasks))

    @property
    def progress_percent(self) -> float:
        """Returns completion percentage."""
        done, total = self.progress
        return (done / total * 100) if total > 0 else 0.0

    @property
    def has_tasks(self) -> bool:
        """Check if document has any tasks."""
        return len(self.tasks) > 0

    @property
    def in_progress_count(self) -> int:
        """Count tasks currently in progress."""
        return sum(1 for t in self.tasks if t.is_in_progress)

    @property
    def todo_count(self) -> int:
        """Count pending tasks."""
        return sum(1 for t in self.tasks if t.is_todo)

    @property
    def done_count(self) -> int:
        """Count completed tasks."""
        return sum(1 for t in self.tasks if t.is_done)
