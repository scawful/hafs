"""Antigravity brain data models (ported from halext-code C++)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class AntigravityTask(BaseModel):
    """A task from an Antigravity brain."""

    description: str
    status: str  # "todo", "in_progress", "done"

    class Config:
        frozen = True

    @property
    def is_done(self) -> bool:
        """Check if task is completed."""
        return self.status == "done"

    @property
    def is_in_progress(self) -> bool:
        """Check if task is in progress."""
        return self.status == "in_progress"


class AntigravityBrain(BaseModel):
    """An Antigravity brain directory."""

    id: str  # UUID
    path: Path
    tasks: list[AntigravityTask] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def title(self) -> str:
        """Extract title from first note or use ID."""
        if self.notes:
            first_line = self.notes[0].strip()
            if first_line.startswith("#"):
                return first_line.lstrip("#").strip()
            return first_line[:50] + "..." if len(first_line) > 50 else first_line
        return self.short_id

    @property
    def short_id(self) -> str:
        """Short version of brain ID for display."""
        return self.id[:8] if len(self.id) > 8 else self.id

    @property
    def task_count(self) -> int:
        """Count total tasks."""
        return len(self.tasks)

    @property
    def completed_tasks(self) -> int:
        """Count completed tasks."""
        return sum(1 for t in self.tasks if t.is_done)

    @property
    def progress(self) -> tuple[int, int]:
        """Returns (completed, total) task counts."""
        return (self.completed_tasks, self.task_count)
