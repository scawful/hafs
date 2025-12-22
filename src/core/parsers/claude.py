"""Claude plan parser (ported from halext-code C++)."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Callable

from core.parsers.base import BaseParser
from models.claude import PlanDocument, PlanTask, TaskStatus


class ClaudePlanParser(BaseParser[PlanDocument]):
    """Parser for Claude plan files (~/.claude/plans/*.md).

    Parses markdown files containing task checkboxes in the format:
    - [ ] Task (todo)
    - [x] Task (done)
    - [/] Task (in progress)
    """

    # Support "-", "*", "+", or numbered lists, and a wider set of status glyphs.
    TASK_PATTERN = re.compile(
        r"^\s*(?:[-*+]|[0-9]+\.)\s*\[([ xX/✓✔✅~>-])\]\s*(.+?)\s*$"
    )

    def default_path(self) -> Path:
        """Return default Claude plans path."""
        return Path.home() / ".claude" / "plans"

    def _get_search_keys(self) -> dict[str, Callable[[PlanDocument], str]]:
        """Get searchable field extractors for Claude plans."""
        return {
            "title": lambda p: p.title,
            "tasks": lambda p: " ".join(t.description for t in p.tasks),
        }

    def parse(self, max_items: int = 50) -> list[PlanDocument]:
        """Parse plan markdown files, newest first.

        Args:
            max_items: Maximum number of plans to return.

        Returns:
            List of PlanDocument objects, sorted by modification time descending.
        """
        plans: list[PlanDocument] = []

        if not self.base_path.exists():
            return plans

        # Get all markdown files sorted by mtime
        md_files = sorted(
            [
                f
                for f in self.base_path.iterdir()
                if f.is_file() and f.suffix in (".md", ".markdown")
            ],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

        for plan_path in md_files[:max_items]:
            plan = self._parse_plan(plan_path)
            if plan:
                plans.append(plan)

        return plans

    def _parse_plan(self, path: Path) -> PlanDocument | None:
        """Parse a single plan markdown file.

        Args:
            path: Path to markdown file.

        Returns:
            PlanDocument object or None on parse error.
        """
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            try:
                modified_at = datetime.fromtimestamp(path.stat().st_mtime)
            except OSError:
                modified_at = None
        except (OSError, UnicodeDecodeError):
            return None

        title = path.stem  # Default to filename
        tasks: list[PlanTask] = []

        for line in content.split("\n"):
            # Extract title from first heading
            if line.startswith("#") and title == path.stem:
                title = line.lstrip("#").strip()
                continue

            # Parse task checkboxes
            match = self.TASK_PATTERN.match(line)
            if match:
                status_char = match.group(1)
                description = match.group(2).strip()

                if status_char in ("x", "X", "✓", "✔", "✅"):
                    status = TaskStatus.DONE
                elif status_char in ("/", "~", ">", "-"):
                    status = TaskStatus.IN_PROGRESS
                else:
                    status = TaskStatus.TODO

                tasks.append(PlanTask(description=description, status=status))

        return PlanDocument(title=title, path=path, tasks=tasks, modified_at=modified_at)

    def search(
        self, query: str, items: list[PlanDocument] | None = None
    ) -> list[PlanDocument]:
        """Search plans by title or task description (fuzzy matching).

        Args:
            query: Search query (case-insensitive).
            items: Optional pre-parsed plans. If None, calls parse().

        Returns:
            List of plans matching the query, sorted by relevance.
        """
        # Use fuzzy search and extract items from results
        results = self.fuzzy_search(query, items, threshold=50)
        return [r.item for r in results]

    def get_active_plans(self) -> list[PlanDocument]:
        """Get plans with in-progress tasks.

        Returns:
            List of PlanDocument objects that have tasks in progress.
        """
        plans = self.parse()
        return [p for p in plans if p.in_progress_count > 0]

    def get_incomplete_plans(self) -> list[PlanDocument]:
        """Get plans with incomplete tasks.

        Returns:
            List of PlanDocument objects that have pending or in-progress tasks.
        """
        plans = self.parse()
        return [p for p in plans if p.todo_count > 0 or p.in_progress_count > 0]
