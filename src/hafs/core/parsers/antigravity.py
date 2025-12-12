"""Antigravity parser (ported from halext-code C++)."""

from __future__ import annotations

import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable

from hafs.core.parsers.base import BaseParser
from hafs.models.antigravity import AntigravityBrain, AntigravityTask

logger = logging.getLogger(__name__)


class AntigravityParser(BaseParser[AntigravityBrain]):
    """Parser for Antigravity brain directory (~/.gemini/antigravity/brain/).

    Parses brain directories containing task.md, implementation_plan.md,
    and walkthrough.md files.
    """

    TASK_PATTERN = re.compile(
        r"^\s*(?:[-*+]|[0-9]+\.)\s*\[([ xX/✓✔✅~>-])\]\s*(.+?)\s*$"
    )

    def default_path(self) -> Path:
        """Return default Antigravity brain path."""
        return Path.home() / ".gemini" / "antigravity" / "brain"

    def _get_search_keys(self) -> dict[str, Callable[[AntigravityBrain], str]]:
        """Get searchable field extractors for Antigravity brains."""
        return {
            "id": lambda b: b.id,
            "title": lambda b: b.title or "",
            "tasks": lambda b: " ".join(t.description for t in b.tasks),
            "notes": lambda b: " ".join(b.notes),
        }

    def parse(self, max_items: int = 50) -> list[AntigravityBrain]:
        """Parse latest brain directories.

        Args:
            max_items: Maximum number of brains to return.

        Returns:
            List of AntigravityBrain objects, sorted by modification time descending.
        """
        brains: list[AntigravityBrain] = []
        self._last_error = None

        if not self.base_path.exists():
            self._set_error(f"Base path does not exist: {self.base_path}")
            return brains

        try:
            # Get directories sorted by mtime
            brain_dirs = sorted(
                [d for d in self.base_path.iterdir() if d.is_dir()],
                key=lambda d: d.stat().st_mtime,
                reverse=True,
            )

            if not brain_dirs:
                self._set_error(f"No brain directories found in {self.base_path}")
                return brains

            for brain_dir in brain_dirs[:max_items]:
                try:
                    brain = self._parse_brain(brain_dir)
                    if brain:
                        brains.append(brain)
                except Exception as e:
                    logger.debug(f"Failed to parse brain {brain_dir}: {e}")

        except PermissionError as e:
            self._set_error(f"Permission denied: {e}")
        except Exception as e:
            self._set_error(f"Error scanning brains: {e}")

        return brains

    def _parse_brain(self, path: Path) -> AntigravityBrain:
        """Parse a single brain directory.

        Args:
            path: Path to brain directory.

        Returns:
            AntigravityBrain object.
        """
        tasks: list[AntigravityTask] = []
        notes: list[str] = []
        plan_summary = ""
        walkthrough_summary = ""
        try:
            updated_at = datetime.fromtimestamp(path.stat().st_mtime)
        except OSError:
            updated_at = None

        # Parse tasks from task.md (prefer resolved if present)
        task_file = self._pick_markdown(path / "task.md")
        if task_file:
            tasks = self._parse_tasks(task_file)

        # Read notes from implementation_plan.md and walkthrough.md
        for note_file in ["implementation_plan.md", "walkthrough.md"]:
            note_path = self._pick_markdown(path / note_file)
            if note_path:
                snippet = self._read_notes(note_path, max_lines=5)
                notes.extend(snippet)
                if note_file == "implementation_plan.md" and note_path.name.startswith("implementation_plan"):
                    plan_summary = "\n".join(snippet[:3])
                elif note_file == "walkthrough.md" and note_path.name.startswith("walkthrough"):
                    walkthrough_summary = "\n".join(snippet[:3])

        return AntigravityBrain(
            id=path.name,
            path=path,
            tasks=tasks,
            notes=notes,
            updated_at=updated_at,
            plan_summary=plan_summary,
            walkthrough_summary=walkthrough_summary,
        )

    def _parse_tasks(self, path: Path) -> list[AntigravityTask]:
        """Extract tasks from a markdown file.

        Args:
            path: Path to markdown file containing tasks.

        Returns:
            List of AntigravityTask objects.
        """
        tasks: list[AntigravityTask] = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    match = self.TASK_PATTERN.match(line)
                    if match:
                        status_char = match.group(1)
                        description = match.group(2).strip()

                        if status_char in ("x", "X", "✓", "✔", "✅"):
                            status = "done"
                        elif status_char in ("/", "~", ">", "-"):
                            status = "in_progress"
                        else:
                            status = "todo"

                        tasks.append(
                            AntigravityTask(description=description, status=status)
                        )
        except (OSError, UnicodeDecodeError):
            pass

        return tasks

    def _read_notes(self, path: Path, max_lines: int = 5) -> list[str]:
        """Read first N non-empty lines from a file.

        Args:
            path: Path to file.
            max_lines: Maximum number of lines to read.

        Returns:
            List of non-empty lines.
        """
        notes: list[str] = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        notes.append(line)
                        if len(notes) >= max_lines:
                            break
        except (OSError, UnicodeDecodeError):
            pass

        return notes

    def _pick_markdown(self, base: Path) -> Path | None:
        """Prefer resolved markdown variants if present."""
        if not base.parent.exists():
            return None

        candidates = []
        resolved_glob = list(base.parent.glob(base.name + ".resolved*"))
        resolved_glob.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        candidates.extend(resolved_glob)
        if base.exists():
            candidates.append(base)

        return candidates[0] if candidates else None

    def get_item_path(self, item: AntigravityBrain) -> Path | None:
        """Return brain directory path."""
        return item.path if item.path.exists() else None

    def delete_item(self, item: AntigravityBrain) -> bool:
        """Delete a brain directory."""
        brain_path = self.get_item_path(item)
        if not brain_path:
            self._set_error(f"Could not find brain directory for {item.short_id}")
            return False
        try:
            shutil.rmtree(brain_path)
            return True
        except OSError as e:
            self._set_error(f"Failed to delete brain: {e}")
            return False

    def save_to_context(self, item: AntigravityBrain, context_dir: Path) -> Path | None:
        """Save an Antigravity brain summary to a context directory."""
        try:
            context_dir.mkdir(parents=True, exist_ok=True)
            timestamp = (item.updated_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
            filename = f"antigravity_brain_{item.short_id}_{timestamp}.md"
            output_path = context_dir / filename

            done, total = item.progress
            lines = [
                f"# Antigravity Brain {item.short_id}",
                "",
                f"**ID:** {item.id}",
                f"**Updated:** {(item.updated_at or datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Progress:** {done}/{total} tasks",
                "",
                "## Tasks",
                "",
            ]

            for task in item.tasks:
                box = "[x]" if task.is_done else "[/]" if task.is_in_progress else "[ ]"
                lines.append(f"- {box} {task.description}")

            if item.notes:
                lines += ["", "## Notes", ""]
                lines.extend(item.notes)

            if item.plan_summary:
                lines += ["", "## Implementation Plan (snippet)", "", item.plan_summary]
            if item.walkthrough_summary:
                lines += ["", "## Walkthrough (snippet)", "", item.walkthrough_summary]

            output_path.write_text("\n".join(lines), encoding="utf-8")
            return output_path
        except OSError as e:
            self._set_error(f"Failed to save brain to context: {e}")
            return None

    def search(
        self, query: str, items: list[AntigravityBrain] | None = None
    ) -> list[AntigravityBrain]:
        """Search brains by task description or notes (fuzzy matching).

        Args:
            query: Search query (case-insensitive).
            items: Optional pre-parsed brains. If None, calls parse().

        Returns:
            List of brains matching the query, sorted by relevance.
        """
        # Use fuzzy search and extract items from results
        results = self.fuzzy_search(query, items, threshold=50)
        return [r.item for r in results]

    def get_active_brains(self) -> list[AntigravityBrain]:
        """Get brains with in-progress tasks.

        Returns:
            List of AntigravityBrain objects with tasks in progress.
        """
        brains = self.parse()
        return [b for b in brains if any(t.is_in_progress for t in b.tasks)]
