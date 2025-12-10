"""Gemini log parser (ported from halext-code C++)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterator

from hafs.core.parsers.base import BaseParser
from hafs.models.gemini import GeminiMessage, GeminiProject, GeminiSession


class GeminiLogParser(BaseParser[GeminiSession]):
    """Parser for Gemini CLI logs (~/.gemini/tmp/<project_hash>/).

    Parses both logs.json (user messages) and session-*.json (full conversations).
    """

    def default_path(self) -> Path:
        """Return default Gemini logs path."""
        return Path.home() / ".gemini" / "tmp"

    def parse(self, max_items: int = 50) -> list[GeminiSession]:
        """Load recent sessions across all projects.

        Args:
            max_items: Maximum number of sessions to return.

        Returns:
            List of GeminiSession objects, sorted by last_updated descending.
        """
        sessions: list[GeminiSession] = []

        for project_dir in self._find_projects():
            for session_path in self._find_sessions(project_dir):
                session = self._parse_session(session_path)
                if session:
                    sessions.append(session)

        # Sort by last_updated descending
        sessions.sort(key=lambda s: s.last_updated, reverse=True)
        return sessions[:max_items]

    def parse_user_logs(self, logs_path: Path) -> list[GeminiMessage]:
        """Parse logs.json for user messages only.

        Args:
            logs_path: Path to logs.json file.

        Returns:
            List of GeminiMessage objects (user messages only).
        """
        messages: list[GeminiMessage] = []
        if not logs_path.exists():
            return messages

        try:
            with open(logs_path, encoding="utf-8") as f:
                data = json.load(f)

            for entry in data:
                timestamp_str = entry.get("timestamp", "")
                try:
                    timestamp = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                except ValueError:
                    timestamp = datetime.now()

                msg = GeminiMessage(
                    id=str(entry.get("messageId", 0)),
                    timestamp=timestamp,
                    type=entry.get("type", "user"),
                    content=entry.get("message", ""),
                )
                messages.append(msg)
        except (json.JSONDecodeError, KeyError, OSError):
            pass

        return messages

    def parse_projects(self) -> list[GeminiProject]:
        """Parse all projects with their sessions.

        Returns:
            List of GeminiProject objects.
        """
        projects: list[GeminiProject] = []

        for project_dir in self._find_projects():
            sessions: list[GeminiSession] = []
            for session_path in self._find_sessions(project_dir):
                session = self._parse_session(session_path)
                if session:
                    sessions.append(session)

            if sessions:
                project = GeminiProject(
                    project_hash=project_dir.name,
                    path=project_dir,
                    sessions=sessions,
                )
                projects.append(project)

        return projects

    def _parse_session(self, session_path: Path) -> GeminiSession | None:
        """Parse a session-*.json file.

        Args:
            session_path: Path to session JSON file.

        Returns:
            GeminiSession object or None on parse error.
        """
        try:
            with open(session_path, encoding="utf-8") as f:
                data = json.load(f)

            messages: list[GeminiMessage] = []
            for msg_data in data.get("messages", []):
                tool_names: list[str] = []
                for tool in msg_data.get("toolCalls", []):
                    name = tool.get("name", "")
                    if name:
                        tool_names.append(name)

                tokens = msg_data.get("tokens", {})
                timestamp_str = msg_data.get("timestamp", "")
                try:
                    timestamp = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                except ValueError:
                    timestamp = datetime.now()

                msg = GeminiMessage(
                    id=msg_data.get("id", ""),
                    timestamp=timestamp,
                    type=msg_data.get("type", ""),
                    content=msg_data.get("content", ""),
                    tool_names=tool_names,
                    model=msg_data.get("model", ""),
                    total_tokens=tokens.get("total", 0),
                )
                messages.append(msg)

            start_str = data.get("startTime", "")
            updated_str = data.get("lastUpdated", "")
            try:
                start_time = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            except ValueError:
                start_time = datetime.now()
            try:
                last_updated = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
            except ValueError:
                last_updated = datetime.now()

            return GeminiSession(
                session_id=data.get("sessionId", session_path.stem),
                project_hash=data.get("projectHash", ""),
                start_time=start_time,
                last_updated=last_updated,
                messages=messages,
            )
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            return None

    def _find_projects(self) -> Iterator[Path]:
        """Find all project directories.

        Yields:
            Path objects for each project directory.
        """
        if not self.base_path.exists():
            return

        for entry in self.base_path.iterdir():
            if entry.is_dir():
                # Check if has chats/ or logs.json
                if (entry / "chats").exists() or (entry / "logs.json").exists():
                    yield entry

    def _find_sessions(self, project_dir: Path) -> Iterator[Path]:
        """Find session files in a project directory.

        Args:
            project_dir: Path to project directory.

        Yields:
            Path objects for each session file.
        """
        chats_dir = project_dir / "chats"
        if not chats_dir.exists():
            return

        for entry in sorted(chats_dir.iterdir(), reverse=True):
            if (
                entry.is_file()
                and entry.name.startswith("session-")
                and entry.suffix == ".json"
            ):
                yield entry

    def search(
        self, query: str, items: list[GeminiSession] | None = None
    ) -> list[GeminiSession]:
        """Search sessions by keyword in messages.

        Args:
            query: Search query (case-insensitive).
            items: Optional pre-parsed sessions. If None, calls parse().

        Returns:
            List of sessions containing matching messages.
        """
        if items is None:
            items = self.parse()

        query_lower = query.lower()
        results: list[GeminiSession] = []

        for session in items:
            for msg in session.messages:
                if query_lower in msg.content.lower():
                    results.append(session)
                    break

        return results

    @staticmethod
    def extract_project_hash(path: Path) -> str:
        """Extract project hash from path.

        Args:
            path: Path potentially containing a project hash.

        Returns:
            64-character hex project hash, or empty string if not found.
        """
        for part in path.parts:
            if len(part) == 64 and all(c in "0123456789abcdef" for c in part.lower()):
                return part
        return ""
