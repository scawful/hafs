"""Gemini log parser (ported from halext-code C++)."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

from hafs.core.parsers.base import BaseParser
from hafs.models.gemini import GeminiMessage, GeminiProject, GeminiSession

logger = logging.getLogger(__name__)


class GeminiLogParser(BaseParser[GeminiSession]):
    """Parser for Gemini CLI logs (~/.gemini/tmp/<project_hash>/).

    Parses both logs.json (user messages) and session-*.json (full conversations).
    """

    def default_path(self) -> Path:
        """Return default Gemini logs path."""
        return Path.home() / ".gemini" / "tmp"

    def _get_search_keys(self) -> dict[str, Callable[[GeminiSession], str]]:
        """Get searchable field extractors for Gemini sessions."""
        return {
            "session_id": lambda s: s.session_id,
            "content": lambda s: " ".join(m.content for m in s.messages),
        }

    def parse(self, max_items: int = 50) -> list[GeminiSession]:
        """Load recent sessions across all projects.

        Args:
            max_items: Maximum number of sessions to return.

        Returns:
            List of GeminiSession objects, sorted by last_updated descending.
        """
        sessions: list[GeminiSession] = []
        self._last_error = None

        if not self.base_path.exists():
            self._set_error(f"Base path does not exist: {self.base_path}")
            return sessions

        project_count = 0
        session_count = 0

        try:
            for project_dir in self._find_projects():
                project_count += 1
                for session_path in self._find_sessions(project_dir):
                    session_count += 1
                    try:
                        session = self._parse_session(session_path)
                        if session:
                            sessions.append(session)
                    except Exception as e:
                        logger.debug(f"Failed to parse session {session_path}: {e}")

            if project_count == 0:
                self._set_error(f"No projects found in {self.base_path}")
            elif session_count == 0:
                self._set_error(f"Found {project_count} projects but no sessions")

        except PermissionError as e:
            self._set_error(f"Permission denied: {e}")
        except Exception as e:
            self._set_error(f"Error scanning projects: {e}")

        # Deduplicate by (project_hash, session_id), keeping the most recent
        deduped: dict[tuple[str, str], GeminiSession] = {}
        for session in sessions:
            key = (session.project_hash or "", session.session_id)
            existing = deduped.get(key)
            if existing is None or session.last_updated > existing.last_updated:
                deduped[key] = session

        sessions = list(deduped.values())

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
            dedup: dict[str, GeminiSession] = {}
            for session_path in self._find_sessions(project_dir):
                session = self._parse_session(session_path)
                if session:
                    existing = dedup.get(session.session_id)
                    if existing is None or session.last_updated > existing.last_updated:
                        dedup[session.session_id] = session

            sessions = list(dedup.values())

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
                thoughts: list[str] = []
                for thought in msg_data.get("thoughts", []) or []:
                    if isinstance(thought, dict):
                        subject = str(thought.get("subject", "")).strip()
                        description = str(thought.get("description", "")).strip()
                        if subject and description:
                            thoughts.append(f"{subject}: {description}")
                        elif description:
                            thoughts.append(description)
                    elif isinstance(thought, str):
                        if thought.strip():
                            thoughts.append(thought.strip())
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
                    thoughts=thoughts,
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

            # Extract project hash from path if not in data
            project_hash = data.get("projectHash", "")
            if not project_hash:
                project_hash = self.extract_project_hash(session_path)

            return GeminiSession(
                session_id=data.get("sessionId", session_path.stem),
                project_hash=project_hash,
                start_time=start_time,
                last_updated=last_updated,
                messages=messages,
                source_path=session_path,  # Store the source path for later operations
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
        """Search sessions by keyword in messages (fuzzy matching).

        Args:
            query: Search query (case-insensitive).
            items: Optional pre-parsed sessions. If None, calls parse().

        Returns:
            List of sessions containing matching messages, sorted by relevance.
        """
        # Use fuzzy search and extract items from results
        results = self.fuzzy_search(query, items, threshold=50)
        return [r.item for r in results]

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

    def get_item_path(self, item: GeminiSession) -> Path | None:
        """Get the file path for a session.

        Args:
            item: The GeminiSession to get the path for.

        Returns:
            Path to the session JSON file.
        """
        # First try the stored source path
        if item.source_path and item.source_path.exists():
            return item.source_path

        # Fall back to constructing from project_hash
        if not item.project_hash:
            return None

        session_path = (
            self.base_path
            / item.project_hash
            / "chats"
            / f"session-{item.session_id}.json"
        )
        if session_path.exists():
            return session_path
        return None

    def delete_item(self, item: GeminiSession) -> bool:
        """Delete a Gemini session.

        Args:
            item: The GeminiSession to delete.

        Returns:
            True if deletion was successful.
        """
        session_path = self.get_item_path(item)
        if not session_path:
            self._set_error(f"Could not find session file for {item.short_id}")
            return False

        if not session_path.exists():
            self._set_error(f"Session file does not exist: {session_path}")
            return False

        try:
            session_path.unlink()
            logger.info(f"Deleted session: {session_path}")
            return True
        except OSError as e:
            self._set_error(f"Failed to delete session file: {e}")
            return False

    def save_to_context(
        self, item: GeminiSession, context_dir: Path
    ) -> Path | None:
        """Save a Gemini session to a context directory.

        Creates a markdown file with the session conversation.

        Args:
            item: The GeminiSession to save.
            context_dir: Directory to save the context file to.

        Returns:
            Path to the saved file, or None if failed.
        """
        try:
            context_dir.mkdir(parents=True, exist_ok=True)

            # Create filename from session ID and timestamp
            timestamp = item.start_time.strftime("%Y%m%d_%H%M%S")
            filename = f"gemini_session_{item.short_id}_{timestamp}.md"
            output_path = context_dir / filename

            # Build markdown content
            lines = [
                f"# Gemini Session {item.short_id}",
                "",
                f"**Started:** {item.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Last Updated:** {item.last_updated.strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Messages:** {len(item.messages)}",
                f"**Tokens:** {item.total_tokens}",
                f"**Models:** {', '.join(item.models_used) or 'N/A'}",
                "",
                "---",
                "",
            ]

            for msg in item.messages:
                role = msg.type.upper() if msg.type else "UNKNOWN"
                timestamp_str = msg.timestamp.strftime("%H:%M:%S")

                lines.append(f"## [{role}] {timestamp_str}")
                if msg.model:
                    lines.append(f"*Model: {msg.model}*")
                lines.append("")
                lines.append(msg.content)
                lines.append("")

                if msg.tool_names:
                    lines.append(f"**Tools:** {', '.join(msg.tool_names)}")
                    lines.append("")

            content = "\n".join(lines)
            output_path.write_text(content, encoding="utf-8")
            logger.info(f"Saved session to context: {output_path}")
            return output_path

        except OSError as e:
            self._set_error(f"Failed to save session to context: {e}")
            return None
