"""Session Manager for the History Pipeline.

Implements session lifecycle management per PROTOCOL_SPEC.md Section 3.2.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from core.history.models import (
    SessionInfo,
    SessionStats,
    SessionStatus,
    SessionSummary,
    SessionSummaryContent,
)

if TYPE_CHECKING:
    from core.history.logger import HistoryLogger

logger = logging.getLogger(__name__)


def _generate_ulid() -> str:
    """Generate a ULID-like identifier."""
    import random
    import time

    timestamp_ms = int(time.time() * 1000)
    chars = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    timestamp_part = ""
    for _ in range(10):
        timestamp_part = chars[timestamp_ms % 32] + timestamp_part
        timestamp_ms //= 32
    random_part = "".join(random.choices(chars, k=16))
    return timestamp_part + random_part


def _get_iso_timestamp() -> str:
    """Get current time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


class SessionManager:
    """Manages session lifecycle and persistence.

    Sessions group related history entries into logical units of work.
    """

    def __init__(
        self,
        sessions_dir: Path | str,
        project_id: Optional[str] = None,
    ) -> None:
        """Initialize the session manager.

        Args:
            sessions_dir: Directory to store session metadata.
            project_id: Optional project identifier.
        """
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.project_id = project_id

        self._current_session: Optional[SessionInfo] = None
        self._history_logger: Optional[HistoryLogger] = None

    @property
    def current_session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self._current_session.id if self._current_session else None

    @property
    def current_session(self) -> Optional[SessionInfo]:
        """Get the current session info."""
        return self._current_session

    def set_history_logger(self, logger: HistoryLogger) -> None:
        """Set the history logger for bidirectional reference."""
        self._history_logger = logger

    def _get_session_file(self, session_id: str) -> Path:
        """Get the file path for a session."""
        return self.sessions_dir / f"{session_id}.json"

    def _save_session(self, session: SessionInfo) -> None:
        """Save session to disk."""
        session_file = self._get_session_file(session.id)
        with open(session_file, "w", encoding="utf-8") as f:
            f.write(session.model_dump_json(indent=2))

    def _load_session(self, session_id: str) -> Optional[SessionInfo]:
        """Load session from disk."""
        session_file = self._get_session_file(session_id)
        if not session_file.exists():
            return None
        try:
            with open(session_file, encoding="utf-8") as f:
                return SessionInfo.model_validate_json(f.read())
        except Exception:
            return None

    def create(
        self,
        parent_session_id: Optional[str] = None,
        extensions: Optional[dict[str, Any]] = None,
    ) -> SessionInfo:
        """Create a new session.

        Args:
            parent_session_id: Optional parent session for delegation.
            extensions: Optional extension data.

        Returns:
            The new session info.
        """
        now = _get_iso_timestamp()
        session = SessionInfo(
            id=_generate_ulid(),
            project_id=self.project_id,
            created_at=now,
            updated_at=now,
            status=SessionStatus.ACTIVE,
            parent_session_id=parent_session_id,
            extensions=extensions or {},
        )

        self._save_session(session)
        self._current_session = session

        # Log system event
        if self._history_logger:
            self._history_logger.log_system_event(
                "session_started",
                {"session_id": session.id, "parent_session_id": parent_session_id},
                session_id=session.id,
            )

        return session

    def suspend(self, session_id: Optional[str] = None) -> Optional[SessionInfo]:
        """Suspend a session.

        Args:
            session_id: Session to suspend (uses current if not provided).

        Returns:
            The updated session info, or None if not found.
        """
        if session_id is None:
            session = self._current_session
        else:
            session = self._load_session(session_id)

        if session is None:
            return None

        session.status = SessionStatus.SUSPENDED
        session.updated_at = _get_iso_timestamp()
        self._save_session(session)

        if self._current_session and self._current_session.id == session.id:
            self._current_session = None

        if self._history_logger:
            self._history_logger.log_system_event(
                "session_suspended",
                {"session_id": session.id},
                session_id=session.id,
            )

        return session

    def resume(self, session_id: str) -> Optional[SessionInfo]:
        """Resume a suspended session.

        Args:
            session_id: Session to resume.

        Returns:
            The updated session info, or None if not found.
        """
        session = self._load_session(session_id)
        if session is None:
            return None

        if session.status != SessionStatus.SUSPENDED:
            return session  # Already active or terminal

        session.status = SessionStatus.ACTIVE
        session.updated_at = _get_iso_timestamp()
        self._save_session(session)

        self._current_session = session

        if self._history_logger:
            self._history_logger.log_system_event(
                "session_resumed",
                {"session_id": session.id},
                session_id=session.id,
            )

        return session

    def complete(
        self,
        session_id: Optional[str] = None,
        summary: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Optional[SessionInfo]:
        """Complete a session normally.

        Args:
            session_id: Session to complete (uses current if not provided).
            summary: Optional summary text.
            title: Optional title.

        Returns:
            The updated session info, or None if not found.
        """
        if session_id is None:
            session = self._current_session
        else:
            session = self._load_session(session_id)

        if session is None:
            return None

        session.status = SessionStatus.COMPLETED
        session.updated_at = _get_iso_timestamp()

        if summary or title:
            session.summary = SessionSummaryContent(title=title, body=summary)

        # Calculate stats if we have a history logger
        if self._history_logger:
            entries = self._history_logger.get_session_entries(session.id)
            tools_used = set()
            files_modified = set()
            for entry in entries:
                if entry.operation.type.value == "tool_call":
                    tools_used.add(entry.operation.name)
                files_modified.update(entry.metadata.files_touched)

            session.stats = SessionStats(
                operation_count=len(entries),
                duration_ms=self._calculate_duration(session),
                tools_used=list(tools_used),
                files_modified=list(files_modified),
            )

            self._history_logger.log_system_event(
                "session_completed",
                {
                    "session_id": session.id,
                    "operation_count": session.stats.operation_count,
                },
                session_id=session.id,
            )

        self._save_session(session)

        if self._current_session and self._current_session.id == session.id:
            self._current_session = None

        self._schedule_summary(session.id)

        return session

    def abort(
        self,
        reason: str,
        session_id: Optional[str] = None,
    ) -> Optional[SessionInfo]:
        """Abort a session due to error.

        Args:
            reason: Reason for abortion.
            session_id: Session to abort (uses current if not provided).

        Returns:
            The updated session info, or None if not found.
        """
        if session_id is None:
            session = self._current_session
        else:
            session = self._load_session(session_id)

        if session is None:
            return None

        session.status = SessionStatus.ABORTED
        session.updated_at = _get_iso_timestamp()
        session.extensions["abort_reason"] = reason
        self._save_session(session)

        if self._history_logger:
            self._history_logger.log_system_event(
                "session_aborted",
                {"session_id": session.id, "reason": reason},
                session_id=session.id,
            )

        if self._current_session and self._current_session.id == session.id:
            self._current_session = None

        return session

    def _calculate_duration(self, session: SessionInfo) -> int:
        """Calculate session duration in milliseconds."""
        try:
            created = datetime.fromisoformat(session.created_at)
            updated = datetime.fromisoformat(session.updated_at)
            return int((updated - created).total_seconds() * 1000)
        except Exception:
            return 0

    def _schedule_summary(self, session_id: str) -> None:
        """Schedule session summarization in the background."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        context_root = self.sessions_dir.parent.parent
        summary_path = context_root / "history" / "summaries" / f"{session_id}.json"
        if summary_path.exists():
            return

        async def _summarize() -> None:
            try:
                from core.history.summaries import HistorySessionSummaryIndex

                index = HistorySessionSummaryIndex(context_root)
                await index.summarize_session(session_id)
            except Exception as exc:
                logger.warning("Session summary failed for %s: %s", session_id, exc)

        loop.create_task(_summarize())

    def get(self, session_id: str) -> Optional[SessionInfo]:
        """Get a session by ID."""
        return self._load_session(session_id)

    def list_sessions(
        self,
        status: Optional[SessionStatus] = None,
        limit: int = 50,
    ) -> list[SessionInfo]:
        """List sessions, optionally filtered by status.

        Args:
            status: Filter by status.
            limit: Maximum number of sessions to return.

        Returns:
            List of session info objects.
        """
        sessions: list[SessionInfo] = []

        for session_file in sorted(
            self.sessions_dir.glob("*.json"), reverse=True
        ):
            try:
                with open(session_file, encoding="utf-8") as f:
                    session = SessionInfo.model_validate_json(f.read())
                if status is None or session.status == status:
                    sessions.append(session)
                if len(sessions) >= limit:
                    break
            except Exception:
                continue

        return sessions

    def get_or_create(self) -> SessionInfo:
        """Get the current session or create a new one."""
        if self._current_session is not None:
            return self._current_session
        return self.create()
