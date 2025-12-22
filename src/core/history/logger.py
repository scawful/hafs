"""History Logger - Append-only JSONL logging.

Implements the history logging per PROTOCOL_SPEC.md Section 2.3 and 2.4.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from core.history.models import (
    HistoryEntry,
    HistoryQuery,
    Operation,
    OperationMetadata,
    OperationType,
    Provenance,
)

if TYPE_CHECKING:
    from core.history.session import SessionManager


def _generate_ulid() -> str:
    """Generate a ULID-like identifier.

    Uses timestamp prefix for sortability + random suffix.
    """
    import random
    import string

    # Timestamp component (first 10 chars)
    timestamp_ms = int(time.time() * 1000)
    # Convert to Crockford Base32-like encoding
    chars = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    timestamp_part = ""
    for _ in range(10):
        timestamp_part = chars[timestamp_ms % 32] + timestamp_part
        timestamp_ms //= 32

    # Random component (last 16 chars)
    random_part = "".join(random.choices(chars, k=16))

    return timestamp_part + random_part


def _get_iso_timestamp() -> str:
    """Get current time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


class HistoryLogger:
    """Append-only history logger using JSONL format.

    Writes to daily log files in the history directory.
    Thread-safe through file append mode.
    """

    def __init__(
        self,
        history_dir: Path | str,
        session_manager: Optional[SessionManager] = None,
        project_id: Optional[str] = None,
    ) -> None:
        """Initialize the history logger.

        Args:
            history_dir: Directory to store history files.
            session_manager: Optional session manager for session tracking.
            project_id: Optional project identifier for entries.
        """
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.session_manager = session_manager
        self.project_id = project_id

        # In-flight operations (for tracking duration)
        self._in_flight: dict[str, tuple[str, float]] = {}

    def _get_log_file(self, date: Optional[datetime] = None) -> Path:
        """Get the log file path for a given date.

        Uses daily rotation: YYYY-MM-DD.jsonl
        """
        if date is None:
            date = datetime.now(timezone.utc)
        filename = date.strftime("%Y-%m-%d.jsonl")
        return self.history_dir / filename

    def _append_entry(self, entry: HistoryEntry) -> None:
        """Append an entry to the log file."""
        log_file = self._get_log_file()
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

    def log(
        self,
        operation_type: OperationType,
        name: str,
        input_data: Optional[dict[str, Any]] = None,
        output: Optional[Any] = None,
        success: bool = True,
        error: Optional[str] = None,
        duration_ms: Optional[int] = None,
        provenance: Optional[Provenance] = None,
        tags: Optional[list[str]] = None,
        files_touched: Optional[list[str]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Log an operation to history.

        Args:
            operation_type: Type of operation.
            name: Name of the operation (tool name, event type, etc.).
            input_data: Input parameters.
            output: Operation result (may be truncated).
            success: Whether operation succeeded.
            error: Error message if failed.
            duration_ms: Duration in milliseconds.
            provenance: Origin tracking info.
            tags: Tags for retrieval.
            files_touched: Files affected.
            session_id: Override session ID (uses current session if not provided).

        Returns:
            The entry ID (ULID).
        """
        entry_id = _generate_ulid()

        # Get session ID from manager if not provided
        if session_id is None and self.session_manager is not None:
            session_id = self.session_manager.current_session_id
        if session_id is None:
            session_id = "unknown"

        entry = HistoryEntry(
            id=entry_id,
            timestamp=_get_iso_timestamp(),
            session_id=session_id,
            project_id=self.project_id,
            operation=Operation(
                type=operation_type,
                name=name,
                input=input_data or {},
                output=output,
                duration_ms=duration_ms,
                success=success,
                error=error,
            ),
            provenance=provenance or Provenance(),
            metadata=OperationMetadata(
                tags=tags or [],
                files_touched=files_touched or [],
            ),
        )

        self._append_entry(entry)
        return entry_id

    def log_tool_start(
        self,
        tool_name: str,
        params: dict[str, Any],
        provenance: Optional[Provenance] = None,
    ) -> str:
        """Log the start of a tool call (for duration tracking).

        Returns entry_id to use with log_tool_complete.
        """
        entry_id = _generate_ulid()
        self._in_flight[entry_id] = (tool_name, time.time())

        # Log the start event
        self.log(
            operation_type=OperationType.TOOL_CALL,
            name=f"{tool_name}:start",
            input_data=params,
            provenance=provenance,
        )

        return entry_id

    def log_tool_complete(
        self,
        entry_id: str,
        result: Any,
        success: bool = True,
        error: Optional[str] = None,
        files_touched: Optional[list[str]] = None,
    ) -> None:
        """Log the completion of a tool call."""
        if entry_id not in self._in_flight:
            # Fallback if start wasn't tracked
            self.log(
                operation_type=OperationType.TOOL_CALL,
                name="unknown:complete",
                output=result,
                success=success,
                error=error,
                files_touched=files_touched,
            )
            return

        tool_name, start_time = self._in_flight.pop(entry_id)
        duration_ms = int((time.time() - start_time) * 1000)

        self.log(
            operation_type=OperationType.TOOL_CALL,
            name=tool_name,
            output=result,
            success=success,
            error=error,
            duration_ms=duration_ms,
            files_touched=files_touched,
        )

    def log_cognitive_state(
        self,
        state_type: str,
        state_data: dict[str, Any],
        session_id: Optional[str] = None,
    ) -> str:
        """Log a cognitive state snapshot.

        Args:
            state_type: Type of state (emotions, metacognition, epistemic, goals).
            state_data: The state data to log.
            session_id: Optional session ID override.

        Returns:
            The entry ID.
        """
        return self.log(
            operation_type=OperationType.COGNITIVE_STATE,
            name=state_type,
            input_data=state_data,
            session_id=session_id,
        )

    def log_user_input(
        self,
        message: str,
        session_id: Optional[str] = None,
    ) -> str:
        """Log user input."""
        return self.log(
            operation_type=OperationType.USER_INPUT,
            name="user_message",
            input_data={"message": message},
            session_id=session_id,
        )

    def log_agent_message(
        self,
        agent_id: str,
        message: str,
        session_id: Optional[str] = None,
    ) -> str:
        """Log agent message."""
        return self.log(
            operation_type=OperationType.AGENT_MESSAGE,
            name="agent_message",
            input_data={"message": message},
            provenance=Provenance(agent_id=agent_id),
            session_id=session_id,
        )

    def log_system_event(
        self,
        event_name: str,
        data: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Log a system event."""
        return self.log(
            operation_type=OperationType.SYSTEM_EVENT,
            name=event_name,
            input_data=data or {},
            session_id=session_id,
        )

    def log_thought_trace(
        self,
        thought_content: str,
        provider: str,
        model: str,
        prompt_preview: Optional[str] = None,
        response_preview: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> str:
        """Log a thought/reasoning trace from Gemini 3 or similar models.

        Args:
            thought_content: The reasoning trace content.
            provider: AI provider (gemini, anthropic, etc.).
            model: Model name that generated the thought.
            prompt_preview: Preview of the original prompt (truncated).
            response_preview: Preview of the response (truncated).
            session_id: Optional session ID override.
            tags: Optional tags for categorization.

        Returns:
            The entry ID.
        """
        return self.log(
            operation_type=OperationType.THOUGHT_TRACE,
            name=f"{provider}:{model}",
            input_data={
                "thought_content": thought_content,
                "provider": provider,
                "model": model,
                "prompt_preview": prompt_preview[:500] if prompt_preview else None,
                "response_preview": response_preview[:500] if response_preview else None,
            },
            output=thought_content,
            provenance=Provenance(model_id=model),
            session_id=session_id,
            tags=tags or ["thought_trace", provider, model],
        )

    def query(self, query: HistoryQuery) -> list[HistoryEntry]:
        """Query history entries.

        Args:
            query: Query parameters.

        Returns:
            List of matching history entries.
        """
        results: list[HistoryEntry] = []

        # Determine which log files to search
        log_files = sorted(self.history_dir.glob("*.jsonl"), reverse=True)

        for log_file in log_files:
            # Check date range
            file_date_str = log_file.stem  # YYYY-MM-DD
            try:
                file_date = datetime.strptime(file_date_str, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
                if query.start_time and file_date.date() < query.start_time.date():
                    continue
                if query.end_time and file_date.date() > query.end_time.date():
                    continue
            except ValueError:
                continue

            # Read and filter entries
            with open(log_file, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = HistoryEntry.model_validate_json(line)
                    except Exception:
                        continue

                    # Apply filters
                    if query.session_id and entry.session_id != query.session_id:
                        continue
                    if query.project_id and entry.project_id != query.project_id:
                        continue
                    if query.operation_types and entry.operation.type not in query.operation_types:
                        continue
                    if query.tags and not any(t in entry.metadata.tags for t in query.tags):
                        continue

                    results.append(entry)

                    if len(results) >= query.limit + query.offset:
                        break

            if len(results) >= query.limit + query.offset:
                break

        # Apply offset and limit
        return results[query.offset : query.offset + query.limit]

    def get_session_entries(self, session_id: str) -> list[HistoryEntry]:
        """Get all entries for a session."""
        return self.query(HistoryQuery(session_id=session_id, limit=10000))

    def get_recent_entries(self, limit: int = 50) -> list[HistoryEntry]:
        """Get the most recent entries."""
        return self.query(HistoryQuery(limit=limit))
