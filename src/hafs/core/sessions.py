"""Session Store for HAFS agentic sessions."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class SessionMetadata(BaseModel):
    """Metadata for a saved session."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    agent_ids: list[str] = Field(default_factory=list)
    task_summary: Optional[str] = None


class SessionData(BaseModel):
    """Full data for a saved session."""

    metadata: SessionMetadata
    messages: list[dict[str, Any]] = Field(default_factory=list)
    shared_context: dict[str, Any] = Field(default_factory=dict)


class SessionStore:
    """Manages saving and loading of agentic sessions."""

    def __init__(self, storage_dir: Optional[Path] = None) -> None:
        if storage_dir is None:
            storage_dir = Path.home() / ".context" / "sessions"
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def list_sessions(self) -> list[SessionMetadata]:
        """List all saved sessions."""
        sessions = []
        for path in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                if "metadata" in data:
                    sessions.append(SessionMetadata.model_validate(data["metadata"]))
            except Exception:
                continue
        return sorted(sessions, key=lambda s: s.updated_at, reverse=True)

    def save_session(self, session: SessionData) -> Path:
        """Save a session to disk."""
        session.metadata.updated_at = datetime.now()
        path = self.storage_dir / f"{session.metadata.id}.json"
        path.write_text(session.model_dump_json(indent=2))
        return path

    def load_session(self, session_id: UUID) -> Optional[SessionData]:
        """Load a session from disk."""
        path = self.storage_dir / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            return SessionData.model_validate_json(path.read_text())
        except Exception:
            return None

    def delete_session(self, session_id: UUID) -> bool:
        """Delete a session from disk."""
        path = self.storage_dir / f"{session_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False
