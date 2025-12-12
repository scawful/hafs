"""Gemini log data models (ported from halext-code C++)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class GeminiMessage(BaseModel):
    """A single message from a Gemini chat session."""

    id: str
    timestamp: datetime
    type: str  # "user" or "gemini"
    content: str = ""
    thoughts: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    model: str = ""
    total_tokens: int = 0

    model_config = ConfigDict(frozen=True)

    @property
    def is_user(self) -> bool:
        """Check if this is a user message."""
        return self.type == "user"

    @property
    def is_gemini(self) -> bool:
        """Check if this is a Gemini response."""
        return self.type == "gemini"

    @property
    def has_tool_calls(self) -> bool:
        """Check if message has tool calls."""
        return len(self.tool_names) > 0


class GeminiSession(BaseModel):
    """A complete Gemini chat session."""

    session_id: str
    project_hash: str
    start_time: datetime
    last_updated: datetime
    messages: list[GeminiMessage] = Field(default_factory=list)
    source_path: Path | None = None  # Path to the source JSON file

    @property
    def user_message_count(self) -> int:
        """Count user messages in session."""
        return sum(1 for m in self.messages if m.is_user)

    @property
    def gemini_message_count(self) -> int:
        """Count Gemini responses in session."""
        return sum(1 for m in self.messages if m.is_gemini)

    @property
    def total_tokens(self) -> int:
        """Sum of all tokens used in session."""
        return sum(m.total_tokens for m in self.messages)

    @property
    def tool_call_count(self) -> int:
        """Total number of tool calls across messages."""
        return sum(len(m.tool_names) for m in self.messages)

    @property
    def models_used(self) -> set[str]:
        """Set of models used in the session."""
        return {m.model for m in self.messages if m.model}

    @property
    def duration(self) -> float:
        """Session duration in seconds."""
        return (self.last_updated - self.start_time).total_seconds()

    @property
    def short_id(self) -> str:
        """Short version of session ID for display."""
        return self.session_id[:8] if len(self.session_id) > 8 else self.session_id


class GeminiProject(BaseModel):
    """A Gemini project directory with its sessions."""

    project_hash: str
    path: Path
    sessions: list[GeminiSession] = Field(default_factory=list)

    @property
    def total_sessions(self) -> int:
        """Count sessions in project."""
        return len(self.sessions)

    @property
    def total_messages(self) -> int:
        """Count all messages across sessions."""
        return sum(len(s.messages) for s in self.sessions)

    @property
    def short_hash(self) -> str:
        """Short version of project hash for display."""
        return self.project_hash[:8] if len(self.project_hash) > 8 else self.project_hash
