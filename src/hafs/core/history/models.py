"""Data models for the History Pipeline.

Defines schemas per PROTOCOL_SPEC.md Section 2 (History Pipeline)
and Section 3 (Session Management).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class OperationType(str, Enum):
    """Types of operations that can be logged to history."""

    TOOL_CALL = "tool_call"
    AGENT_MESSAGE = "agent_message"
    USER_INPUT = "user_input"
    SYSTEM_EVENT = "system_event"
    COGNITIVE_STATE = "cognitive_state"


class Provenance(BaseModel):
    """Tracks the origin of a history entry."""

    agent_id: Optional[str] = None
    model_id: Optional[str] = None
    parent_message_id: Optional[str] = None
    parent_entry_id: Optional[str] = None


class OperationMetadata(BaseModel):
    """Metadata for retrieval and filtering."""

    tags: list[str] = Field(default_factory=list)
    files_touched: list[str] = Field(default_factory=list)
    token_count: Optional[int] = None
    redacted: bool = False


class Operation(BaseModel):
    """Details of the operation performed."""

    type: OperationType
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: Optional[Any] = None
    duration_ms: Optional[int] = None
    success: bool = True
    error: Optional[str] = None


class HistoryEntry(BaseModel):
    """A single entry in the history log.

    Represents an immutable record of an agent operation per PROTOCOL_SPEC.md Section 2.1.
    """

    id: str  # ULID
    timestamp: str  # ISO 8601
    session_id: str
    project_id: Optional[str] = None

    operation: Operation
    provenance: Provenance = Field(default_factory=Provenance)
    metadata: OperationMetadata = Field(default_factory=OperationMetadata)

    extensions: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        extra = "allow"


class SessionStatus(str, Enum):
    """Session lifecycle states."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    ABORTED = "aborted"


class SessionStats(BaseModel):
    """Computed statistics for a session."""

    operation_count: int = 0
    duration_ms: int = 0
    files_modified: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)


class SessionSummaryContent(BaseModel):
    """LLM-generated session summary."""

    title: Optional[str] = None
    body: Optional[str] = None


class SessionInfo(BaseModel):
    """Information about a session.

    Per PROTOCOL_SPEC.md Section 3.1.
    """

    id: str  # ULID
    project_id: Optional[str] = None
    created_at: str  # ISO 8601
    updated_at: str  # ISO 8601

    status: SessionStatus = SessionStatus.ACTIVE
    parent_session_id: Optional[str] = None

    stats: SessionStats = Field(default_factory=SessionStats)
    summary: Optional[SessionSummaryContent] = None

    extensions: dict[str, Any] = Field(default_factory=dict)


class SessionEntity(BaseModel):
    """An entity extracted from a session for knowledge graph."""

    name: str
    type: str  # "file" | "function" | "concept" | "person" | custom
    mentions: int = 1


class SessionSummary(BaseModel):
    """Full session summary with embeddings.

    Per PROTOCOL_SPEC.md Section 3.3.
    """

    session_id: str
    project_id: Optional[str] = None
    created_at: str

    summary: str
    entities: list[SessionEntity] = Field(default_factory=list)
    stats: SessionStats = Field(default_factory=SessionStats)

    # Embedding (dimension flexible per implementation)
    embedding: Optional[list[float]] = None
    embedding_model: Optional[str] = None

    extensions: dict[str, Any] = Field(default_factory=dict)


class HistoryQuery(BaseModel):
    """Query parameters for history retrieval."""

    session_id: Optional[str] = None
    project_id: Optional[str] = None
    operation_types: Optional[list[OperationType]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    tags: Optional[list[str]] = None
    limit: int = 100
    offset: int = 0
