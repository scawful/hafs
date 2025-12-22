"""History Pipeline for AFS Cognitive Protocol v0.2.

This module implements the immutable episodic memory layer as defined in PROTOCOL_SPEC.md.
History entries capture all agent operations for retrieval and analysis.
"""

from core.history.models import (
    HistoryEntry,
    OperationType,
    Provenance,
    OperationMetadata,
    SessionInfo,
    SessionStatus,
    SessionSummary,
    HistoryQuery,
)
from core.history.logger import HistoryLogger
from core.history.session import SessionManager
from core.history.embeddings import HistoryEmbeddingIndex
from core.history.summaries import HistorySessionSummaryIndex
from core.history.agent_memory import (
    AgentMemory,
    AgentMemoryManager,
    MemoryEntry,
)

__all__ = [
    "HistoryEntry",
    "OperationType",
    "Provenance",
    "OperationMetadata",
    "SessionInfo",
    "SessionStatus",
    "SessionSummary",
    "HistoryLogger",
    "SessionManager",
    "HistoryQuery",
    "HistoryEmbeddingIndex",
    "HistorySessionSummaryIndex",
    # Agent memory
    "AgentMemory",
    "AgentMemoryManager",
    "MemoryEntry",
]
