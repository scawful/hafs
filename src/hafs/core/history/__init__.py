"""History Pipeline for AFS Cognitive Protocol v0.2.

This module implements the immutable episodic memory layer as defined in PROTOCOL_SPEC.md.
History entries capture all agent operations for retrieval and analysis.
"""

from hafs.core.history.models import (
    HistoryEntry,
    OperationType,
    Provenance,
    OperationMetadata,
    SessionInfo,
    SessionStatus,
    SessionSummary,
    HistoryQuery,
)
from hafs.core.history.logger import HistoryLogger
from hafs.core.history.session import SessionManager

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
]
