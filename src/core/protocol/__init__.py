"""Cognitive protocol management."""

from core.protocol.io_manager import IOManager, get_io_manager
from core.protocol.validation import (
    SchemaValidator,
    validate_goals_file,
    validate_metacognition_file,
)

__all__ = [
    "IOManager",
    "get_io_manager",
    "SchemaValidator",
    "validate_goals_file",
    "validate_metacognition_file",
]
