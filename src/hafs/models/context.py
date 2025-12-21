"""Context Engineering Models based on AFS research.

Implements memory type taxonomy and context item models from
"Everything is Context: Agentic File System Abstraction for Context Engineering"

Memory Types:
- SCRATCHPAD: Volatile working memory (plans, todos, reasoning)
- EPISODIC: Interaction history (conversation turns, session logs)
- FACT: Static knowledge (documentation, reference material)
- EXPERIENTIAL: Learned patterns (successful strategies, failure cases)
- PROCEDURAL: Workflows and procedures (how-to guides, runbooks)
- USER: User preferences and profile (settings, past decisions)
- HISTORICAL: Long-term archival memory (compressed older context)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4


class MemoryType(str, Enum):
    """Extended memory type taxonomy for context engineering.

    Based on AFS research paper's memory classification:
    - Different types have different retention policies
    - Different types have different access patterns
    - Different types have different compression strategies
    """

    # Volatile working memory
    SCRATCHPAD = "scratchpad"

    # Interaction history
    EPISODIC = "episodic"

    # Static knowledge
    FACT = "fact"

    # Learned patterns
    EXPERIENTIAL = "experiential"

    # Workflows/procedures
    PROCEDURAL = "procedural"

    # User preferences
    USER = "user"

    # Long-term archival
    HISTORICAL = "historical"


class ContextPriority(str, Enum):
    """Priority levels for context items."""

    CRITICAL = "critical"  # Always include
    HIGH = "high"          # Include if space permits
    MEDIUM = "medium"      # Include if relevant
    LOW = "low"            # Include only if abundant space
    BACKGROUND = "background"  # Compress or summarize


@dataclass
class ContextItem:
    """A single item in the context window.

    Represents a piece of context that can be included in prompts.
    Tracks metadata for prioritization and compression decisions.
    """

    id: UUID = field(default_factory=uuid4)
    content: str = ""
    memory_type: MemoryType = MemoryType.FACT
    priority: ContextPriority = ContextPriority.MEDIUM

    # Source tracking
    source_path: Optional[Path] = None
    source_type: str = "text"  # text, code, markdown, json

    # Token estimation (4 chars ~= 1 token)
    estimated_tokens: int = 0

    # Temporal metadata
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    # Relevance scoring
    relevance_score: float = 0.5  # 0-1, updated by evaluator
    access_count: int = 0

    # Compression state
    is_compressed: bool = False
    original_tokens: int = 0  # Tokens before compression
    compression_ratio: float = 1.0

    # Embedding for semantic search
    embedding: Optional[list[float]] = None

    def __post_init__(self) -> None:
        if not self.estimated_tokens and self.content:
            self.estimated_tokens = len(self.content) // 4
        if not self.original_tokens:
            self.original_tokens = self.estimated_tokens

    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = datetime.now()
        self.access_count += 1

    def is_expired(self) -> bool:
        """Check if item has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    @property
    def age_hours(self) -> float:
        """Hours since creation."""
        delta = datetime.now() - self.created_at
        return delta.total_seconds() / 3600

    @property
    def staleness(self) -> float:
        """Hours since last access."""
        delta = datetime.now() - self.accessed_at
        return delta.total_seconds() / 3600

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": str(self.id),
            "content": self.content,
            "memory_type": self.memory_type.value,
            "priority": self.priority.value,
            "source_path": str(self.source_path) if self.source_path else None,
            "source_type": self.source_type,
            "estimated_tokens": self.estimated_tokens,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "relevance_score": self.relevance_score,
            "access_count": self.access_count,
            "is_compressed": self.is_compressed,
            "original_tokens": self.original_tokens,
            "compression_ratio": self.compression_ratio,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextItem":
        """Deserialize from dictionary."""
        item = cls(
            content=data.get("content", ""),
            memory_type=MemoryType(data.get("memory_type", "fact")),
            priority=ContextPriority(data.get("priority", "medium")),
            source_type=data.get("source_type", "text"),
            estimated_tokens=data.get("estimated_tokens", 0),
            relevance_score=data.get("relevance_score", 0.5),
            access_count=data.get("access_count", 0),
            is_compressed=data.get("is_compressed", False),
            original_tokens=data.get("original_tokens", 0),
            compression_ratio=data.get("compression_ratio", 1.0),
        )

        if data.get("id"):
            item.id = UUID(data["id"])
        if data.get("source_path"):
            item.source_path = Path(data["source_path"])
        if data.get("created_at"):
            item.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("accessed_at"):
            item.accessed_at = datetime.fromisoformat(data["accessed_at"])
        if data.get("expires_at"):
            item.expires_at = datetime.fromisoformat(data["expires_at"])

        return item


@dataclass
class ContextWindow:
    """Represents the current context window state.

    Manages a collection of context items within token budget constraints.
    """

    items: list[ContextItem] = field(default_factory=list)
    max_tokens: int = 128000  # Default to 128k
    reserved_tokens: int = 4000  # Reserve for system prompt + response

    @property
    def total_tokens(self) -> int:
        """Total tokens used by all items."""
        return sum(item.estimated_tokens for item in self.items)

    @property
    def available_tokens(self) -> int:
        """Tokens available for context."""
        return self.max_tokens - self.reserved_tokens

    @property
    def used_percentage(self) -> float:
        """Percentage of available tokens used."""
        available = self.available_tokens
        if available <= 0:
            return 100.0
        return (self.total_tokens / available) * 100

    @property
    def remaining_tokens(self) -> int:
        """Tokens remaining in budget."""
        return self.available_tokens - self.total_tokens

    def can_fit(self, item: ContextItem) -> bool:
        """Check if an item can fit in remaining budget."""
        return item.estimated_tokens <= self.remaining_tokens

    def add(self, item: ContextItem) -> bool:
        """Add item if it fits. Returns success status."""
        if self.can_fit(item):
            self.items.append(item)
            return True
        return False

    def remove(self, item_id: UUID) -> bool:
        """Remove item by ID."""
        for i, item in enumerate(self.items):
            if item.id == item_id:
                self.items.pop(i)
                return True
        return False

    def get_by_type(self, memory_type: MemoryType) -> list[ContextItem]:
        """Get all items of a specific type."""
        return [item for item in self.items if item.memory_type == memory_type]

    def get_by_priority(self, priority: ContextPriority) -> list[ContextItem]:
        """Get all items of a specific priority."""
        return [item for item in self.items if item.priority == priority]

    def sort_by_relevance(self) -> None:
        """Sort items by relevance score (highest first)."""
        self.items.sort(key=lambda x: x.relevance_score, reverse=True)

    def sort_by_recency(self) -> None:
        """Sort items by access time (most recent first)."""
        self.items.sort(key=lambda x: x.accessed_at, reverse=True)

    def prune_expired(self) -> int:
        """Remove expired items. Returns count removed."""
        original_count = len(self.items)
        self.items = [item for item in self.items if not item.is_expired()]
        return original_count - len(self.items)

    def to_prompt(self, separator: str = "\n\n---\n\n") -> str:
        """Render context window as prompt text."""
        return separator.join(item.content for item in self.items if item.content)


@dataclass
class TokenBudget:
    """Token budget allocation for different context types.

    Manages how tokens are allocated across memory types.
    """

    total_budget: int = 128000
    reserved_output: int = 4000  # Space for model response
    reserved_system: int = 2000  # System prompt overhead

    # Type allocations (percentages of available)
    scratchpad_pct: float = 0.15
    episodic_pct: float = 0.25
    fact_pct: float = 0.20
    experiential_pct: float = 0.10
    procedural_pct: float = 0.10
    user_pct: float = 0.10
    historical_pct: float = 0.10

    @property
    def available(self) -> int:
        """Tokens available for context."""
        return self.total_budget - self.reserved_output - self.reserved_system

    def get_allocation(self, memory_type: MemoryType) -> int:
        """Get token allocation for a memory type."""
        pct_map = {
            MemoryType.SCRATCHPAD: self.scratchpad_pct,
            MemoryType.EPISODIC: self.episodic_pct,
            MemoryType.FACT: self.fact_pct,
            MemoryType.EXPERIENTIAL: self.experiential_pct,
            MemoryType.PROCEDURAL: self.procedural_pct,
            MemoryType.USER: self.user_pct,
            MemoryType.HISTORICAL: self.historical_pct,
        }
        return int(self.available * pct_map.get(memory_type, 0.1))

    def rebalance(self, usage: dict[MemoryType, int]) -> dict[MemoryType, int]:
        """Rebalance allocations based on actual usage.

        Redistributes unused tokens from under-utilized types
        to over-utilized types.

        Args:
            usage: Actual tokens used per type

        Returns:
            New allocations per type
        """
        allocations = {}
        surplus = 0
        deficits: dict[MemoryType, int] = {}

        # Calculate surplus and deficits
        for mt in MemoryType:
            allocated = self.get_allocation(mt)
            used = usage.get(mt, 0)

            if used < allocated:
                surplus += allocated - used
                allocations[mt] = used
            else:
                deficits[mt] = used - allocated
                allocations[mt] = allocated

        # Redistribute surplus to deficits (proportionally)
        if surplus > 0 and deficits:
            total_deficit = sum(deficits.values())
            for mt, deficit in deficits.items():
                extra = int(surplus * (deficit / total_deficit))
                allocations[mt] += extra

        return allocations
