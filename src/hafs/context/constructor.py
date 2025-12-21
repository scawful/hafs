"""Context Constructor for the Context Engineering Pipeline.

Implements the Constructor phase from AFS research:
- Selection: Choose relevant context items
- Prioritization: Rank items by importance
- Compression: Summarize low-priority items to fit budget

Based on "Everything is Context: Agentic File System Abstraction"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from hafs.models.context import (
    ContextItem,
    ContextPriority,
    ContextWindow,
    MemoryType,
    TokenBudget,
)

logger = logging.getLogger(__name__)


@dataclass
class SelectionCriteria:
    """Criteria for context selection."""

    # Query for relevance matching
    query: str = ""

    # Memory types to include
    include_types: set[MemoryType] = field(
        default_factory=lambda: set(MemoryType)
    )

    # Minimum relevance score (0-1)
    min_relevance: float = 0.0

    # Maximum age in hours (None = no limit)
    max_age_hours: Optional[float] = None

    # Only include items accessed in last N hours
    recently_accessed_hours: Optional[float] = None

    # Include expired items
    include_expired: bool = False


@dataclass
class ConstructorConfig:
    """Configuration for context constructor."""

    # Token budget
    token_budget: TokenBudget = field(default_factory=TokenBudget)

    # Compression threshold (compress items when budget < this %)
    compression_threshold_pct: float = 0.2

    # Minimum tokens to keep per item after compression
    min_compressed_tokens: int = 50

    # Target compression ratio
    target_compression_ratio: float = 0.3

    # Priority weights for scoring
    priority_weights: dict[ContextPriority, float] = field(
        default_factory=lambda: {
            ContextPriority.CRITICAL: 1.0,
            ContextPriority.HIGH: 0.8,
            ContextPriority.MEDIUM: 0.5,
            ContextPriority.LOW: 0.3,
            ContextPriority.BACKGROUND: 0.1,
        }
    )

    # Memory type weights for scoring
    type_weights: dict[MemoryType, float] = field(
        default_factory=lambda: {
            MemoryType.SCRATCHPAD: 0.9,  # Current working memory
            MemoryType.USER: 0.85,        # User preferences
            MemoryType.EPISODIC: 0.8,     # Recent interactions
            MemoryType.PROCEDURAL: 0.7,   # How-to knowledge
            MemoryType.FACT: 0.6,         # Reference material
            MemoryType.EXPERIENTIAL: 0.5, # Learned patterns
            MemoryType.HISTORICAL: 0.3,   # Archival
        }
    )


class ContextConstructor:
    """Constructs optimized context windows from available items.

    The Constructor phase of the Context Engineering Pipeline:
    1. Selects relevant items based on criteria
    2. Prioritizes items using multi-factor scoring
    3. Compresses low-priority items if needed
    4. Assembles final context window within budget

    Example:
        constructor = ContextConstructor()
        window = constructor.construct(
            items=all_context_items,
            criteria=SelectionCriteria(query="implement auth"),
        )
    """

    def __init__(
        self,
        config: Optional[ConstructorConfig] = None,
        compressor: Optional[Callable[[str, int], str]] = None,
    ):
        """Initialize the constructor.

        Args:
            config: Constructor configuration
            compressor: Optional function(text, target_tokens) -> compressed_text
                        For LLM-based compression, inject a compressor function
        """
        self.config = config or ConstructorConfig()
        self._compressor = compressor or self._simple_compress

    def construct(
        self,
        items: list[ContextItem],
        criteria: Optional[SelectionCriteria] = None,
    ) -> ContextWindow:
        """Construct an optimized context window.

        Args:
            items: All available context items
            criteria: Selection criteria

        Returns:
            Optimized ContextWindow within budget
        """
        criteria = criteria or SelectionCriteria()

        # Phase 1: Select
        selected = self._select(items, criteria)
        logger.debug(f"Selected {len(selected)} of {len(items)} items")

        # Phase 2: Prioritize
        prioritized = self._prioritize(selected, criteria.query)
        logger.debug(f"Prioritized {len(prioritized)} items")

        # Phase 3: Fit to budget (with compression if needed)
        window = self._fit_to_budget(prioritized)
        logger.info(
            f"Constructed context window: {window.total_tokens} tokens "
            f"({window.used_percentage:.1f}% of available)"
        )

        return window

    def _select(
        self,
        items: list[ContextItem],
        criteria: SelectionCriteria,
    ) -> list[ContextItem]:
        """Select items matching criteria."""
        selected = []

        for item in items:
            # Type filter
            if item.memory_type not in criteria.include_types:
                continue

            # Expiration filter
            if not criteria.include_expired and item.is_expired():
                continue

            # Relevance filter
            if item.relevance_score < criteria.min_relevance:
                continue

            # Age filter
            if criteria.max_age_hours is not None:
                if item.age_hours > criteria.max_age_hours:
                    continue

            # Recently accessed filter
            if criteria.recently_accessed_hours is not None:
                if item.staleness > criteria.recently_accessed_hours:
                    continue

            selected.append(item)

        return selected

    def _prioritize(
        self,
        items: list[ContextItem],
        query: str = "",
    ) -> list[ContextItem]:
        """Prioritize items by multi-factor scoring.

        Scoring factors:
        - Priority weight
        - Memory type weight
        - Relevance score
        - Recency (inverse staleness)
        - Access frequency
        """
        scored_items: list[tuple[float, ContextItem]] = []

        for item in items:
            score = self._compute_score(item, query)
            scored_items.append((score, item))

        # Sort by score descending
        scored_items.sort(key=lambda x: x[0], reverse=True)

        return [item for _, item in scored_items]

    def _compute_score(self, item: ContextItem, query: str = "") -> float:
        """Compute priority score for an item."""
        # Base weights
        priority_weight = self.config.priority_weights.get(
            item.priority, 0.5
        )
        type_weight = self.config.type_weights.get(
            item.memory_type, 0.5
        )

        # Relevance (0-1)
        relevance = item.relevance_score

        # Recency factor (decay with staleness)
        # 0 hours -> 1.0, 24 hours -> 0.5, 168 hours (1 week) -> 0.1
        recency = max(0.1, 1.0 - (item.staleness / 168))

        # Access frequency factor (log scale)
        import math
        frequency = min(1.0, 0.2 + 0.2 * math.log1p(item.access_count))

        # Combine factors (weighted sum)
        score = (
            0.25 * priority_weight +
            0.20 * type_weight +
            0.30 * relevance +
            0.15 * recency +
            0.10 * frequency
        )

        return score

    def _fit_to_budget(
        self,
        prioritized: list[ContextItem],
    ) -> ContextWindow:
        """Fit items to token budget, compressing if needed."""
        window = ContextWindow(
            max_tokens=self.config.token_budget.total_budget,
            reserved_tokens=(
                self.config.token_budget.reserved_output +
                self.config.token_budget.reserved_system
            ),
        )

        # First pass: add items until budget exhausted
        remaining: list[ContextItem] = []
        for item in prioritized:
            if window.can_fit(item):
                window.add(item)
                item.touch()
            else:
                remaining.append(item)

        # If we still have important items, try compression
        if remaining and window.available_tokens > 0:
            budget_pct = window.remaining_tokens / window.available_tokens

            if budget_pct < self.config.compression_threshold_pct:
                # Budget is tight; try to compress and fit more items
                self._compress_and_fit(window, remaining)

        return window

    def _compress_and_fit(
        self,
        window: ContextWindow,
        items: list[ContextItem],
    ) -> None:
        """Compress items and add to window if they fit."""
        for item in items:
            # Only compress if item is compressible
            if item.priority in (
                ContextPriority.CRITICAL,
                ContextPriority.HIGH,
            ):
                # Don't compress critical/high priority
                continue

            if item.is_compressed:
                # Already compressed
                continue

            # Calculate target tokens
            target_tokens = int(
                item.estimated_tokens * self.config.target_compression_ratio
            )
            target_tokens = max(target_tokens, self.config.min_compressed_tokens)

            if target_tokens > window.remaining_tokens:
                # Won't fit even compressed
                continue

            # Compress
            compressed_content = self._compressor(
                item.content,
                target_tokens,
            )

            # Create compressed item
            compressed_item = ContextItem(
                id=item.id,
                content=compressed_content,
                memory_type=item.memory_type,
                priority=item.priority,
                source_path=item.source_path,
                source_type=item.source_type,
                estimated_tokens=len(compressed_content) // 4,
                created_at=item.created_at,
                accessed_at=item.accessed_at,
                relevance_score=item.relevance_score,
                access_count=item.access_count,
                is_compressed=True,
                original_tokens=item.original_tokens,
                compression_ratio=(
                    len(compressed_content) // 4
                ) / max(1, item.original_tokens),
            )

            if window.can_fit(compressed_item):
                window.add(compressed_item)
                item.touch()

    def _simple_compress(self, text: str, target_tokens: int) -> str:
        """Simple truncation-based compression.

        For better results, inject an LLM-based compressor.
        """
        target_chars = target_tokens * 4

        if len(text) <= target_chars:
            return text

        # Try to split on paragraph boundaries
        paragraphs = text.split("\n\n")

        if len(paragraphs) > 2:
            # Keep first and last paragraphs, summarize middle
            first = paragraphs[0]
            last = paragraphs[-1]
            middle_count = len(paragraphs) - 2

            result = f"{first}\n\n[...{middle_count} sections omitted...]\n\n{last}"

            if len(result) <= target_chars:
                return result

        # Fall back to simple truncation
        return text[:target_chars - 20] + "\n\n[...truncated...]"

    def construct_for_agent(
        self,
        items: list[ContextItem],
        agent_role: str,
        task: Optional[str] = None,
    ) -> ContextWindow:
        """Construct context optimized for a specific agent role.

        Args:
            items: Available context items
            agent_role: Agent role (planner, coder, critic, etc.)
            task: Current task description

        Returns:
            Role-optimized context window
        """
        # Adjust type weights based on role
        role_type_preferences = self._get_role_preferences(agent_role)

        # Create role-specific config
        role_config = ConstructorConfig(
            token_budget=self.config.token_budget,
            type_weights={
                mt: self.config.type_weights.get(mt, 0.5) * role_type_preferences.get(mt, 1.0)
                for mt in MemoryType
            },
        )

        # Use role-specific constructor
        role_constructor = ContextConstructor(
            config=role_config,
            compressor=self._compressor,
        )

        criteria = SelectionCriteria(query=task or "")

        return role_constructor.construct(items, criteria)

    def _get_role_preferences(
        self,
        role: str,
    ) -> dict[MemoryType, float]:
        """Get memory type preference multipliers for a role."""
        role = role.lower()

        if role == "planner":
            return {
                MemoryType.EPISODIC: 1.5,   # Past interactions
                MemoryType.PROCEDURAL: 1.3,  # Workflows
                MemoryType.EXPERIENTIAL: 1.2,  # Learned patterns
            }
        elif role == "coder":
            return {
                MemoryType.SCRATCHPAD: 1.5,  # Current plans
                MemoryType.FACT: 1.3,         # Documentation
                MemoryType.PROCEDURAL: 1.2,   # How-to
            }
        elif role == "critic":
            return {
                MemoryType.EXPERIENTIAL: 1.5,  # Past failures/successes
                MemoryType.FACT: 1.3,           # Standards
                MemoryType.EPISODIC: 1.2,       # History
            }
        elif role == "researcher":
            return {
                MemoryType.FACT: 1.5,           # Knowledge
                MemoryType.HISTORICAL: 1.3,     # Archives
                MemoryType.EXPERIENTIAL: 1.2,   # Patterns
            }

        return {}


class ContextStore:
    """Persistent storage for context items.

    Manages context items on disk in ~/.context/items/
    """

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize context store.

        Args:
            base_path: Base path for storage (default ~/.context)
        """
        self.base_path = base_path or Path.home() / ".context"
        self.items_path = self.base_path / "items"
        self.items_path.mkdir(parents=True, exist_ok=True)

        self._items: dict[str, ContextItem] = {}
        self._loaded = False

    def load(self) -> None:
        """Load items from disk."""
        import json

        for file in self.items_path.glob("*.json"):
            try:
                data = json.loads(file.read_text())
                item = ContextItem.from_dict(data)
                self._items[str(item.id)] = item
            except Exception as e:
                logger.warning(f"Failed to load context item {file}: {e}")

        self._loaded = True
        logger.info(f"Loaded {len(self._items)} context items")

    def save(self, item: ContextItem) -> None:
        """Save a single item to disk."""
        import json

        self._items[str(item.id)] = item

        file_path = self.items_path / f"{item.id}.json"
        file_path.write_text(json.dumps(item.to_dict(), indent=2))

    def get(self, item_id: str) -> Optional[ContextItem]:
        """Get item by ID."""
        if not self._loaded:
            self.load()
        return self._items.get(item_id)

    def get_all(self) -> list[ContextItem]:
        """Get all items."""
        if not self._loaded:
            self.load()
        return list(self._items.values())

    def get_by_type(self, memory_type: MemoryType) -> list[ContextItem]:
        """Get all items of a specific type."""
        if not self._loaded:
            self.load()
        return [
            item for item in self._items.values()
            if item.memory_type == memory_type
        ]

    def delete(self, item_id: str) -> bool:
        """Delete an item."""
        if item_id in self._items:
            del self._items[item_id]
            file_path = self.items_path / f"{item_id}.json"
            if file_path.exists():
                file_path.unlink()
            return True
        return False

    def prune_expired(self) -> int:
        """Remove expired items. Returns count removed."""
        expired = [
            item_id for item_id, item in self._items.items()
            if item.is_expired()
        ]

        for item_id in expired:
            self.delete(item_id)

        return len(expired)
