"""Token Budget Manager for context window optimization.

Provides intelligent token allocation and management for context windows.
Based on "Everything is Context: Agentic File System Abstraction" research.

Key features:
- Dynamic token allocation across memory types
- Budget rebalancing based on usage patterns
- Compression triggers and strategies
- Model-specific budget configurations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

from hafs.models.context import (
    ContextItem,
    ContextPriority,
    MemoryType,
    TokenBudget,
)

logger = logging.getLogger(__name__)


class ModelCapability(str, Enum):
    """Context window capabilities by model family."""

    # Small context windows (4k-8k)
    SMALL = "small"

    # Medium context windows (32k)
    MEDIUM = "medium"

    # Large context windows (128k)
    LARGE = "large"

    # Extra large context windows (200k+)
    XLARGE = "xlarge"


@dataclass
class ModelConfig:
    """Configuration for a specific model's token limits."""

    name: str
    capability: ModelCapability
    max_tokens: int
    max_output_tokens: int

    # Optimal usage range (percentage)
    optimal_min_pct: float = 0.6
    optimal_max_pct: float = 0.85

    # Cost per 1k tokens (input, output)
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


# Common model configurations
MODEL_CONFIGS = {
    "claude-opus-4-5": ModelConfig(
        name="Claude Opus 4.5",
        capability=ModelCapability.LARGE,
        max_tokens=200000,
        max_output_tokens=8192,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
    ),
    "claude-sonnet-4": ModelConfig(
        name="Claude Sonnet 4",
        capability=ModelCapability.LARGE,
        max_tokens=200000,
        max_output_tokens=8192,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
    ),
    "claude-haiku-3.5": ModelConfig(
        name="Claude Haiku 3.5",
        capability=ModelCapability.LARGE,
        max_tokens=200000,
        max_output_tokens=8192,
        cost_per_1k_input=0.0008,
        cost_per_1k_output=0.004,
    ),
    "gpt-4o": ModelConfig(
        name="GPT-4o",
        capability=ModelCapability.LARGE,
        max_tokens=128000,
        max_output_tokens=16384,
        cost_per_1k_input=0.0025,
        cost_per_1k_output=0.01,
    ),
    "gemini-2.0-flash": ModelConfig(
        name="Gemini 2.0 Flash",
        capability=ModelCapability.XLARGE,
        max_tokens=1000000,
        max_output_tokens=8192,
        cost_per_1k_input=0.0,  # Free tier
        cost_per_1k_output=0.0,
    ),
}


@dataclass
class BudgetAllocation:
    """Token budget allocation result."""

    # Per-type allocations
    allocations: dict[MemoryType, int] = field(default_factory=dict)

    # Total available
    total_available: int = 0

    # Reserved for system/output
    reserved_system: int = 0
    reserved_output: int = 0

    # Utilization stats
    current_usage: dict[MemoryType, int] = field(default_factory=dict)

    # Compression recommendations
    should_compress: dict[MemoryType, bool] = field(default_factory=dict)

    # Estimated cost
    estimated_cost: float = 0.0

    def get_remaining(self, memory_type: MemoryType) -> int:
        """Get remaining tokens for a memory type."""
        allocated = self.allocations.get(memory_type, 0)
        used = self.current_usage.get(memory_type, 0)
        return max(0, allocated - used)

    def utilization_pct(self, memory_type: MemoryType) -> float:
        """Get utilization percentage for a memory type."""
        allocated = self.allocations.get(memory_type, 1)
        used = self.current_usage.get(memory_type, 0)
        return (used / allocated) * 100 if allocated > 0 else 0.0


@dataclass
class BudgetManagerConfig:
    """Configuration for token budget manager."""

    # Model configuration
    model_config: ModelConfig = field(
        default_factory=lambda: MODEL_CONFIGS["claude-opus-4-5"]
    )

    # Base budget allocations (percentages of available)
    base_allocations: dict[MemoryType, float] = field(
        default_factory=lambda: {
            MemoryType.SCRATCHPAD: 0.15,
            MemoryType.EPISODIC: 0.25,
            MemoryType.FACT: 0.20,
            MemoryType.EXPERIENTIAL: 0.10,
            MemoryType.PROCEDURAL: 0.10,
            MemoryType.USER: 0.10,
            MemoryType.HISTORICAL: 0.10,
        }
    )

    # System prompt overhead
    system_prompt_tokens: int = 2000

    # Compression threshold (compress when type usage > allocation * threshold)
    compression_threshold: float = 0.9

    # Rebalancing aggressiveness (0-1)
    rebalance_factor: float = 0.3

    # Enable cost tracking
    track_costs: bool = True


class TokenBudgetManager:
    """Manages token budgets across memory types.

    Features:
    - Allocates tokens based on model capabilities
    - Rebalances based on actual usage patterns
    - Triggers compression when approaching limits
    - Tracks costs for budgeting

    Example:
        manager = TokenBudgetManager()

        # Get allocation for current state
        allocation = manager.allocate(items)

        # Check if we can add an item
        if manager.can_add(item, allocation):
            items.append(item)

        # Rebalance after usage changes
        manager.rebalance(items)
    """

    def __init__(
        self,
        config: Optional[BudgetManagerConfig] = None,
        token_counter: Optional[Callable[[str], int]] = None,
    ):
        """Initialize the budget manager.

        Args:
            config: Manager configuration
            token_counter: Optional function for accurate token counting
                          Default uses char/4 estimation
        """
        self.config = config or BudgetManagerConfig()
        self._token_counter = token_counter or (lambda s: len(s) // 4)

        # Usage history for rebalancing
        self._usage_history: list[dict[MemoryType, int]] = []

        # Cost tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._session_start = datetime.now()

    @property
    def max_tokens(self) -> int:
        """Maximum tokens for current model."""
        return self.config.model_config.max_tokens

    @property
    def available_tokens(self) -> int:
        """Tokens available for context (excluding reserved)."""
        return (
            self.max_tokens -
            self.config.system_prompt_tokens -
            self.config.model_config.max_output_tokens
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self._token_counter(text)

    def allocate(
        self,
        items: list[ContextItem],
        task: Optional[str] = None,
    ) -> BudgetAllocation:
        """Calculate token allocation for given items.

        Args:
            items: Current context items
            task: Optional task for prioritization

        Returns:
            BudgetAllocation with per-type allocations
        """
        allocation = BudgetAllocation(
            total_available=self.available_tokens,
            reserved_system=self.config.system_prompt_tokens,
            reserved_output=self.config.model_config.max_output_tokens,
        )

        # Calculate current usage
        for item in items:
            current = allocation.current_usage.get(item.memory_type, 0)
            allocation.current_usage[item.memory_type] = current + item.estimated_tokens

        # Calculate base allocations
        for mt, pct in self.config.base_allocations.items():
            allocation.allocations[mt] = int(self.available_tokens * pct)

        # Apply adaptive rebalancing
        if self._usage_history:
            self._apply_adaptive_rebalancing(allocation)

        # Determine compression needs
        for mt in MemoryType:
            allocated = allocation.allocations.get(mt, 0)
            used = allocation.current_usage.get(mt, 0)
            if allocated > 0 and used > allocated * self.config.compression_threshold:
                allocation.should_compress[mt] = True
            else:
                allocation.should_compress[mt] = False

        # Calculate estimated cost
        if self.config.track_costs:
            total_tokens = sum(allocation.current_usage.values())
            allocation.estimated_cost = (
                total_tokens / 1000 * self.config.model_config.cost_per_1k_input
            )

        # Record usage for history
        self._usage_history.append(dict(allocation.current_usage))
        # Keep last 20 records
        self._usage_history = self._usage_history[-20:]

        return allocation

    def _apply_adaptive_rebalancing(self, allocation: BudgetAllocation) -> None:
        """Rebalance allocations based on usage history."""
        if not self._usage_history:
            return

        # Calculate average usage per type
        avg_usage: dict[MemoryType, float] = {}
        for mt in MemoryType:
            usages = [h.get(mt, 0) for h in self._usage_history]
            avg_usage[mt] = sum(usages) / len(usages) if usages else 0

        # Find under-utilized and over-utilized types
        surplus = 0
        deficits: dict[MemoryType, int] = {}

        for mt, allocated in allocation.allocations.items():
            used = avg_usage.get(mt, 0)
            if used < allocated * 0.5:
                # Under-utilized: reclaim some
                reclaim = int((allocated - used) * self.config.rebalance_factor)
                allocation.allocations[mt] -= reclaim
                surplus += reclaim
            elif used > allocated * 0.9:
                # Over-utilized: needs more
                needed = int((used - allocated) * self.config.rebalance_factor)
                deficits[mt] = needed

        # Redistribute surplus to deficits
        if surplus > 0 and deficits:
            total_deficit = sum(deficits.values())
            for mt, deficit in deficits.items():
                extra = int(surplus * (deficit / total_deficit))
                allocation.allocations[mt] += extra

    def can_add(
        self,
        item: ContextItem,
        allocation: BudgetAllocation,
    ) -> bool:
        """Check if an item can be added within budget.

        Args:
            item: Item to check
            allocation: Current allocation

        Returns:
            True if item fits in budget
        """
        remaining = allocation.get_remaining(item.memory_type)
        return item.estimated_tokens <= remaining

    def get_compression_target(
        self,
        memory_type: MemoryType,
        allocation: BudgetAllocation,
    ) -> int:
        """Get target token count for compression.

        Args:
            memory_type: Type to compress
            allocation: Current allocation

        Returns:
            Target token count after compression
        """
        allocated = allocation.allocations.get(memory_type, 0)
        used = allocation.current_usage.get(memory_type, 0)

        # Target 70% of allocation
        target = int(allocated * 0.7)

        # At minimum, reduce by 30%
        min_target = int(used * 0.7)

        return max(target, min_target)

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int = 0,
    ) -> float:
        """Estimate cost for a request.

        Args:
            input_tokens: Input token count
            output_tokens: Expected output tokens

        Returns:
            Estimated cost in dollars
        """
        model = self.config.model_config
        cost = (
            input_tokens / 1000 * model.cost_per_1k_input +
            output_tokens / 1000 * model.cost_per_1k_output
        )
        return cost

    def track_usage(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Track token usage for cost reporting.

        Args:
            input_tokens: Tokens used for input
            output_tokens: Tokens used for output
        """
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens

    def get_usage_report(self) -> dict:
        """Get usage and cost report for current session.

        Returns:
            Report dictionary
        """
        model = self.config.model_config
        session_duration = (datetime.now() - self._session_start).total_seconds()

        total_cost = self.estimate_cost(
            self._total_input_tokens,
            self._total_output_tokens,
        )

        return {
            "model": model.name,
            "session_duration_seconds": session_duration,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
            "total_cost_usd": total_cost,
            "cost_per_hour_usd": (
                total_cost / (session_duration / 3600)
                if session_duration > 0 else 0.0
            ),
            "avg_tokens_per_minute": (
                (self._total_input_tokens + self._total_output_tokens) /
                (session_duration / 60)
                if session_duration > 0 else 0.0
            ),
        }

    def suggest_model(
        self,
        required_context: int,
        budget_per_hour: Optional[float] = None,
    ) -> ModelConfig:
        """Suggest optimal model for requirements.

        Args:
            required_context: Required context window size
            budget_per_hour: Optional cost budget per hour

        Returns:
            Suggested model configuration
        """
        candidates = []

        for model_id, config in MODEL_CONFIGS.items():
            # Check context fits
            if config.max_tokens < required_context:
                continue

            # Estimate hourly cost (assuming 60 requests/hour)
            hourly_cost = (
                60 * (required_context / 1000 * config.cost_per_1k_input +
                      1000 / 1000 * config.cost_per_1k_output)
            )

            if budget_per_hour and hourly_cost > budget_per_hour:
                continue

            candidates.append((hourly_cost, config))

        if not candidates:
            # Return largest available
            return MODEL_CONFIGS["gemini-2.0-flash"]

        # Sort by cost (cheapest first)
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]


def create_budget_for_model(model_id: str) -> TokenBudget:
    """Create a TokenBudget for a specific model.

    Args:
        model_id: Model identifier (e.g., "claude-opus-4-5")

    Returns:
        Configured TokenBudget
    """
    config = MODEL_CONFIGS.get(model_id, MODEL_CONFIGS["claude-opus-4-5"])

    return TokenBudget(
        total_budget=config.max_tokens,
        reserved_output=config.max_output_tokens,
        reserved_system=2000,
    )
