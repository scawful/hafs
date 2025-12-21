"""Base Expert class for Mixture of Experts system."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from hafs.core.orchestrator_v2 import TaskTier, UnifiedOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class ExpertConfig:
    """Configuration for an expert agent."""

    name: str
    display_name: str
    specialization: str
    keywords: list[str]
    confidence_threshold: float
    model_name: Optional[str] = None
    lora_adapter_path: Optional[Path] = None
    tier: TaskTier = TaskTier.CODING
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass
class ExpertResponse:
    """Response from an expert."""

    expert_name: str
    content: str
    confidence: float
    reasoning: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class BaseExpert(ABC):
    """Base class for specialized expert agents.

    Each expert specializes in a specific domain (ASM, YAZE, Debug)
    and can be backed by either:
    - A fine-tuned model with LoRA adapters
    - A base model with domain-specific system prompt
    - A frontier model via orchestrator
    """

    def __init__(
        self,
        config: ExpertConfig,
        orchestrator: Optional[UnifiedOrchestrator] = None,
    ):
        """Initialize expert.

        Args:
            config: Expert configuration.
            orchestrator: Optional orchestrator for API routing.
        """
        self.config = config
        self.orchestrator = orchestrator or UnifiedOrchestrator()
        self.initialized = False

        logger.info(f"Created expert: {config.name} ({config.specialization})")

    async def initialize(self) -> None:
        """Initialize the expert (load model, connect to APIs, etc.)."""
        if self.initialized:
            return

        # Initialize orchestrator
        if not hasattr(self.orchestrator, '_initialized') or not self.orchestrator._initialized:
            await self.orchestrator.initialize()

        # Load LoRA adapter if specified
        if self.config.lora_adapter_path:
            await self._load_adapter(self.config.lora_adapter_path)

        self.initialized = True
        logger.info(f"✓ Expert initialized: {self.config.name}")

    async def _load_adapter(self, adapter_path: Path) -> None:
        """Load LoRA adapter for fine-tuned model.

        Args:
            adapter_path: Path to LoRA adapter files.
        """
        # This will be implemented when we have trained models
        # For now, we'll use base models with specialized prompts
        logger.info(f"LoRA adapter path configured: {adapter_path}")
        # TODO: Integrate with Ollama adapter loading once models trained

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this expert.

        Returns:
            System prompt tailored to expert's specialization.
        """
        pass

    async def can_handle(self, user_intent: str) -> tuple[bool, float]:
        """Check if this expert can handle the given task.

        Args:
            user_intent: User's task description.

        Returns:
            (can_handle, confidence) tuple.
        """
        # Check for keywords
        intent_lower = user_intent.lower()
        keyword_matches = sum(
            1 for keyword in self.config.keywords
            if keyword.lower() in intent_lower
        )

        if keyword_matches == 0:
            return (False, 0.0)

        # Calculate confidence based on keyword density
        confidence = min(
            keyword_matches / len(self.config.keywords),
            1.0
        )

        # Apply confidence threshold
        can_handle = confidence >= self.config.confidence_threshold

        logger.debug(
            f"{self.config.name}: {keyword_matches} keywords matched, "
            f"confidence={confidence:.2f}, can_handle={can_handle}"
        )

        return (can_handle, confidence)

    async def generate(
        self,
        user_intent: str,
        context: Optional[dict[str, Any]] = None,
    ) -> ExpertResponse:
        """Generate response for the given task.

        Args:
            user_intent: User's task description.
            context: Optional context (previous messages, files, etc.).

        Returns:
            Expert's response.
        """
        if not self.initialized:
            await self.initialize()

        # Build prompt with context
        prompt = self._build_prompt(user_intent, context)

        # Get system prompt
        system_prompt = self.get_system_prompt()

        # Generate using orchestrator
        result = await self.orchestrator.generate(
            prompt=prompt,
            tier=self.config.tier,
            system=system_prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        # Extract confidence (if provided in response)
        confidence = await self._extract_confidence(result.content)

        return ExpertResponse(
            expert_name=self.config.name,
            content=result.content,
            confidence=confidence,
            reasoning=self._extract_reasoning(result.content),
            metadata={
                "model": result.model,
                "provider": result.provider.value,
                "tokens_used": result.tokens_used,
                "latency_ms": result.latency_ms,
            },
        )

    def _build_prompt(
        self,
        user_intent: str,
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        """Build prompt with user intent and context.

        Args:
            user_intent: User's task description.
            context: Optional context.

        Returns:
            Complete prompt.
        """
        prompt_parts = []

        # Add context if provided
        if context:
            if "previous_messages" in context:
                prompt_parts.append("Previous conversation:")
                for msg in context["previous_messages"][-3:]:  # Last 3 messages
                    prompt_parts.append(f"- {msg}")
                prompt_parts.append("")

            if "files" in context:
                prompt_parts.append("Relevant files:")
                for file_path, content in context["files"].items():
                    prompt_parts.append(f"File: {file_path}")
                    prompt_parts.append(f"```\n{content}\n```")
                prompt_parts.append("")

        # Add user intent
        prompt_parts.append("Task:")
        prompt_parts.append(user_intent)

        return "\n".join(prompt_parts)

    async def _extract_confidence(self, content: str) -> float:
        """Extract confidence score from response.

        Args:
            content: Expert's response content.

        Returns:
            Confidence score (0.0-1.0).
        """
        # For now, return default confidence
        # TODO: Parse confidence from response if model outputs it
        return 0.8

    def _extract_reasoning(self, content: str) -> Optional[str]:
        """Extract reasoning/explanation from response.

        Args:
            content: Expert's response content.

        Returns:
            Reasoning text if found.
        """
        # Check if response has explicit reasoning section
        if "Reasoning:" in content or "Explanation:" in content:
            lines = content.split("\n")
            reasoning_lines = []
            in_reasoning = False

            for line in lines:
                if "Reasoning:" in line or "Explanation:" in line:
                    in_reasoning = True
                    continue
                if in_reasoning:
                    if line.startswith("#") or line.startswith("---"):
                        break
                    reasoning_lines.append(line)

            if reasoning_lines:
                return "\n".join(reasoning_lines).strip()

        return None

    def __str__(self) -> str:
        """String representation."""
        return f"Expert({self.config.name}: {self.config.specialization})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"Expert(name={self.config.name}, "
            f"specialization={self.config.specialization}, "
            f"keywords={len(self.config.keywords)})"
        )
