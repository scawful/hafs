"""Task Classifier for Mixture of Experts routing."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

from hafs.core.orchestrator_v2 import TaskTier, UnifiedOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class Classification:
    """Result of task classification."""

    expert_names: list[str]
    confidences: list[float]
    reasoning: str
    is_multi_expert: bool

    @property
    def primary_expert(self) -> str:
        """Get the primary (highest confidence) expert."""
        if not self.expert_names:
            return "unknown"
        return self.expert_names[0]

    @property
    def primary_confidence(self) -> float:
        """Get confidence for primary expert."""
        if not self.confidences:
            return 0.0
        return self.confidences[0]


class TaskClassifier:
    """Classifies user intents and routes to appropriate expert(s).

    Uses a lightweight model to quickly determine which expert(s)
    should handle a given ROM hacking task.

    Supports:
    - Single expert routing (most tasks)
    - Multi-expert routing (complex tasks requiring multiple domains)
    - Confidence thresholds for routing decisions
    """

    # Expert categories
    EXPERTS = {
        "asm": {
            "keywords": [
                "asm", "assembly", "routine", "bank", "memory", "register",
                "optimization", "65816", "instruction", "opcode", "subroutine",
                "stack", "pointer", "address", "hex", "disassembly"
            ],
            "description": "65816 assembly code generation and modification",
        },
        "yaze": {
            "keywords": [
                "yaze", "rom", "graphics", "sprite", "tile", "map", "palette",
                "tool", "patch", "editor", "compress", "decompress", "overworld",
                "dungeon", "c++", "function", "api"
            ],
            "description": "YAZE ROM editor tools and ROM manipulation",
        },
        "debug": {
            "keywords": [
                "error", "bug", "crash", "fix", "debug", "problem", "issue",
                "fail", "broken", "wrong", "help", "diagnose", "trace",
                "exception", "segfault", "assert"
            ],
            "description": "Error diagnostics, debugging, and failure analysis",
        },
    }

    def __init__(
        self,
        orchestrator: Optional[UnifiedOrchestrator] = None,
        multi_expert_threshold: float = 0.6,
        max_tokens: int = 200,
        temperature: float = 0.3,
    ):
        """Initialize task classifier.

        Args:
            orchestrator: Orchestrator for API calls.
            multi_expert_threshold: Confidence threshold for including additional experts.
            max_tokens: Maximum tokens for classification.
            temperature: Temperature for classification (lower = more consistent).
        """
        self.orchestrator = orchestrator or UnifiedOrchestrator()
        self.multi_expert_threshold = multi_expert_threshold
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.initialized = False

    async def initialize(self) -> None:
        """Initialize classifier."""
        if self.initialized:
            return

        if not hasattr(self.orchestrator, '_initialized') or not self.orchestrator._initialized:
            await self.orchestrator.initialize()

        self.initialized = True
        logger.info("✓ Task classifier initialized")

    async def classify(self, user_intent: str) -> Classification:
        """Classify user intent to determine which expert(s) should handle it.

        Args:
            user_intent: User's task description.

        Returns:
            Classification with expert assignments and confidences.
        """
        if not self.initialized:
            await self.initialize()

        # First, try keyword-based classification (fast path)
        keyword_classification = self._classify_by_keywords(user_intent)

        # If keyword classification is confident (>0.8), use it
        if keyword_classification.primary_confidence > 0.8:
            logger.info(
                f"Fast classification: {keyword_classification.primary_expert} "
                f"(confidence={keyword_classification.primary_confidence:.2f})"
            )
            return keyword_classification

        # Otherwise, use LLM for more nuanced classification
        llm_classification = await self._classify_by_llm(user_intent)

        logger.info(
            f"LLM classification: experts={llm_classification.expert_names}, "
            f"confidences={[f'{c:.2f}' for c in llm_classification.confidences]}"
        )

        return llm_classification

    def _classify_by_keywords(self, user_intent: str) -> Classification:
        """Fast keyword-based classification.

        Args:
            user_intent: User's task description.

        Returns:
            Classification based on keyword matching.
        """
        intent_lower = user_intent.lower()
        expert_scores: dict[str, float] = {}

        # Score each expert based on keyword matches
        for expert_name, expert_info in self.EXPERTS.items():
            keywords = expert_info["keywords"]
            matches = sum(
                1 for keyword in keywords
                if keyword in intent_lower
            )

            # Normalize by number of keywords
            expert_scores[expert_name] = matches / len(keywords) if keywords else 0.0

        # Sort by score
        sorted_experts = sorted(
            expert_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Build classification
        expert_names = []
        confidences = []
        for expert_name, confidence in sorted_experts:
            if confidence > 0.0:
                expert_names.append(expert_name)
                confidences.append(confidence)

        # Determine if multi-expert
        is_multi_expert = (
            len(expert_names) > 1 and
            len([c for c in confidences if c >= self.multi_expert_threshold]) > 1
        )

        return Classification(
            expert_names=expert_names if expert_names else ["asm"],  # Default to ASM
            confidences=confidences if confidences else [0.5],
            reasoning="Keyword-based classification",
            is_multi_expert=is_multi_expert,
        )

    async def _classify_by_llm(self, user_intent: str) -> Classification:
        """LLM-based classification for nuanced tasks.

        Args:
            user_intent: User's task description.

        Returns:
            Classification from LLM analysis.
        """
        # Build expert descriptions
        expert_descriptions = []
        for expert_name, expert_info in self.EXPERTS.items():
            expert_descriptions.append(
                f"- {expert_name}: {expert_info['description']}"
            )

        prompt = f"""
Classify this ROM hacking task to determine which expert(s) should handle it.

Available experts:
{chr(10).join(expert_descriptions)}

Task: {user_intent}

Analyze the task and determine:
1. Which expert(s) are needed
2. Confidence level for each expert (0.0-1.0)
3. Brief reasoning

Output ONLY valid JSON in this format:
{{
  "experts": ["asm", "yaze"],
  "confidences": [0.9, 0.7],
  "reasoning": "Task requires both assembly code and YAZE tools"
}}

Important:
- Use confidence 0.8+ for primary expert
- Include secondary experts only if confidence >= 0.6
- If only one expert needed, output single-element arrays
- Valid expert names: asm, yaze, debug

JSON output:
"""

        # Use FAST tier for quick classification
        result = await self.orchestrator.generate(
            prompt=prompt,
            tier=TaskTier.FAST,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            content = result.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            data = json.loads(content)

            expert_names = data.get("experts", ["asm"])
            confidences = data.get("confidences", [0.5])
            reasoning = data.get("reasoning", "LLM classification")

            # Validate
            valid_experts = [e for e in expert_names if e in self.EXPERTS]
            if not valid_experts:
                valid_experts = ["asm"]  # Fallback
                confidences = [0.5]

            # Ensure confidences match experts
            if len(confidences) < len(valid_experts):
                confidences.extend([0.5] * (len(valid_experts) - len(confidences)))
            elif len(confidences) > len(valid_experts):
                confidences = confidences[:len(valid_experts)]

            # Determine if multi-expert
            is_multi_expert = (
                len(valid_experts) > 1 and
                len([c for c in confidences if c >= self.multi_expert_threshold]) > 1
            )

            return Classification(
                expert_names=valid_experts,
                confidences=confidences,
                reasoning=reasoning,
                is_multi_expert=is_multi_expert,
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM classification: {e}")
            logger.warning(f"Response: {result.content[:200]}")

            # Fallback to keyword classification
            return self._classify_by_keywords(user_intent)

    def get_expert_description(self, expert_name: str) -> str:
        """Get description for an expert.

        Args:
            expert_name: Name of expert.

        Returns:
            Description of expert's capabilities.
        """
        if expert_name in self.EXPERTS:
            return self.EXPERTS[expert_name]["description"]
        return f"Unknown expert: {expert_name}"
