"""Synthesizer for combining outputs from multiple experts."""

from __future__ import annotations

import logging
from typing import Optional

from hafs.agents.moe.expert import ExpertResponse
from hafs.core.orchestrator_v2 import TaskTier, UnifiedOrchestrator

logger = logging.getLogger(__name__)


class Synthesizer:
    """Combines outputs from multiple experts into a cohesive solution.

    When multiple experts contribute to a task, the synthesizer:
    - Integrates their outputs
    - Resolves conflicts
    - Creates a unified solution
    - Ensures consistency
    """

    def __init__(
        self,
        orchestrator: Optional[UnifiedOrchestrator] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        """Initialize synthesizer.

        Args:
            orchestrator: Orchestrator for API calls.
            max_tokens: Maximum tokens for synthesis.
            temperature: Temperature for synthesis.
        """
        self.orchestrator = orchestrator or UnifiedOrchestrator()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.initialized = False

    async def initialize(self) -> None:
        """Initialize synthesizer."""
        if self.initialized:
            return

        if not hasattr(self.orchestrator, '_initialized') or not self.orchestrator._initialized:
            await self.orchestrator.initialize()

        self.initialized = True
        logger.info("✓ Synthesizer initialized")

    async def synthesize(
        self,
        user_intent: str,
        expert_responses: list[ExpertResponse],
    ) -> str:
        """Synthesize multiple expert outputs into unified solution.

        Args:
            user_intent: Original user task.
            expert_responses: Responses from expert(s).

        Returns:
            Synthesized solution integrating all expert insights.
        """
        if not self.initialized:
            await self.initialize()

        # If only one expert, return its response directly
        if len(expert_responses) == 1:
            logger.info(f"Single expert response from {expert_responses[0].expert_name}")
            return expert_responses[0].content

        # Multiple experts - need to synthesize
        logger.info(
            f"Synthesizing {len(expert_responses)} expert responses: "
            f"{[r.expert_name for r in expert_responses]}"
        )

        # Build synthesis prompt
        expert_outputs_text = []
        for response in expert_responses:
            expert_outputs_text.append(
                f"### {response.expert_name.upper()} Expert "
                f"(confidence: {response.confidence:.2f})\n"
                f"{response.content}"
            )

        prompt = f"""
You are synthesizing insights from multiple ROM hacking experts into a cohesive solution.

User's Task:
{user_intent}

Expert Outputs:
{chr(10).join(expert_outputs_text)}

Create a unified solution that:
1. Integrates insights from all experts
2. Presents a clear step-by-step approach
3. Resolves any conflicts between experts
4. Provides a complete answer to the user's task
5. Credits experts where appropriate (e.g., "ASM Expert suggests...")

Structure your response:
1. Overview of the solution
2. Detailed implementation steps
3. Code/examples if applicable
4. Testing/validation recommendations
5. Additional notes or warnings

Focus on creating a practical, actionable response that the user can follow.
"""

        result = await self.orchestrator.generate(
            prompt=prompt,
            tier=TaskTier.REASONING,  # Use reasoning tier for synthesis
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        logger.info(
            f"✓ Synthesized solution ({result.tokens_used} tokens, "
            f"{result.latency_ms}ms)"
        )

        return result.content

    async def resolve_conflict(
        self,
        expert_responses: list[ExpertResponse],
        conflict_description: str,
    ) -> str:
        """Resolve a specific conflict between expert responses.

        Args:
            expert_responses: Conflicting expert responses.
            conflict_description: Description of the conflict.

        Returns:
            Resolution of the conflict.
        """
        if not self.initialized:
            await self.initialize()

        expert_positions = []
        for response in expert_responses:
            expert_positions.append(
                f"**{response.expert_name.upper()} Expert:**\n{response.content}"
            )

        prompt = f"""
Resolve this conflict between ROM hacking experts:

Conflict: {conflict_description}

Expert Positions:
{chr(10).join(expert_positions)}

Provide:
1. Analysis of each expert's position
2. Your resolution/recommendation
3. Reasoning for your decision
4. Trade-offs if applicable
"""

        result = await self.orchestrator.generate(
            prompt=prompt,
            tier=TaskTier.REASONING,
            max_tokens=1024,
            temperature=0.6,
        )

        return result.content

    async def explain_synthesis(
        self,
        user_intent: str,
        expert_responses: list[ExpertResponse],
        synthesized_result: str,
    ) -> str:
        """Explain how the synthesized solution was created.

        Args:
            user_intent: Original user task.
            expert_responses: Expert responses that were synthesized.
            synthesized_result: The synthesized result.

        Returns:
            Explanation of the synthesis process.
        """
        expert_names = [r.expert_name for r in expert_responses]

        prompt = f"""
Explain how this synthesized solution was created from multiple expert inputs.

Task: {user_intent}
Experts involved: {', '.join(expert_names)}

Synthesized Solution:
{synthesized_result}

Provide a brief explanation (2-3 paragraphs) of:
1. Why multiple experts were needed
2. What each expert contributed
3. How their insights were combined
"""

        result = await self.orchestrator.generate(
            prompt=prompt,
            tier=TaskTier.FAST,
            max_tokens=512,
            temperature=0.7,
        )

        return result.content
