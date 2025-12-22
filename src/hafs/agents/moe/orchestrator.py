"""Mixture of Experts Orchestrator for ROM hacking tasks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from agents.moe.classifier import Classification, TaskClassifier
from agents.moe.expert import ExpertResponse
from agents.moe.experts.asm_expert import AsmExpert
from agents.moe.experts.debug_expert import DebugExpert
from agents.moe.experts.registry_expert import RegistryExpert
from agents.moe.experts.yaze_expert import YazeExpert
from agents.moe.registry import ModelRegistry, RoutingTable
from agents.moe.synthesizer import Synthesizer
from core.orchestrator_v2 import UnifiedOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class MoEResult:
    """Result from MoE orchestrator."""

    content: str
    classification: Classification
    expert_responses: list[ExpertResponse]
    synthesis_used: bool
    metadata: dict[str, Any]

    @property
    def experts_used(self) -> list[str]:
        """Get list of expert names used."""
        return [r.expert_name for r in self.expert_responses]


class MoEOrchestrator:
    """Mixture of Experts orchestrator for ROM hacking.

    Coordinates multiple specialized experts to handle complex tasks:
    - ASM Expert: 65816 assembly code
    - YAZE Expert: ROM editor tools and C++ API
    - Debug Expert: Error diagnosis and debugging

    Workflow:
    1. Classify task → determine which expert(s) to use
    2. Route to expert(s) → get specialized responses
    3. Synthesize (if multiple experts) → combine into unified solution
    4. Return result

    Example:
        orchestrator = MoEOrchestrator()
        await orchestrator.initialize()

        result = await orchestrator.execute(
            "Add a new item to ALTTP that uses custom YAZE graphics"
        )
        # Uses both ASM Expert (item code) and YAZE Expert (graphics)

        print(result.content)
    """

    def __init__(
        self,
        api_orchestrator: Optional[UnifiedOrchestrator] = None,
        model_registry: Optional[ModelRegistry] = None,
        routing_table: Optional[RoutingTable] = None,
    ):
        """Initialize MoE orchestrator.

        Args:
            api_orchestrator: Optional orchestrator for API routing.
        """
        self.api_orchestrator = api_orchestrator or UnifiedOrchestrator()
        self.model_registry = model_registry or ModelRegistry.load()
        self.routing_table = routing_table or RoutingTable.load()

        # Initialize components
        self.classifier = TaskClassifier(
            self.api_orchestrator,
            routing_table=self.routing_table,
            model_registry=self.model_registry,
        )
        self.synthesizer = Synthesizer(self.api_orchestrator)

        # Initialize experts
        self.experts = {
            "asm": AsmExpert(self.api_orchestrator),
            "yaze": YazeExpert(self.api_orchestrator),
            "debug": DebugExpert(self.api_orchestrator),
        }
        self._register_registry_experts()

        self.initialized = False
        logger.info("MoE Orchestrator created with %d experts", len(self.experts))

    def _register_registry_experts(self) -> None:
        """Register experts defined in the model registry."""
        if not self.model_registry:
            return

        for model in self.model_registry.models.values():
            if not model.enabled or model.name in self.experts:
                continue
            self.experts[model.name] = RegistryExpert(
                model,
                orchestrator=self.api_orchestrator,
                routing_table=self.routing_table,
            )

    async def initialize(self) -> None:
        """Initialize all components."""
        if self.initialized:
            return

        logger.info("Initializing MoE Orchestrator...")

        # Initialize API orchestrator
        if not hasattr(self.api_orchestrator, '_initialized') or not self.api_orchestrator._initialized:
            await self.api_orchestrator.initialize()

        # Initialize classifier and synthesizer
        await self.classifier.initialize()
        await self.synthesizer.initialize()

        # Initialize all experts
        for expert_name, expert in self.experts.items():
            await expert.initialize()
            logger.info(f"  ✓ {expert_name} expert ready")

        self.initialized = True
        logger.info("✓ MoE Orchestrator initialized")

    async def execute(
        self,
        user_intent: str,
        context: Optional[dict[str, Any]] = None,
        force_experts: Optional[list[str]] = None,
    ) -> MoEResult:
        """Execute a ROM hacking task using appropriate expert(s).

        Args:
            user_intent: User's task description.
            context: Optional context (files, previous messages, etc.).
            force_experts: Optional list of expert names to force use of.

        Returns:
            MoE result with synthesized solution.
        """
        if not self.initialized:
            await self.initialize()

        logger.info(f"Executing task: {user_intent[:100]}...")

        # Step 1: Classify task (unless forcing specific experts)
        if force_experts:
            classification = Classification(
                expert_names=force_experts,
                confidences=[0.9] * len(force_experts),
                reasoning=f"Forced experts: {force_experts}",
                is_multi_expert=len(force_experts) > 1,
            )
            logger.info(f"Using forced experts: {force_experts}")
        else:
            classification = await self.classifier.classify(user_intent)
            logger.info(
                f"Classification: experts={classification.expert_names}, "
                f"confidences={[f'{c:.2f}' for c in classification.confidences]}"
            )

        # Step 2: Route to expert(s)
        expert_responses = await self._route_to_experts(
            classification,
            user_intent,
            context,
        )

        # Step 3: Synthesize if multiple experts
        if len(expert_responses) > 1:
            logger.info(f"Synthesizing {len(expert_responses)} expert responses...")
            content = await self.synthesizer.synthesize(user_intent, expert_responses)
            synthesis_used = True
        else:
            content = expert_responses[0].content
            synthesis_used = False

        # Build result
        result = MoEResult(
            content=content,
            classification=classification,
            expert_responses=expert_responses,
            synthesis_used=synthesis_used,
            metadata={
                "total_tokens": sum(
                    r.metadata.get("tokens_used", 0)
                    for r in expert_responses
                    if r.metadata
                ),
                "total_latency_ms": sum(
                    r.metadata.get("latency_ms", 0)
                    for r in expert_responses
                    if r.metadata
                ),
            },
        )

        logger.info(
            f"✓ Task completed using {len(expert_responses)} expert(s): "
            f"{result.experts_used}"
        )

        return result

    async def _route_to_experts(
        self,
        classification: Classification,
        user_intent: str,
        context: Optional[dict[str, Any]] = None,
    ) -> list[ExpertResponse]:
        """Route task to appropriate expert(s).

        Args:
            classification: Task classification.
            user_intent: User's task description.
            context: Optional context.

        Returns:
            List of expert responses.
        """
        responses = []

        # Execute experts in parallel (if multiple)
        import asyncio

        tasks = []
        for expert_name in classification.expert_names:
            if expert_name in self.experts:
                expert = self.experts[expert_name]
                tasks.append(expert.generate(user_intent, context))
            else:
                logger.warning(f"Unknown expert: {expert_name}")

        # Gather responses
        if tasks:
            responses = await asyncio.gather(*tasks)

        return responses

    async def explain_routing(self, user_intent: str) -> str:
        """Explain which experts would be used for a task (without executing).

        Args:
            user_intent: User's task description.

        Returns:
            Explanation of expert routing.
        """
        if not self.initialized:
            await self.initialize()

        classification = await self.classifier.classify(user_intent)

        explanation_parts = [
            f"Task: {user_intent}",
            "",
            "Expert Routing:",
        ]

        for expert_name, confidence in zip(
            classification.expert_names,
            classification.confidences
        ):
            if expert_name in self.experts:
                expert = self.experts[expert_name]
                explanation_parts.append(
                    f"- {expert.config.display_name} "
                    f"(confidence: {confidence:.2%})"
                )
                explanation_parts.append(f"  Specialization: {expert.config.specialization}")

        explanation_parts.append("")
        explanation_parts.append(f"Classification reasoning: {classification.reasoning}")

        if classification.is_multi_expert:
            explanation_parts.append("")
            explanation_parts.append(
                "⚠ Multi-expert task: Outputs will be synthesized into unified solution"
            )

        return "\n".join(explanation_parts)

    async def list_experts(self) -> dict[str, str]:
        """List all available experts and their specializations.

        Returns:
            Dict mapping expert names to specializations.
        """
        return {
            name: expert.config.specialization
            for name, expert in self.experts.items()
        }

    async def get_expert_info(self, expert_name: str) -> Optional[dict[str, Any]]:
        """Get detailed information about an expert.

        Args:
            expert_name: Name of expert.

        Returns:
            Expert info dict or None if not found.
        """
        if expert_name not in self.experts:
            return None

        expert = self.experts[expert_name]
        config = expert.config

        return {
            "name": config.name,
            "display_name": config.display_name,
            "specialization": config.specialization,
            "keywords": config.keywords,
            "model_name": config.model_name,
            "lora_adapter_path": str(config.lora_adapter_path) if config.lora_adapter_path else None,
            "confidence_threshold": config.confidence_threshold,
            "tier": config.tier.value,
        }
