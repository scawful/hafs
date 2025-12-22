"""Cross-Domain Sample Generator for Training Data Diversity.

Generates hybrid training samples that combine multiple domains:
- ASM + Oracle: Compare vanilla vs ROM hack implementations
- YAZE + Narrative: Pair C++ code with game design context
- ASM + Gigaleak: Pair production code with Nintendo commentary

This increases diversity by teaching relationships between domains.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from agents.training.base import SourceItem, TrainingSample
from agents.training.json_utils import extract_json_from_response
from config.prompts import get_prompt

logger = logging.getLogger(__name__)


@dataclass
class CrossDomainPair:
    """A pair of source items from different domains."""

    primary: SourceItem
    secondary: SourceItem
    combination_type: str  # "asm+oracle", "yaze+narrative", "asm+gigaleak"
    relationship: str  # "vanilla_vs_hack", "code_with_context", "production_with_commentary"


class CrossDomainGenerator:
    """Generates training samples that combine multiple domains.

    Creates samples that teach relationships between different knowledge areas:
    - How ROM hacks modify vanilla code
    - How game code implements design concepts
    - How production code relates to design documents
    """

    def __init__(self, orchestrator=None):
        """Initialize cross-domain generator.

        Args:
            orchestrator: Model orchestrator for generation
        """
        self._orchestrator = orchestrator

    async def setup(self):
        """Initialize orchestrator if not provided."""
        if not self._orchestrator:
            from core.orchestrator_v2 import UnifiedOrchestrator

            self._orchestrator = UnifiedOrchestrator()

    async def generate_asm_oracle_pair(
        self,
        asm_item: SourceItem,
        oracle_item: SourceItem,
    ) -> Optional[TrainingSample]:
        """Generate sample comparing vanilla ASM vs Oracle ROM hack.

        Args:
            asm_item: Vanilla ASM routine
            oracle_item: Oracle ROM hack routine (ideally hooks this vanilla routine)

        Returns:
            TrainingSample teaching ROM hacking technique, or None if generation failed
        """
        if not self._orchestrator:
            await self.setup()

        prompt = self._build_asm_oracle_prompt(asm_item, oracle_item)

        try:
            from core.orchestrator_v2 import Provider, TaskTier

            response_obj = await asyncio.wait_for(
                self._orchestrator.generate(
                    prompt=prompt,
                    tier=TaskTier.CODING,
                    provider=Provider.GEMINI,
                ),
                timeout=120.0,
            )

            response = response_obj.content

            # Extract JSON from response
            data = extract_json_from_response(response)
            if not data:
                logger.warning("Failed to extract JSON from ASM+Oracle response")
                return None

            # Create cross-domain sample
            return TrainingSample(
                instruction=str(data.get("instruction", "")).strip(),
                input=str(data.get("input", "")).strip(),
                output=str(data.get("output", "")).strip(),
                domain="asm+oracle",  # Cross-domain tag
                source=f"{asm_item.source}_x_{oracle_item.source}",
                teacher_model="gemini-3-flash-preview",
                teacher_prompt=str(prompt),
                kg_entities=[asm_item.name, oracle_item.name],
            )

        except asyncio.TimeoutError:
            logger.warning("Timeout generating ASM+Oracle pair")
            return None
        except Exception as e:
            logger.error(f"Failed to generate ASM+Oracle pair: {e}")
            return None

    def _build_asm_oracle_prompt(
        self,
        asm_item: SourceItem,
        oracle_item: SourceItem,
    ) -> str:
        """Build prompt for ASM+Oracle cross-domain generation."""
        template = get_prompt(
            "agents.training.cross_domain.asm_oracle_prompt",
            """You are teaching ROM hacking by comparing vanilla ALTTP code with Oracle-of-Secrets modifications.

VANILLA ROUTINE:
Name: {vanilla_name}
{vanilla_content}

ORACLE ROM HACK:
Name: {oracle_name}
{oracle_content}

Generate a training sample that teaches ROM hacking by comparing these two implementations.

JSON FORMAT:
{{
  "instruction": "Question asking how Oracle modifies vanilla behavior, or how to implement similar ROM hack",
  "input": "Technical context: vanilla address, Oracle technique, what changes (include ROM addresses as $BB:AAAA)",
  "output": "Comprehensive explanation (250-400 words) covering:
    1. What the vanilla routine does
    2. How Oracle modifies it (code comparison)
    3. The ROM hacking technique used (hooks, bank allocation, etc.)
    4. Why this approach was chosen
    5. How to apply this technique to other vanilla routines"
}}

CRITICAL:
- Focus on teaching the ROM HACKING TECHNIQUE, not just describing what changed.
- Include exact ROM/RAM addresses in $BB:AAAA (ROM) and $7E:XXXX (WRAM) format.
- Treat addresses as first-class details (call out hook sites and target banks).""",
        )

        return template.format(
            vanilla_name=asm_item.name,
            vanilla_content=asm_item.content[:1000],
            oracle_name=oracle_item.name,
            oracle_content=oracle_item.content[:1000],
        )

    async def generate_yaze_narrative_pair(
        self,
        yaze_item: SourceItem,
        narrative: str,
    ) -> Optional[TrainingSample]:
        """Generate sample pairing YAZE C++ code with game design narrative.

        Args:
            yaze_item: YAZE C++ code unit
            narrative: Game design narrative or feature description

        Returns:
            TrainingSample explaining how code implements design, or None if failed
        """
        if not self._orchestrator:
            await self.setup()

        prompt = self._build_yaze_narrative_prompt(yaze_item, narrative)

        try:
            from core.orchestrator_v2 import Provider, TaskTier

            response_obj = await asyncio.wait_for(
                self._orchestrator.generate(
                    prompt=prompt,
                    tier=TaskTier.CODING,
                    provider=Provider.GEMINI,
                ),
                timeout=120.0,
            )

            response = response_obj.content

            # Extract JSON from response
            data = extract_json_from_response(response)
            if not data:
                logger.warning("Failed to extract JSON from YAZE+narrative response")
                return None

            return TrainingSample(
                instruction=str(data.get("instruction", "")).strip(),
                input=str(data.get("input", "")).strip(),
                output=str(data.get("output", "")).strip(),
                domain="yaze+narrative",
                source=f"{yaze_item.source}_x_narrative",
                teacher_model="gemini-3-flash-preview",
                teacher_prompt=str(prompt),
                kg_entities=[yaze_item.name],
            )

        except asyncio.TimeoutError:
            logger.warning("Timeout generating YAZE+narrative pair")
            return None
        except Exception as e:
            logger.error(f"Failed to generate YAZE+narrative pair: {e}")
            return None

    def _build_yaze_narrative_prompt(
        self,
        yaze_item: SourceItem,
        narrative: str,
    ) -> str:
        """Build prompt for YAZE+narrative cross-domain generation."""
        template = get_prompt(
            "agents.training.cross_domain.yaze_narrative_prompt",
            """You are teaching game development by pairing C++ emulator/editor code with game design concepts.

GAME DESIGN NARRATIVE:
{narrative}

C++ IMPLEMENTATION (YAZE):
{code_content}

Generate a training sample that teaches how this code implements the game design concept.

JSON FORMAT:
{{
  "instruction": "Question about how the game feature works or how to implement similar functionality",
  "input": "Design requirements or player-visible behavior",
  "output": "Explanation (200-350 words) covering:
    1. What the player experiences (game design perspective)
    2. How the code implements this (architecture overview)
    3. Key algorithms or data structures used
    4. Trade-offs and design decisions
    5. How to extend or modify this implementation"
}}

CRITICAL: Bridge the gap between DESIGN (what players see) and CODE (how it's built).""",
        )

        return template.format(
            narrative=narrative[:500],
            code_content=yaze_item.content[:1000],
        )

    async def generate_asm_gigaleak_pair(
        self,
        asm_item: SourceItem,
        gigaleak_item: SourceItem,
    ) -> Optional[TrainingSample]:
        """Generate sample pairing ASM code with Nintendo Gigaleak commentary.

        Args:
            asm_item: Disassembled ASM routine
            gigaleak_item: Nintendo source with comments/variable names

        Returns:
            TrainingSample explaining production code insights, or None if failed
        """
        if not self._orchestrator:
            await self.setup()

        prompt = self._build_asm_gigaleak_prompt(asm_item, gigaleak_item)

        try:
            from core.orchestrator_v2 import Provider, TaskTier

            response_obj = await asyncio.wait_for(
                self._orchestrator.generate(
                    prompt=prompt,
                    tier=TaskTier.CODING,
                    provider=Provider.GEMINI,
                ),
                timeout=120.0,
            )

            response = response_obj.content

            # Extract JSON from response
            data = extract_json_from_response(response)
            if not data:
                logger.warning("Failed to extract JSON from ASM+gigaleak response")
                return None

            return TrainingSample(
                instruction=str(data.get("instruction", "")).strip(),
                input=str(data.get("input", "")).strip(),
                output=str(data.get("output", "")).strip(),
                domain="asm+gigaleak",
                source=f"{asm_item.source}_x_gigaleak",
                teacher_model="gemini-3-flash-preview",
                teacher_prompt=str(prompt),
                kg_entities=[asm_item.name, gigaleak_item.name],
            )

        except asyncio.TimeoutError:
            logger.warning("Timeout generating ASM+gigaleak pair")
            return None
        except Exception as e:
            logger.error(f"Failed to generate ASM+gigaleak pair: {e}")
            return None

    def _build_asm_gigaleak_prompt(
        self,
        asm_item: SourceItem,
        gigaleak_item: SourceItem,
    ) -> str:
        """Build prompt for ASM+gigaleak cross-domain generation."""
        template = get_prompt(
            "agents.training.cross_domain.asm_gigaleak_prompt",
            """You are teaching SNES development by comparing disassembled code with Nintendo's original source.

DISASSEMBLED CODE:
{disasm_content}

NINTENDO ORIGINAL SOURCE (Gigaleak):
{gigaleak_content}

Generate a training sample that teaches production-quality SNES development practices.

JSON FORMAT:
{{
  "instruction": "Question about Nintendo's implementation approach or best practices",
  "input": "Context about what this code does in the game (include ROM addresses where possible)",
  "output": "Explanation (200-350 words) covering:
    1. What the code accomplishes
    2. Nintendo's variable naming and code organization
    3. Production-quality patterns and techniques
    4. Why Nintendo chose this approach (performance, memory, maintainability)
    5. Lessons for modern ROM hacking or SNES development"
}}

CRITICAL: Extract PROFESSIONAL INSIGHTS from Nintendo's production code. Include precise addresses when present.""",
        )

        return template.format(
            disasm_content=asm_item.content[:1000],
            gigaleak_content=gigaleak_item.content[:1000],
        )

    async def find_related_pairs(
        self,
        primary_items: list[SourceItem],
        secondary_items: list[SourceItem],
        combination_type: str,
        max_pairs: int = 50,
    ) -> list[CrossDomainPair]:
        """Find related pairs of items from different domains.

        Args:
            primary_items: Items from primary domain
            secondary_items: Items from secondary domain
            combination_type: Type of combination (asm+oracle, etc.)
            max_pairs: Maximum number of pairs to find

        Returns:
            List of related CrossDomainPairs
        """
        pairs: list[CrossDomainPair] = []

        # Simple name-based matching for now
        # TODO: Use embeddings for semantic matching
        for primary in primary_items[:max_pairs * 2]:
            for secondary in secondary_items:
                # Check if names are related
                if self._are_related(primary, secondary, combination_type):
                    relationship = self._determine_relationship(combination_type)
                    pairs.append(
                        CrossDomainPair(
                            primary=primary,
                            secondary=secondary,
                            combination_type=combination_type,
                            relationship=relationship,
                        )
                    )

                    if len(pairs) >= max_pairs:
                        return pairs

        logger.info(f"Found {len(pairs)} {combination_type} pairs")
        return pairs

    def _are_related(
        self,
        item1: SourceItem,
        item2: SourceItem,
        combination_type: str,
    ) -> bool:
        """Check if two items are related based on combination type."""
        name1 = item1.name.lower()
        name2 = item2.name.lower()

        # Simple heuristic: names are similar or one contains the other
        if name1 == name2:
            return True

        if name1 in name2 or name2 in name1:
            return True

        # For ASM+Oracle, check if Oracle hooks this vanilla routine
        if combination_type == "asm+oracle":
            # Check if oracle_item mentions vanilla routine name
            if hasattr(item2, "hooks_vanilla"):
                hooks = getattr(item2, "hooks_vanilla", "")
                if hooks and name1 in hooks.lower():
                    return True

        return False

    def _determine_relationship(self, combination_type: str) -> str:
        """Determine relationship type from combination type."""
        relationship_map = {
            "asm+oracle": "vanilla_vs_hack",
            "yaze+narrative": "code_with_context",
            "asm+gigaleak": "production_with_commentary",
        }
        return relationship_map.get(combination_type, "unknown")
