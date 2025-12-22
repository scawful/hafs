"""Gigaleak Data Generator for Nintendo original source training data.

Generates instruction-tuning data from Gigaleak symbols.json (320K+ lines),
focusing on original Nintendo ALTTP source code with Japanese-to-English context.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agents.training.base import DataGenerator, SourceItem, TrainingSample
from agents.training.json_utils import extract_json_from_response
from config.prompts import get_prompt

logger = logging.getLogger(__name__)


@dataclass
class GigaleakSourceItem(SourceItem):
    """Source item for Gigaleak symbols."""

    symbol_type: str = ""  # EQU, GLB, EXT, label
    file_path: str = ""
    line_number: int = 0
    japanese_comment: str = ""
    english_translation: str = ""
    related_usdasm_symbol: Optional[str] = None
    code_context: str = ""

    @property
    def item_id(self) -> str:
        return f"gigaleak:{self.name}:{self.symbol_type}"


class GigaleakDataGenerator(DataGenerator):
    """Generate instruction-tuning data from Gigaleak symbols.

    Extracts from original Nintendo ALTTP source code (gigaleak symbols.json),
    focusing on Japanese-to-English translation and bridging to modern disassembly.

    Teacher LLM generates natural language instructions explaining:
    - What this original Nintendo symbol represents
    - Japanese context and translation
    - Relationship to modern usdasm disassembly
    """

    GIGALEAK_KB_PATH = Path.home() / ".context" / "knowledge" / "gigaleak" / "symbols.json"

    def __init__(self):
        super().__init__(
            name="GigaleakDataGenerator",
            domain="gigaleak",
            teacher_tier="coding",
        )
        self._symbols: dict[str, dict] = {}
        self._orchestrator = None

    async def setup(self):
        """Initialize resources and load Gigaleak symbols."""
        await super().setup()

        # Load Gigaleak symbols.json
        if not self.GIGALEAK_KB_PATH.exists():
            logger.error(f"Gigaleak KB not found: {self.GIGALEAK_KB_PATH}")
            raise FileNotFoundError(f"Missing Gigaleak KB: {self.GIGALEAK_KB_PATH}")

        with open(self.GIGALEAK_KB_PATH) as f:
            self._symbols = json.load(f)

        logger.info(f"Loaded {len(self._symbols)} Gigaleak symbols")

        # Initialize orchestrator
        from core.orchestrator_v2 import UnifiedOrchestrator

        self._orchestrator = UnifiedOrchestrator()

    async def extract_source_items(self) -> list[GigaleakSourceItem]:
        """Extract symbols from Gigaleak KB.

        Filters for high-quality samples:
        - Skip symbols with empty names or trivial code context
        - Prefer symbols with Japanese comments or translations
        - Prefer symbols with usdasm cross-references
        """
        if not self._symbols:
            await self.setup()

        items: list[GigaleakSourceItem] = []

        for name, symbol in self._symbols.items():
            # Skip if name is too short or generic
            if len(name) < 3:
                continue

            # Skip if code context is just "$" or empty
            code_ctx = symbol.get("code_context", "")
            if code_ctx in ("", "$"):
                continue

            # Extract symbol data
            japanese = symbol.get("japanese_comment", "")
            english = symbol.get("english_translation", "")
            usdasm_ref = symbol.get("related_usdasm_symbol")

            # Build content for embedding/display
            content_parts = [f"Symbol: {name} ({symbol.get('symbol_type', 'UNKNOWN')})"]
            if japanese:
                content_parts.append(f"Japanese: {japanese}")
            if english:
                content_parts.append(f"English: {english}")
            if usdasm_ref:
                content_parts.append(f"Modern equivalent: {usdasm_ref}")
            if code_ctx and code_ctx != "$":
                content_parts.append(f"Code: {code_ctx}")

            content = "\n".join(content_parts)

            items.append(
                GigaleakSourceItem(
                    name=name,
                    content=content,
                    source="gigaleak",
                    symbol_type=symbol.get("symbol_type", ""),
                    file_path=symbol.get("file_path", ""),
                    line_number=symbol.get("line_number", 0),
                    japanese_comment=japanese,
                    english_translation=english,
                    related_usdasm_symbol=usdasm_ref,
                    code_context=code_ctx,
                )
            )

        logger.info(f"Extracted {len(items)} Gigaleak source items")
        return items

    def get_teacher_prompt(self, item: SourceItem) -> str:
        """Generate teacher prompt for Gigaleak symbol."""
        if not isinstance(item, GigaleakSourceItem):
            raise TypeError(f"Expected GigaleakSourceItem, got {type(item)}")

        # Build context sections
        context_parts = []

        # File context
        if item.file_path:
            # Extract just the file name from the long Japanese path
            file_name = Path(item.file_path).name
            context_parts.append(f"Original file: {file_name} (line {item.line_number})")

        # Japanese context
        if item.japanese_comment:
            context_parts.append(f"Japanese comment: {item.japanese_comment}")

        # English translation
        if item.english_translation:
            context_parts.append(f"English translation: {item.english_translation}")

        # Modern equivalent
        if item.related_usdasm_symbol:
            context_parts.append(
                f"Modern disassembly equivalent: {item.related_usdasm_symbol}"
            )

        # Code context
        if item.code_context and item.code_context != "$":
            context_parts.append(f"Code: {item.code_context}")

        context = "\n".join(context_parts) if context_parts else "No additional context"

        template = get_prompt("agents.training.generators.gigaleak_generator.prompt", "")
        if not template:
            template = (
                "You are an expert at Nintendo SNES development and ALTTP ROM hacking. "
                "Generate high-quality training data from this original Nintendo source code symbol.\n\n"
                "SYMBOL: {name}\n"
                "TYPE: {symbol_type}\n"
                "CONTEXT:\n{context}\n\n"
                "Generate a JSON object with:\n\n"
                "1. \"instruction\": A clear, specific question about this symbol. Make it natural and varied:\n"
                "   - Ask about technical purpose and implementation\n"
                "   - Ask about relationship to game mechanics or hardware\n"
                "   - Ask about Japanese-to-English translation context\n"
                "   - Ask about modern ROM hacking usage\n\n"
                "2. \"input\": Context snippet (1-2 sentences). Use format:\n"
                "   \"Source File: {{file}} (line {{line}}); Context: {{brief_description}}\"\n\n"
                "3. \"output\": A comprehensive technical explanation (150-300 words) covering:\n\n"
                "   **Technical Purpose (REQUIRED):**\n"
                "   - What this symbol represents (variable, constant, routine, label)\n"
                "   - Technical function in the game engine or hardware interface\n"
                "   - Memory location, register usage, or data structure details\n\n"
                "   **Japanese Context (if present):**\n"
                "   - Original Japanese comment: {{japanese}}\n"
                "   - English translation: {{english}}\n"
                "   - Cultural or naming conventions insight\n\n"
                "   **Modern ROM Hacking Connection:**\n"
                "   - How modern disassemblies reference this (e.g., usdasm symbol {{modern_name}})\n"
                "   - RAM address mappings (format: $7E:XXXX)\n"
                "   - Common modifications or usage in ROM hacks\n\n"
                "   **Code Analysis (if code context provided):**\n"
                "   - Line-by-line explanation of assembly operations\n"
                "   - Hardware registers accessed (PPU: $21XX, CPU: $42XX, APU: $2140-$2143)\n"
                "   - Timing or performance considerations\n\n"
                "QUALITY REQUIREMENTS:\n"
                "- Be technically precise with addresses, opcodes, and register names\n"
                "- Use proper 65816 assembly terminology (LDA, STA, JSL, etc.)\n"
                "- Include specific examples and concrete details\n"
                "- Maintain coherent flow between sections\n"
                "- Avoid vague statements - be specific\n\n"
                "JSON FORMAT:\n"
                "{\n"
                "  \"instruction\": \"...\",\n"
                "  \"input\": \"...\",\n"
                "  \"output\": \"...\"\n"
                "}\n"
            )

        return template.format(
            name=item.name,
            symbol_type=item.symbol_type,
            context=context,
        )

    async def generate_sample(self, item: SourceItem) -> Optional[TrainingSample]:
        """Use teacher model to generate instruction from Gigaleak symbol."""
        if not isinstance(item, GigaleakSourceItem):
            return None

        if not self._orchestrator:
            await self.setup()

        prompt = self.get_teacher_prompt(item)

        try:
            from core.orchestrator_v2 import Provider, TaskTier

            response_obj = await asyncio.wait_for(
                self._orchestrator.generate(
                    prompt=prompt,
                    tier=TaskTier.CODING,
                    provider=Provider.GEMINI,
                ),
                timeout=120.0,  # Increased for GPU/slower models
            )

            response = response_obj.content

            # Extract JSON from response using robust parser
            data = extract_json_from_response(response)
            if not data:
                logger.warning(f"Failed to extract JSON from response for {item.name}")
                return None

            # Ensure all fields are strings (defensive conversion)
            instruction = str(data.get("instruction", ""))
            input_text = str(data.get("input", ""))
            output = str(data.get("output", item.content if isinstance(item.content, str) else ""))

            # Build KG entities safely
            kg_entities = [str(item.name)]
            if item.related_usdasm_symbol:
                kg_entities.append(str(item.related_usdasm_symbol))

            return TrainingSample(
                instruction=instruction,
                input=input_text,
                output=output,
                domain="gigaleak",
                source=item.source,
                teacher_model="gemini-2.0-flash",
                teacher_prompt=prompt,
                kg_entities=kg_entities,
            )

        except asyncio.TimeoutError:
            logger.warning(f"Timeout generating for {item.name}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed for {item.name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to generate for {item.name}: {e}")
            return None


if __name__ == "__main__":

    async def main():
        gen = GigaleakDataGenerator()
        await gen.setup()

        # Quick test
        items = await gen.extract_source_items()
        print(f"Found {len(items)} Gigaleak symbols")

        if items:
            # Test first 5
            result = await gen.run_generation(
                limit=5,
                output_path=Path("test_gigaleak_train.jsonl"),
            )
            print(f"Generated {result.processed} samples")

    import asyncio

    asyncio.run(main())
