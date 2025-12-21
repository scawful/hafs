"""ASM Data Generator for 65816 assembly training data.

Refactored from asm_instruction_generator.py to use the abstract DataGenerator
interface. Generates instruction-tuning data from ALTTP disassembly routines.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agents.training.base import DataGenerator, SourceItem, TrainingSample

logger = logging.getLogger(__name__)


@dataclass
class AsmSourceItem(SourceItem):
    """Source item for 65816 assembly routines."""

    code: str = ""
    bank: str = ""
    memory_access: list[str] = field(default_factory=list)
    description: str = ""
    address: str = ""

    @property
    def item_id(self) -> str:
        return f"{self.source}:{self.name}:{self.address}"


class AsmDataGenerator(DataGenerator):
    """Generate instruction-tuning data from 65816 assembly.

    Uses a teacher LLM (Gemini 2.0-Flash) to reverse-engineer the intent
    of assembly routines and generate natural language instructions.
    """

    def __init__(self):
        super().__init__(
            name="AsmDataGenerator",
            domain="asm",
            teacher_tier="coding",
        )
        self._unified_kb = None
        self._orchestrator = None

    async def setup(self):
        """Initialize resources and knowledge bases."""
        await super().setup()

        # Lazy import to avoid circular deps
        from agents.knowledge.alttp_unified import UnifiedALTTPKnowledge

        self._unified_kb = UnifiedALTTPKnowledge()
        await self._unified_kb.setup()

        # Use orchestrator from unified KB
        self._orchestrator = self._unified_kb._orchestrator

    async def extract_source_items(self) -> list[AsmSourceItem]:
        """Extract routines from vanilla and hack KBs."""
        if not self._unified_kb:
            await self.setup()

        items: list[AsmSourceItem] = []

        # Extract from vanilla KB
        if self._unified_kb._vanilla_kb:
            for name, routine in self._unified_kb._vanilla_kb._routines.items():
                routine_data = (
                    routine.to_dict() if hasattr(routine, "to_dict") else dict(routine)
                )

                # Safely extract code as string
                code = routine_data.get("code", "")
                if isinstance(code, (bytes, bytearray, memoryview)):
                    code = code.decode("utf-8", errors="replace")
                elif not isinstance(code, str):
                    code = str(code)

                items.append(
                    AsmSourceItem(
                        name=str(name),
                        content=code,
                        source="vanilla",
                        code=code,
                        bank=str(routine_data.get("bank", "")),
                        memory_access=[str(m) for m in routine_data.get("memory_access", [])],
                        description=str(routine_data.get("description", "")),
                        address=str(routine_data.get("address", "")),
                    )
                )

        # Extract from hack KB
        if self._unified_kb._hack_kb:
            for name, routine in self._unified_kb._hack_kb._routines.items():
                routine_data = (
                    routine.to_dict() if hasattr(routine, "to_dict") else dict(routine)
                )

                # Safely extract code as string
                code = routine_data.get("code", "")
                if isinstance(code, (bytes, bytearray, memoryview)):
                    code = code.decode("utf-8", errors="replace")
                elif not isinstance(code, str):
                    code = str(code)

                items.append(
                    AsmSourceItem(
                        name=str(name),
                        content=code,
                        source="hack",
                        code=code,
                        bank=str(routine_data.get("bank", "")),
                        memory_access=[str(m) for m in routine_data.get("memory_access", [])],
                        description=str(routine_data.get("description", "")),
                        address=str(routine_data.get("address", "")),
                    )
                )

        logger.info(f"Extracted {len(items)} ASM routines")
        return items

    def get_teacher_prompt(self, item: SourceItem) -> str:
        """Generate teacher prompt for ASM routine."""
        if not isinstance(item, AsmSourceItem):
            raise TypeError(f"Expected AsmSourceItem, got {type(item)}")

        return f"""I will give you a valid 65816 assembly routine used in the Zelda: A Link to the Past disassembly (usdasm).
Your task is to reverse-engineer the intent and write the user prompt (Instruction) that would request this specific code.

ROUTINE NAME: {item.name}
BANK: {item.bank}
EXISTING DESCRIPTION: {item.description}
MEMORY ACCESS: {", ".join(item.memory_access)}

CODE:
```asm
{item.code}
```

Respond with a JSON object containing:
1. "instruction": A natural language request that would lead to this code. Be specific about what the code does.
2. "input": Any necessary context (RAM addresses, hardware registers, specific symbols). Leave empty if not needed.
3. "output": The assembly code exactly as provided.

JSON FORMAT:
{{
  "instruction": "...",
  "input": "...",
  "output": "..."
}}
"""

    async def generate_sample(self, item: SourceItem) -> Optional[TrainingSample]:
        """Use teacher model to generate instruction from ASM routine."""
        if not isinstance(item, AsmSourceItem):
            return None

        if not self._orchestrator:
            await self.setup()

        prompt = self.get_teacher_prompt(item)

        try:
            from hafs.core.orchestrator_v2 import Provider, TaskTier

            response_obj = await asyncio.wait_for(
                self._orchestrator.generate(
                    prompt=prompt,
                    tier=TaskTier.CODING,
                    provider=Provider.GEMINI,
                ),
                timeout=30.0,
            )

            response = response_obj.content

            # Extract JSON from response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "{" in response:
                response = response[response.find("{") : response.rfind("}") + 1]

            data = json.loads(response)

            # Ensure all fields are strings (defensive conversion)
            # Convert to str() to avoid MemoryView issues with bytes/binary data
            instruction = str(data.get("instruction", "")).strip()
            input_text = str(data.get("input", "")).strip()
            output_data = data.get("output", item.code)

            # Safely convert output to string
            if isinstance(output_data, (bytes, bytearray, memoryview)):
                output = output_data.decode("utf-8", errors="replace")
            elif isinstance(output_data, str):
                output = output_data
            else:
                output = str(output_data)

            # Safely convert KG entities
            kg_entities = [str(item.name)]
            for m in item.memory_access:
                if isinstance(m, str):
                    kg_entities.append(m)
                else:
                    kg_entities.append(str(m))

            return TrainingSample(
                instruction=instruction,
                input=input_text,
                output=output,
                domain="asm",
                source=str(item.source),
                teacher_model="gemini-2.0-flash",
                teacher_prompt=str(prompt),
                kg_entities=kg_entities,
            )

        except asyncio.TimeoutError:
            logger.warning(f"Timeout generating for {item.name}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed for {item.name}: {e}")
            return None
        except Exception as e:
            import traceback
            logger.error(f"Failed to generate for {item.name}: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return None


if __name__ == "__main__":
    # Test script
    async def main():
        from agents.knowledge.alttp import ALTTPKnowledgeBase

        # Skip embeddings for speed
        original_load = ALTTPKnowledgeBase._load_embeddings
        ALTTPKnowledgeBase._load_embeddings = lambda self: None

        gen = AsmDataGenerator()
        await gen.setup()

        # Quick test
        result = await gen.run_generation(
            limit=5,
            output_path=Path("test_asm_train.jsonl"),
        )
        print(f"Generated {result.processed} samples in {result.duration_seconds:.1f}s")

    import asyncio

    asyncio.run(main())
