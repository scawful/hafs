"""Zelda3 Disassembly Generator for vanilla ALTTP training data.

Generates instruction-tuning data from the zelda3 vanilla disassembly project,
which has fully labeled routines and detailed comments.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agents.training.base import DataGenerator, SourceItem, TrainingSample
from agents.training.json_utils import extract_json_from_response
from agents.training.generators.prompt_templates import PromptTemplateRotator
from agents.training.resource_discovery import ZeldaResourceIndexer
from config.prompts import get_prompt

logger = logging.getLogger(__name__)

DEFAULT_TEACHER_PROMPT = """You are an expert SNES 65816 assembly programmer specializing in Zelda: A Link to the Past. {instruction_prefix}.

This is from the zelda3 vanilla disassembly project - fully labeled, production-quality Nintendo code.

ROUTINE: {label}
CONTEXT:
{context}

CODE:
```asm
{code}
```

Generate a JSON object with:

1. "instruction": A clear, specific request for this assembly code. Vary the perspective:
   - Explain what this vanilla ALTTP routine does
   - Ask how to implement this game mechanic
   - Request optimization or alternative approaches
   - Ask about hardware/timing considerations
   - Request step-by-step breakdown

2. "input": Technical context (2-3 sentences):
   - What game system/feature this routine belongs to
   - RAM addresses or hardware registers used
   - Timing constraints or NMI/IRQ considerations
   - Related routines in the call graph

3. "output": Complete explanation with the assembly code:
   - Brief overview of what the routine accomplishes
   - Line-by-line code with explanations
   - Why specific registers/addressing modes are used
   - How it fits into the larger game system
   - Edge cases or special handling

QUALITY REQUIREMENTS:
- Use precise 65816 syntax and mnemonics
- Include full addresses ($7E:XXXX for RAM, $XX:XXXX for ROM)
- Explain WHY specific techniques are used (not just WHAT)
- Reference Nintendo's original design intent where apparent
- Maintain technical accuracy about SNES hardware

JSON FORMAT:
{{
  "instruction": "...",
  "input": "...",
  "output": "..."
}}
"""


@dataclass
class Zelda3SourceItem(SourceItem):
    """Source item from zelda3 disassembly."""

    file_path: str = ""
    code: str = ""
    label: str = ""  # Main label/routine name
    address: str = ""  # ROM address (if found in comments)
    comments: list[str] = field(default_factory=list)  # Extracted comments
    labels_called: list[str] = field(default_factory=list)  # JSR/JSL targets
    line_start: int = 0
    line_end: int = 0

    @property
    def item_id(self) -> str:
        return f"zelda3:{self.file_path}:{self.label}"


class Zelda3DisasmGenerator(DataGenerator):
    """Generate from zelda3 vanilla disassembly.

    The zelda3 project has:
    - Fully labeled routines with descriptive names
    - Detailed comments explaining game logic
    - Organized by subsystem (sprites, dungeons, overworld, etc.)

    This provides high-quality vanilla ALTTP assembly as training data.
    """

    ZELDA3_PATH = Path.home() / "Code" / "zelda3"

    # Regex to detect labels (start of routine)
    LABEL_PATTERN = re.compile(r"^(\w+):\s*(?:;.*)?$")

    # Regex to detect JSR/JSL calls
    CALL_PATTERN = re.compile(r"\b(?:JSR|JSL)\s+(\w+)")

    # Regex to extract addresses from comments
    ADDRESS_PATTERN = re.compile(r"\$([0-9A-Fa-f]{2}):([0-9A-Fa-f]{4})")

    def __init__(self, use_template_variation: bool = True, max_routine_lines: int = 100):
        """Initialize zelda3 disassembly generator.

        Args:
            use_template_variation: Whether to use prompt template rotation
            max_routine_lines: Maximum lines per routine (avoid huge routines)
        """
        super().__init__(
            name="Zelda3DisasmGenerator",
            domain="asm",
            teacher_tier="coding",
        )
        self._indexer: Optional[ZeldaResourceIndexer] = None
        self._orchestrator = None
        self.use_template_variation = use_template_variation
        self.max_routine_lines = max_routine_lines

        # Initialize template rotator for diversity
        if self.use_template_variation:
            self.template_rotator = PromptTemplateRotator(domain="asm")

    async def setup(self):
        """Initialize resources and index zelda3 files."""
        await super().setup()

        # Initialize resource indexer
        self._indexer = ZeldaResourceIndexer()

        # Load or build index
        index_result = self._indexer.load_index()
        if not index_result:
            logger.info("Building zelda3 resource index...")
            index_result = await self._indexer.discover_and_index()

        logger.info(f"Zelda3 index loaded: {index_result.total_files} files")

        # Initialize orchestrator
        from core.orchestrator_v2 import UnifiedOrchestrator

        self._orchestrator = UnifiedOrchestrator()

    async def extract_source_items(self) -> list[Zelda3SourceItem]:
        """Extract labeled routines from zelda3 .asm files."""
        if not self._indexer:
            await self.setup()

        items: list[Zelda3SourceItem] = []

        # Get ALL ASM files - vanilla disassembly is scattered across sources
        # Filter out ROM hacks (Oracle-of-Secrets) to focus on vanilla
        zelda3_root = self.ZELDA3_PATH.resolve()
        asm_files = [
            f for f in self._indexer._files
            if f.file_type in ("asm", "asm_include")
            and Path(f.source_dir).resolve() == zelda3_root
            and "Oracle-of-Secrets" not in f.source_dir  # Exclude ROM hacks
            and "lib" not in f.relative_path  # Exclude library code
            and "imgui" not in f.relative_path  # Exclude imgui docs
        ]

        logger.info(f"Found {len(asm_files)} vanilla ASM files to process")

        for resource_file in asm_files:
            try:
                file_items = await self._extract_routines_from_file(resource_file.path)
                items.extend(file_items)
            except Exception as e:
                logger.error(f"Error processing {resource_file.path}: {e}")

        logger.info(f"Extracted {len(items)} routines from zelda3 disassembly")
        return items

    async def _extract_routines_from_file(self, path: Path) -> list[Zelda3SourceItem]:
        """Extract labeled routines from a single ASM file."""
        items: list[Zelda3SourceItem] = []

        try:
            content = path.read_text(errors="replace")
            lines = content.split("\n")

            current_routine: Optional[dict] = None

            for i, line in enumerate(lines):
                stripped = line.strip()

                # Check for label (start of new routine)
                label_match = self.LABEL_PATTERN.match(stripped)
                if label_match:
                    # Save previous routine if exists
                    if current_routine:
                        item = self._finalize_routine(current_routine, path)
                        if item:
                            items.append(item)

                    # Start new routine
                    label = label_match.group(1)
                    current_routine = {
                        "label": label,
                        "line_start": i,
                        "code_lines": [line],
                        "comments": [],
                        "labels_called": [],
                        "addresses": [],
                    }

                elif current_routine:
                    # Add line to current routine
                    current_routine["code_lines"].append(line)

                    # Extract comments
                    if ";" in line:
                        comment_start = line.index(";")
                        comment = line[comment_start + 1:].strip()
                        if comment:
                            current_routine["comments"].append(comment)

                    # Extract JSR/JSL calls
                    call_matches = self.CALL_PATTERN.findall(line)
                    current_routine["labels_called"].extend(call_matches)

                    # Extract addresses from comments
                    addr_matches = self.ADDRESS_PATTERN.findall(line)
                    for bank, addr in addr_matches:
                        current_routine["addresses"].append(f"${bank}:{addr}")

                    # Check for end of routine (empty line, new label, or max lines)
                    routine_length = len(current_routine["code_lines"])
                    is_empty = not stripped
                    is_next_label = i + 1 < len(lines) and self.LABEL_PATTERN.match(lines[i + 1].strip())

                    if is_empty or is_next_label or routine_length >= self.max_routine_lines:
                        item = self._finalize_routine(current_routine, path)
                        if item:
                            items.append(item)
                        current_routine = None

            # Save final routine if exists
            if current_routine:
                item = self._finalize_routine(current_routine, path)
                if item:
                    items.append(item)

        except Exception as e:
            logger.error(f"Error extracting routines from {path}: {e}")

        return items

    def _finalize_routine(self, routine_data: dict, path: Path) -> Optional[Zelda3SourceItem]:
        """Convert routine data dict to Zelda3SourceItem."""
        # Filter out very short routines (< 5 lines)
        if len(routine_data["code_lines"]) < 5:
            return None

        # Join code lines
        code = "\n".join(routine_data["code_lines"])

        # Build content for embedding
        content_parts = [
            f"Label: {routine_data['label']}",
            f"File: {path.name}",
        ]

        if routine_data["comments"]:
            # Use first few comments as description
            description = " ".join(routine_data["comments"][:3])
            content_parts.append(f"Description: {description}")

        if routine_data["addresses"]:
            content_parts.append(f"Addresses: {', '.join(set(routine_data['addresses'][:5]))}")

        content = "\n".join(content_parts)

        # Determine primary address (first one found)
        address = routine_data["addresses"][0] if routine_data["addresses"] else ""

        return Zelda3SourceItem(
            name=routine_data["label"],
            content=content,
            source="zelda3_disasm",
            file_path=str(path),
            code=code,
            label=routine_data["label"],
            address=address,
            comments=routine_data["comments"],
            labels_called=list(set(routine_data["labels_called"])),  # Deduplicate
            line_start=routine_data["line_start"],
            line_end=routine_data["line_start"] + len(routine_data["code_lines"]),
        )

    def get_teacher_prompt(self, item: SourceItem) -> str:
        """Generate teacher prompt for zelda3 routine."""
        if not isinstance(item, Zelda3SourceItem):
            raise TypeError(f"Expected Zelda3SourceItem, got {type(item)}")

        # Use template variation for instruction diversity
        instruction_prefix = "Generate high-quality training data for this vanilla ALTTP assembly routine"
        if self.use_template_variation and hasattr(self, 'template_rotator'):
            # Get next template and create varied instruction request
            template = self.template_rotator.get_next_template()
            # Use template as instruction variation hint
            instruction_prefix = f"Generate training data using this perspective: '{template.template}'"

        # Build context
        context_parts = []

        if item.file_path:
            file_name = Path(item.file_path).name
            context_parts.append(f"Source file: {file_name}")

        if item.address:
            context_parts.append(f"ROM address: {item.address}")

        if item.comments:
            # Use first few comments as description
            desc = " ".join(item.comments[:3])
            context_parts.append(f"Description: {desc}")

        if item.labels_called:
            called = ", ".join(item.labels_called[:5])
            context_parts.append(f"Calls: {called}")

        context = "\n".join(context_parts) if context_parts else "No additional context"

        template = get_prompt(
            "agents.training.generators.zelda3_generator.prompt",
            default=DEFAULT_TEACHER_PROMPT,
        )
        return template.format(
            instruction_prefix=instruction_prefix,
            label=item.label,
            context=context,
            code=item.code,
        )

    async def generate_sample(self, item: SourceItem) -> Optional[TrainingSample]:
        """Use teacher model to generate instruction from zelda3 routine."""
        if not isinstance(item, Zelda3SourceItem):
            return None

        if not self._orchestrator:
            await self.setup()

        prompt = self.get_teacher_prompt(item)

        try:
            response, model_name = await asyncio.wait_for(
                self.generate_with_rotation(prompt, tier="coding"),
                timeout=120.0,
            )
            if not response:
                return None

            # Extract JSON from response
            data = extract_json_from_response(response)
            if not data:
                logger.warning(f"Failed to extract JSON from response for {item.label}")
                return None

            # Build training sample
            instruction = str(data.get("instruction", "")).strip()
            input_text = str(data.get("input", "")).strip()
            output = str(data.get("output", item.code)).strip()

            # KG entities: label name, called labels, addresses
            kg_entities = [item.label] + item.labels_called
            if item.address:
                kg_entities.append(item.address)

            return TrainingSample(
                instruction=instruction,
                input=input_text,
                output=output,
                domain="asm",
                source="zelda3_disasm",
                teacher_model=model_name,
                teacher_prompt=str(prompt),
                kg_entities=kg_entities,
            )

        except asyncio.TimeoutError:
            logger.warning(f"Timeout generating for {item.label}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed for {item.label}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to generate for {item.label}: {e}")
            return None
