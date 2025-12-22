"""C++ Data Generator for yaze source code training data.

Generates instruction-tuning data from yaze C++ source code,
focusing on emulation, ROM editing, and Zelda-specific modules.
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
from agents.training.generators.prompt_templates import PromptTemplateRotator
from config.prompts import get_prompt

logger = logging.getLogger(__name__)


@dataclass
class CppSourceItem(SourceItem):
    """Source item for C++ code units."""

    file_path: str = ""
    code: str = ""
    kind: str = "function"  # function, class, method, struct
    signature: str = ""
    docstring: str = ""
    includes: list[str] = field(default_factory=list)
    namespace: str = ""
    class_name: str = ""

    @property
    def item_id(self) -> str:
        return f"{self.file_path}:{self.kind}:{self.name}"


class CppDataGenerator(DataGenerator):
    """Generate instruction-tuning data from yaze C++ source.

    Parses C++ files to extract functions, classes, and methods,
    then uses a teacher LLM to generate natural language instructions.
    """

    # Default paths relative to home directory
    DEFAULT_YAZE_PATH = Path.home() / "Code" / "yaze"

    # Focus areas for extraction
    FOCUS_AREAS = [
        "src/app/core",  # Core application
        "src/app/emu",  # Emulation
        "src/app/zelda3",  # Zelda-specific
        "src/app/gfx",  # Graphics
        "src/app/rom",  # ROM handling
        "src/lib",  # Libraries
    ]

    EXCLUDE_PATH_PARTS = {
        "ui",
        "gui",
        "widgets",
        "views",
        "imgui",
        "qt",
    }

    # Regex patterns for C++ parsing
    FUNCTION_PATTERN = re.compile(
        r"(?P<return_type>[\w:<>\s\*&]+)\s+"
        r"(?P<name>\w+)\s*\("
        r"(?P<params>[^)]*)\)\s*"
        r"(?:const)?\s*(?:override)?\s*{"
    )

    CLASS_PATTERN = re.compile(
        r"(?:class|struct)\s+(?P<name>\w+)\s*"
        r"(?::\s*(?:public|private|protected)\s+[\w:]+)?\s*{"
    )

    METHOD_PATTERN = re.compile(
        r"(?P<return_type>[\w:<>\s\*&]+)\s+"
        r"(?P<class>\w+)::(?P<name>\w+)\s*\("
        r"(?P<params>[^)]*)\)\s*"
        r"(?:const)?\s*(?:override)?\s*{"
    )

    def __init__(self, yaze_path: Optional[Path] = None, use_template_variation: bool = True):
        super().__init__(
            name="CppDataGenerator",
            domain="cpp",
            teacher_tier="coding",
        )
        self.yaze_path = yaze_path or self.DEFAULT_YAZE_PATH
        self._orchestrator = None
        self.use_template_variation = use_template_variation

        # Initialize template rotator for diversity
        if self.use_template_variation:
            self.template_rotator = PromptTemplateRotator(domain="cpp")

    async def setup(self):
        """Initialize resources."""
        await super().setup()

        # Initialize orchestrator
        from core.orchestrator_v2 import UnifiedOrchestrator

        self._orchestrator = UnifiedOrchestrator()

    async def extract_source_items(self) -> list[CppSourceItem]:
        """Extract functions and classes from yaze codebase."""
        items: list[CppSourceItem] = []

        for area in self.FOCUS_AREAS:
            area_path = self.yaze_path / area
            if not area_path.exists():
                logger.warning(f"Path not found: {area_path}")
                continue

            # Process .cc and .h files
            for pattern in ["*.cc", "*.h"]:
                for file_path in area_path.rglob(pattern):
                    try:
                        if self._should_skip_path(file_path):
                            continue
                        file_items = await self._parse_cpp_file(file_path)
                        items.extend(file_items)
                    except Exception as e:
                        logger.error(f"Error parsing {file_path}: {e}")

        logger.info(f"Extracted {len(items)} C++ code units from yaze")
        return items

    def _should_skip_path(self, path: Path) -> bool:
        """Skip UI/boilerplate paths to keep ROM-hacking signal high."""
        lowered_parts = {part.lower() for part in path.parts}
        return bool(self.EXCLUDE_PATH_PARTS & lowered_parts)

    async def _parse_cpp_file(self, path: Path) -> list[CppSourceItem]:
        """Parse a C++ file to extract functions, classes, methods."""
        items: list[CppSourceItem] = []
        content = path.read_text(errors="replace")
        lines = content.split("\n")

        # Extract includes
        includes = [
            line.strip()
            for line in lines
            if line.strip().startswith("#include")
        ]

        # Extract namespace
        namespace = ""
        ns_match = re.search(r"namespace\s+(\w+)", content)
        if ns_match:
            namespace = ns_match.group(1)

        # Simple brace-matching for code extraction
        def extract_block(start_idx: int, lines: list[str]) -> tuple[str, int]:
            """Extract a code block starting from a line with opening brace."""
            brace_count = 0
            block_lines = []
            found_open = False

            for i in range(start_idx, len(lines)):
                line = lines[i]
                block_lines.append(line)

                brace_count += line.count("{") - line.count("}")

                if "{" in line:
                    found_open = True

                if found_open and brace_count == 0:
                    return "\n".join(block_lines), i

            return "\n".join(block_lines), len(lines) - 1

        # Find functions (not methods)
        for i, line in enumerate(lines):
            # Skip lines inside comments or strings
            if line.strip().startswith("//") or line.strip().startswith("/*"):
                continue

            match = self.FUNCTION_PATTERN.search(line)
            if match and "::" not in match.group("name"):
                code, end_idx = extract_block(i, lines)

                # Get docstring (comment above)
                docstring = ""
                if i > 0 and lines[i - 1].strip().startswith("//"):
                    docstring = lines[i - 1].strip()[2:].strip()
                elif i > 0 and lines[i - 1].strip().endswith("*/"):
                    # Find start of block comment
                    for j in range(i - 1, max(0, i - 20), -1):
                        if "/*" in lines[j]:
                            docstring = "\n".join(lines[j : i])
                            break

                items.append(
                    CppSourceItem(
                        name=match.group("name"),
                        content=code,
                        source=str(path.relative_to(self.yaze_path)),
                        file_path=str(path),
                        code=code,
                        kind="function",
                        signature=f"{match.group('return_type')} {match.group('name')}({match.group('params')})",
                        docstring=docstring,
                        includes=includes,
                        namespace=namespace,
                    )
                )

        # Find class/struct definitions
        for match in self.CLASS_PATTERN.finditer(content):
            class_name = match.group("name")
            start_idx = content[: match.start()].count("\n")
            code, _ = extract_block(start_idx, lines)

            items.append(
                CppSourceItem(
                    name=class_name,
                    content=code,
                    source=str(path.relative_to(self.yaze_path)),
                    file_path=str(path),
                    code=code,
                    kind="class",
                    signature=f"class {class_name}",
                    includes=includes,
                    namespace=namespace,
                )
            )

        # Find method implementations (ClassName::MethodName)
        for match in self.METHOD_PATTERN.finditer(content):
            start_idx = content[: match.start()].count("\n")
            code, _ = extract_block(start_idx, lines)

            items.append(
                CppSourceItem(
                    name=match.group("name"),
                    content=code,
                    source=str(path.relative_to(self.yaze_path)),
                    file_path=str(path),
                    code=code,
                    kind="method",
                    signature=f"{match.group('return_type')} {match.group('class')}::{match.group('name')}({match.group('params')})",
                    class_name=match.group("class"),
                    includes=includes,
                    namespace=namespace,
                )
            )

        return items

    def get_teacher_prompt(self, item: SourceItem) -> str:
        """Generate teacher prompt for C++ code unit."""
        if not isinstance(item, CppSourceItem):
            raise TypeError(f"Expected CppSourceItem, got {type(item)}")

        context_parts = []
        if item.namespace:
            context_parts.append(f"Namespace: {item.namespace}")
        if item.class_name:
            context_parts.append(f"Class: {item.class_name}")
        if item.docstring:
            context_parts.append(f"Documentation: {item.docstring}")
        if item.includes:
            context_parts.append(f"Includes: {', '.join(item.includes[:5])}")

        context = "\n".join(context_parts) if context_parts else "No additional context"

        # Use template variation for instruction diversity
        task_prefix = "reverse-engineer the intent and write the user prompt (Instruction) that would request this specific code"
        if self.use_template_variation and hasattr(self, 'template_rotator'):
            # Get next template and create varied task request
            template = self.template_rotator.get_next_template()
            # Use template as task variation hint
            task_prefix = f"reverse-engineer using this perspective: '{template.template}', then write the user prompt"

        template = get_prompt("agents.training.generators.cpp_generator.prompt", "")
        if not template:
            template = (
                "I will give you a C++ {kind} from the yaze project (a SNES emulator and ROM editor for Zelda: A Link to the Past).\n"
                "Your task is to {task_prefix}.\n\n"
                "CODE TYPE: {kind}\n"
                "SIGNATURE: {signature}\n"
                "FILE: {source}\n"
                "CONTEXT:\n{context}\n\n"
                "CODE:\n```cpp\n{code}\n```\n\n"
                "Respond with a JSON object containing:\n"
                "1. \"instruction\": A natural language request that would lead to writing this code. Be specific about what it does.\n"
                "2. \"input\": Any necessary context (APIs, dependencies, constraints). Leave empty if self-contained.\n"
                "3. \"output\": The C++ code exactly as provided.\n\n"
                "JSON FORMAT:\n"
                "{{\n"
                "  \"instruction\": \"...\",\n"
                "  \"input\": \"...\",\n"
                "  \"output\": \"...\"\n"
                "}}\n"
            )

        return template.format(
            kind=item.kind,
            signature=item.signature,
            source=item.source,
            context=context,
            code=item.code,
            task_prefix=task_prefix,
        )

    async def generate_sample(self, item: SourceItem) -> Optional[TrainingSample]:
        """Use teacher model to generate instruction from C++ code."""
        if not isinstance(item, CppSourceItem):
            return None

        if not self._orchestrator:
            await self.setup()

        # Skip very short code (likely declarations only)
        if len(item.code) < 50:
            return None

        prompt = self.get_teacher_prompt(item)

        try:
            from core.orchestrator_v2 import Provider, TaskTier

            response_obj = await asyncio.wait_for(
                self._orchestrator.generate(
                    prompt=prompt,
                    tier=TaskTier.CODING,
                    provider=Provider.GEMINI,
                ),
                timeout=45.0,
            )

            response = response_obj.content

            # Extract JSON from response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "{" in response:
                response = response[response.find("{") : response.rfind("}") + 1]

            data = json.loads(response)

            return TrainingSample(
                instruction=data.get("instruction", ""),
                input=data.get("input", ""),
                output=data.get("output", item.code),
                domain="cpp",
                source=item.source,
                teacher_model="gemini-2.0-flash",
                teacher_prompt=prompt,
                kg_entities=[item.name, item.class_name] if item.class_name else [item.name],
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
        gen = CppDataGenerator()
        await gen.setup()

        # Quick test
        items = await gen.extract_source_items()
        print(f"Found {len(items)} C++ code units")

        if items:
            result = await gen.run_generation(
                limit=5,
                output_path=Path("test_cpp_train.jsonl"),
            )
            print(f"Generated {result.processed} samples")

    import asyncio
    asyncio.run(main())
