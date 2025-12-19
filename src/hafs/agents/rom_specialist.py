"""ROM Hacking Specialist Agent.

Expert in retro game ROM hacking with focus on:
- ALTTP (A Link to the Past) internals
- 65816 assembly language
- Oracle-of-Secrets ROM hack development
- Asar patch generation

References knowledge bases in local disassemblies and source code.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from hafs.agents.base import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class AsmReference:
    """Reference to an ASM symbol or routine."""

    name: str
    file_path: str
    line_number: int
    content: str
    category: str  # routine, macro, constant, variable
    description: Optional[str] = None


@dataclass
class VanillaPattern:
    """Pattern from vanilla ALTTP source."""

    name: str
    description: str
    file_path: str
    code_snippet: str
    usage_examples: list[str] = field(default_factory=list)


class RomHackingSpecialist(BaseAgent):
    """Expert agent for retro game ROM hacking.

    Specializes in:
    1. ALTTP (A Link to the Past) internals and mechanics
    2. 65816 assembly language analysis and generation
    3. Oracle-of-Secrets ROM hack development patterns
    4. Asar-compatible patch generation

    Knowledge Sources (loaded lazily):
    - ~/Code/alttp-gigaleak: Nintendo source materials
    - ~/Code/usdasm: ALTTP disassembly
    - ~/Code/Oracle-of-Secrets: Custom ROM hack
    - ~/Code/YAZE: Editor source code

    Example:
        agent = RomHackingSpecialist()
        await agent.setup()

        # Analyze some ASM
        analysis = await agent.analyze_asm(asm_code)

        # Find vanilla reference for behavior
        refs = await agent.find_vanilla_reference("boomerang throw")

        # Generate a patch
        patch = await agent.generate_patch("Add sword beam at full health")
    """

    # Knowledge source directories
    KNOWLEDGE_SOURCES = {
        "gigaleak": Path.home() / "Code" / "alttp-gigaleak",
        "disassembly": Path.home() / "Code" / "usdasm",
        "oracle": Path.home() / "Code" / "Oracle-of-Secrets",
        "yaze": Path.home() / "Code" / "YAZE",
    }

    # Common 65816 patterns and their meanings
    ASM_PATTERNS = {
        r"LDA\s+#?\$([0-9A-Fa-f]+)": "Load accumulator with value",
        r"STA\s+\$([0-9A-Fa-f]+)": "Store accumulator to address",
        r"JSR\s+(\w+)": "Jump to subroutine",
        r"JSL\s+(\w+)": "Jump to subroutine (long)",
        r"RTL": "Return from subroutine (long)",
        r"RTS": "Return from subroutine",
        r"BEQ\s+(\w+)": "Branch if equal (zero)",
        r"BNE\s+(\w+)": "Branch if not equal",
        r"PHB\s*$": "Push data bank register",
        r"PLB\s*$": "Pull data bank register",
        r"REP\s+#?\$(\d+)": "Reset processor status bits",
        r"SEP\s+#?\$(\d+)": "Set processor status bits",
    }

    # Important ALTTP memory addresses
    ALTTP_ADDRESSES = {
        "$7E0010": "Submodule index",
        "$7E0011": "Main module index (game state)",
        "$7E0012": "Sub-submodule index",
        "$7E0020": "Link's X coordinate (low)",
        "$7E0022": "Link's Y coordinate (low)",
        "$7E002E": "Link's direction",
        "$7E0031": "Button input pressed",
        "$7E0032": "Button input held",
        "$7E0303": "Link's current sword",
        "$7E0304": "Link's current shield",
        "$7E0343": "Link's current health",
        "$7E0360": "Link's max health",
        "$7E036C": "Current world (LW=0x00, DW=0x40)",
        "$7E040C": "Room ID (low)",
        "$7E048E": "Object interaction table",
        "$7EF300": "SRAM block start",
    }

    def __init__(self):
        super().__init__(
            "RomHackingSpecialist",
            "Expert in ALTTP ROM hacking, 65816 ASM, and Oracle-of-Secrets development."
        )

        # Cached knowledge
        self._asm_index: dict[str, list[AsmReference]] = {}
        self._vanilla_patterns: list[VanillaPattern] = []
        self._knowledge_loaded = False

        # Use reasoning tier for complex ASM analysis
        self.model_tier = "reasoning"

    async def setup(self):
        """Initialize the ROM hacking specialist."""
        await super().setup()

        # Check which knowledge sources are available
        available = []
        for name, path in self.KNOWLEDGE_SOURCES.items():
            if path.exists():
                available.append(name)

        logger.info(f"RomHackingSpecialist initialized. Available sources: {available}")

    async def _ensure_knowledge_loaded(self):
        """Lazy load knowledge from source files."""
        if self._knowledge_loaded:
            return

        logger.info("Loading ROM hacking knowledge base...")

        # Index key files from each source
        await asyncio.gather(
            self._index_source("disassembly", ["*.asm", "*.inc"]),
            self._index_source("oracle", ["*.asm", "*.inc"]),
            self._index_source("yaze", ["*.asm", "*.h"]),
        )

        self._knowledge_loaded = True
        logger.info(f"Loaded {sum(len(v) for v in self._asm_index.values())} ASM references")

    async def _index_source(self, source_name: str, patterns: list[str]):
        """Index ASM files from a source."""
        source_path = self.KNOWLEDGE_SOURCES.get(source_name)
        if not source_path or not source_path.exists():
            return

        self._asm_index[source_name] = []

        for pattern in patterns:
            for asm_file in source_path.rglob(pattern):
                try:
                    await self._index_asm_file(source_name, asm_file)
                except Exception as e:
                    logger.debug(f"Error indexing {asm_file}: {e}")

    async def _index_asm_file(self, source_name: str, file_path: Path):
        """Extract symbols and routines from an ASM file."""
        try:
            content = file_path.read_text(errors="ignore")
            lines = content.split("\n")

            for i, line in enumerate(lines):
                # Match labels (routine definitions)
                if match := re.match(r"^(\w+):\s*(?:;(.*))?$", line):
                    self._asm_index[source_name].append(AsmReference(
                        name=match.group(1),
                        file_path=str(file_path),
                        line_number=i + 1,
                        content=self._get_context(lines, i, 5),
                        category="routine",
                        description=match.group(2) if match.group(2) else None,
                    ))

                # Match constants/equates
                elif match := re.match(r"^(\w+)\s*=\s*\$?([0-9A-Fa-f]+)\s*(?:;(.*))?$", line):
                    self._asm_index[source_name].append(AsmReference(
                        name=match.group(1),
                        file_path=str(file_path),
                        line_number=i + 1,
                        content=line,
                        category="constant",
                        description=match.group(3) if match.group(3) else None,
                    ))

                # Match macros
                elif match := re.match(r"^macro\s+(\w+)", line, re.IGNORECASE):
                    self._asm_index[source_name].append(AsmReference(
                        name=match.group(1),
                        file_path=str(file_path),
                        line_number=i + 1,
                        content=self._get_context(lines, i, 10),
                        category="macro",
                    ))
        except Exception as e:
            logger.debug(f"Failed to index {file_path}: {e}")

    def _get_context(self, lines: list[str], center: int, radius: int) -> str:
        """Get surrounding lines for context."""
        start = max(0, center - radius)
        end = min(len(lines), center + radius + 1)
        return "\n".join(lines[start:end])

    async def analyze_asm(self, code: str) -> dict[str, Any]:
        """Analyze 65816 assembly code.

        Args:
            code: ASM code to analyze.

        Returns:
            Analysis with explanation, patterns found, and suggestions.
        """
        await self._ensure_knowledge_loaded()

        # Pattern detection
        patterns_found = []
        for pattern, meaning in self.ASM_PATTERNS.items():
            if matches := re.findall(pattern, code, re.MULTILINE):
                patterns_found.append({
                    "pattern": pattern,
                    "meaning": meaning,
                    "occurrences": len(matches),
                })

        # Address references
        address_refs = []
        for match in re.findall(r"\$7E[0-9A-Fa-f]{4}", code, re.IGNORECASE):
            addr = match.upper()
            if addr in self.ALTTP_ADDRESSES:
                address_refs.append({
                    "address": addr,
                    "meaning": self.ALTTP_ADDRESSES[addr],
                })

        # Generate detailed analysis using LLM
        analysis_prompt = f"""Analyze this 65816 assembly code from an ALTTP context:

```asm
{code}
```

I've detected these patterns:
{patterns_found}

And these ALTTP-specific address references:
{address_refs}

Provide:
1. A plain English explanation of what this code does
2. Any ALTTP-specific behaviors (sprites, graphics, game state, etc.)
3. Potential issues or improvements
4. Related vanilla routines this might be based on

Format as structured analysis."""

        llm_analysis = await self.generate_thought(analysis_prompt)

        return {
            "code_length": len(code.split("\n")),
            "patterns_found": patterns_found,
            "address_references": address_refs,
            "analysis": llm_analysis,
        }

    async def find_vanilla_reference(
        self,
        behavior: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search disassembly for similar behavior.

        Args:
            behavior: Description of the behavior to find.
            limit: Maximum references to return.

        Returns:
            List of matching references with context.
        """
        await self._ensure_knowledge_loaded()

        # Search index for relevant routines
        search_terms = behavior.lower().split()
        matches = []

        for source, refs in self._asm_index.items():
            for ref in refs:
                # Score by term matches in name and description
                score = 0
                searchable = f"{ref.name} {ref.description or ''}".lower()

                for term in search_terms:
                    if term in searchable:
                        score += 1

                if score > 0:
                    matches.append({
                        "source": source,
                        "name": ref.name,
                        "file": ref.file_path,
                        "line": ref.line_number,
                        "category": ref.category,
                        "description": ref.description,
                        "score": score,
                        "context": ref.content[:300],
                    })

        # Sort by score
        matches.sort(key=lambda x: x["score"], reverse=True)
        top_matches = matches[:limit]

        # Enhance with LLM analysis if we have good matches
        if top_matches and self.orchestrator:
            enhance_prompt = f"""Given these vanilla ALTTP code references for "{behavior}":

{top_matches}

Briefly explain which is most relevant and why. How could these be adapted?"""

            try:
                self.model_tier = "fast"  # Quick analysis
                enhancement = await self.generate_thought(enhance_prompt)
                self.model_tier = "reasoning"  # Restore

                return {
                    "query": behavior,
                    "matches": top_matches,
                    "recommendation": enhancement,
                }
            except:
                pass

        return {
            "query": behavior,
            "matches": top_matches,
        }

    async def suggest_implementation(self, feature: str) -> str:
        """Suggest how to implement a feature based on vanilla patterns.

        Args:
            feature: Description of desired feature.

        Returns:
            Implementation suggestion with code examples.
        """
        await self._ensure_knowledge_loaded()

        # Find related vanilla code first
        refs = await self.find_vanilla_reference(feature)

        # Build context from matches
        vanilla_context = ""
        if refs.get("matches"):
            for match in refs["matches"][:3]:
                vanilla_context += f"\n\n--- {match['source']}: {match['name']} ---\n"
                vanilla_context += match.get("context", "")

        prompt = f"""I need to implement this feature in an ALTTP ROM hack:

FEATURE: {feature}

VANILLA REFERENCES FOUND:
{vanilla_context}

KNOWN MEMORY ADDRESSES:
{self.ALTTP_ADDRESSES}

Provide:
1. High-level implementation strategy
2. Key memory addresses to use
3. Suggested ASM code skeleton (Asar-compatible)
4. Potential pitfalls to avoid
5. Testing recommendations"""

        return await self.generate_thought(prompt)

    async def generate_patch(
        self,
        spec: str,
        base_address: Optional[str] = None,
    ) -> str:
        """Generate an Asar-compatible ASM patch.

        Args:
            spec: Specification of what the patch should do.
            base_address: Optional base address for the patch.

        Returns:
            Asar-compatible ASM patch code.
        """
        await self._ensure_knowledge_loaded()

        # Get implementation suggestion first
        suggestion = await self.suggest_implementation(spec)

        prompt = f"""Generate a complete Asar-compatible ASM patch for ALTTP.

SPECIFICATION: {spec}

IMPLEMENTATION NOTES:
{suggestion}

{f"BASE ADDRESS: {base_address}" if base_address else "Use freespace (org $XX:XXXX or freecode)"}

Requirements:
1. Use Asar syntax (org, freecode, macro, etc.)
2. Include proper bank handling (PHB/PLB)
3. Add comments explaining each section
4. Use REP/SEP correctly for 8/16 bit modes
5. Include a header comment with description

Generate the complete patch file content."""

        patch = await self.generate_thought(prompt)

        # Clean up if wrapped in code blocks
        if "```" in patch:
            lines = patch.split("\n")
            cleaned = []
            in_code = False
            for line in lines:
                if line.startswith("```"):
                    in_code = not in_code
                elif in_code:
                    cleaned.append(line)
            patch = "\n".join(cleaned)

        return patch

    async def explain_address(self, address: str) -> str:
        """Explain what an ALTTP memory address is used for.

        Args:
            address: Memory address (e.g., "$7E0343").

        Returns:
            Explanation of the address.
        """
        # Normalize address format
        addr = address.upper().replace("0X", "$")
        if not addr.startswith("$"):
            addr = f"${addr}"

        # Check known addresses
        if addr in self.ALTTP_ADDRESSES:
            known = self.ALTTP_ADDRESSES[addr]
        else:
            known = "Unknown address"

        # Generate detailed explanation
        prompt = f"""Explain this ALTTP memory address: {addr}

Known info: {known}

Provide:
1. What this address controls
2. Common values and their meanings
3. Related addresses
4. How it's typically used in ROM hacks"""

        return await self.generate_thought(prompt)

    async def review_patch(self, patch_code: str) -> dict[str, Any]:
        """Review a patch for issues and improvements.

        Args:
            patch_code: The patch code to review.

        Returns:
            Review with issues, suggestions, and rating.
        """
        analysis = await self.analyze_asm(patch_code)

        prompt = f"""Review this ALTTP ASM patch for issues:

```asm
{patch_code}
```

Initial analysis:
{analysis}

Check for:
1. Bank handling issues (missing PHB/PLB)
2. Accumulator/index size issues (REP/SEP)
3. Stack imbalance
4. Potential crashes or softlocks
5. Compatibility with common ROM hacks
6. Code style and organization

Provide:
- List of issues (critical/warning/info)
- Suggested fixes
- Overall quality rating (1-10)"""

        review = await self.generate_thought(prompt)

        return {
            "analysis": analysis,
            "review": review,
        }

    async def run_task(self, task: str) -> dict[str, Any]:
        """Run a ROM hacking task.

        Args:
            task: Task specification:
                - "analyze:CODE" - Analyze ASM code
                - "find:BEHAVIOR" - Find vanilla reference
                - "impl:FEATURE" - Get implementation suggestion
                - "patch:SPEC" - Generate patch
                - "explain:ADDRESS" - Explain memory address
                - "review:CODE" - Review patch code

        Returns:
            Task result.
        """
        if task.startswith("analyze:"):
            code = task[8:].strip()
            return await self.analyze_asm(code)

        elif task.startswith("find:"):
            behavior = task[5:].strip()
            return await self.find_vanilla_reference(behavior)

        elif task.startswith("impl:"):
            feature = task[5:].strip()
            suggestion = await self.suggest_implementation(feature)
            return {"suggestion": suggestion}

        elif task.startswith("patch:"):
            spec = task[6:].strip()
            patch = await self.generate_patch(spec)
            return {"patch": patch}

        elif task.startswith("explain:"):
            address = task[8:].strip()
            explanation = await self.explain_address(address)
            return {"explanation": explanation}

        elif task.startswith("review:"):
            code = task[7:].strip()
            return await self.review_patch(code)

        else:
            return {
                "error": "Unknown task",
                "usage": [
                    "analyze:CODE - Analyze ASM code",
                    "find:BEHAVIOR - Find vanilla reference",
                    "impl:FEATURE - Get implementation suggestion",
                    "patch:SPEC - Generate patch",
                    "explain:ADDRESS - Explain memory address",
                    "review:CODE - Review patch code",
                ],
            }
