"""ALTTP Knowledge Base Agent.

Builds a comprehensive, searchable knowledge base from ALTTP disassembly sources.
Extracts symbols, routines, and semantics from 65816 ASM files.
Generates embeddings for semantic search.
Uses Gemini for deep analysis and documentation generation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agents.core import BaseAgent
from hafs.core.embeddings import BatchEmbeddingManager
from hafs.core.orchestrator_v2 import UnifiedOrchestrator, TaskTier, Provider

logger = logging.getLogger(__name__)


@dataclass
class AsmSymbol:
    """A symbol extracted from ASM source."""

    name: str
    address: str
    category: str  # wram, register, routine, constant, data
    description: str = ""
    bank: Optional[str] = None
    file_path: Optional[str] = None
    line_number: int = 0
    code_context: str = ""

    # Related symbols
    references: list[str] = field(default_factory=list)
    referenced_by: list[str] = field(default_factory=list)

    # Semantic data
    embedding_id: Optional[str] = None
    semantic_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def id(self) -> str:
        """Unique identifier for this symbol."""
        return f"{self.category}:{self.name}"


@dataclass
class AsmRoutine:
    """A routine/function extracted from ASM source."""

    name: str
    address: str
    bank: str
    file_path: str
    line_start: int
    line_end: int
    code: str
    description: str = ""

    # Analysis results
    calls: list[str] = field(default_factory=list)
    called_by: list[str] = field(default_factory=list)
    memory_access: list[str] = field(default_factory=list)

    # Semantic data
    embedding_id: Optional[str] = None
    purpose: str = ""
    complexity: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GameModule:
    """A game module (from Module_MainRouting)."""

    id: int
    name: str
    description: str = ""
    routines: list[str] = field(default_factory=list)


class ALTTPKnowledgeBase(BaseAgent):
    """Comprehensive ALTTP knowledge base with semantic search.

    Extracts and indexes:
    - WRAM symbols and their purposes
    - Hardware register definitions
    - Routines and their call graphs
    - Game modules and state machines
    - Data tables and their formats

    Uses:
    - UnifiedOrchestrator v2 for multi-provider LLM access
    - Gemini for deep semantic analysis
    - Embeddings for similarity search

    Example:
        kb = ALTTPKnowledgeBase()
        await kb.setup()

        # Build from disassembly
        await kb.build_from_source("/path/to/usdasm")

        # Search
        results = await kb.search("Link's position")

        # Get detailed analysis
        analysis = await kb.analyze_routine("Module07_Underworld")
    """

    # Known categories of WRAM regions
    WRAM_CATEGORIES = {
        (0x0000, 0x00FF): "direct_page",
        (0x0100, 0x01FF): "stack",
        (0x0200, 0x02FF): "oam_buffer",
        (0x0300, 0x03FF): "link_state",
        (0x0400, 0x04FF): "objects",
        (0x0500, 0x05FF): "sprites",
        (0x0D00, 0x0DFF): "ancillae",
        (0x0E00, 0x0FFF): "misc",
        (0xF300, 0xF4FF): "sram_mirror",
    }

    # 65816 instruction patterns for analysis
    INSTRUCTION_PATTERNS = {
        "JSR": r"JSR\.?\w?\s+(\w+)",
        "JSL": r"JSL\.?\w?\s+(\w+)",
        "JMP": r"JMP\.?\w?\s+(\w+)",
        "LDA": r"LDA\.?\w?\s+[#\$\w]+",
        "STA": r"STA\.?\w?\s+[#\$\w]+",
        "STZ": r"STZ\.?\w?\s+[#\$\w]+",
    }

    def __init__(
        self,
        source_path: Optional[Path] = None,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        super().__init__(
            "ALTTPKnowledgeBase",
            "Build and search ALTTP disassembly knowledge base with semantic embeddings."
        )

        # Source configuration
        self.source_path = source_path or Path.home() / "Code" / "usdasm"

        # Knowledge base storage
        self.kb_dir = self.context_root / "knowledge" / "alttp"
        self.kb_dir.mkdir(parents=True, exist_ok=True)

        self.symbols_file = self.kb_dir / "symbols.json"
        self.routines_file = self.kb_dir / "routines.json"
        self.modules_file = self.kb_dir / "modules.json"
        self._embedding_provider = embedding_provider
        self._embedding_model = embedding_model
        storage_id = BatchEmbeddingManager.resolve_storage_id(
            self._embedding_provider,
            self._embedding_model,
        )
        self.embeddings_dir = BatchEmbeddingManager.resolve_embeddings_dir(
            self.kb_dir,
            storage_id,
        )
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        # In-memory caches
        self._symbols: dict[str, AsmSymbol] = {}
        self._routines: dict[str, AsmRoutine] = {}
        self._modules: dict[int, GameModule] = {}
        self._embeddings: dict[str, list[float]] = {}

        # Orchestrator for LLM access
        self._orchestrator: Optional[UnifiedOrchestrator] = None
        self._embedding_manager: Optional[BatchEmbeddingManager] = None

        # Use reasoning tier for deep analysis
        self.model_tier = "reasoning"

    async def setup(self):
        """Initialize the knowledge base."""
        await super().setup()

        # Initialize orchestrator
        self._orchestrator = UnifiedOrchestrator()
        await self._orchestrator.initialize()
        self._embedding_manager = BatchEmbeddingManager(
            kb_dir=self.kb_dir,
            orchestrator=self._orchestrator,
            embedding_provider=self._embedding_provider,
            embedding_model=self._embedding_model,
        )

        # Load existing knowledge if available
        self._load_knowledge()

        logger.info(f"ALTTPKnowledgeBase ready. {len(self._symbols)} symbols, {len(self._routines)} routines")

    def _load_knowledge(self):
        """Load existing knowledge from disk."""
        if self.symbols_file.exists():
            try:
                data = json.loads(self.symbols_file.read_text())
                for item in data:
                    sym = AsmSymbol(**item)
                    self._symbols[sym.id] = sym
            except Exception as e:
                logger.warning(f"Failed to load symbols: {e}")

        if self.routines_file.exists():
            try:
                data = json.loads(self.routines_file.read_text())
                for item in data:
                    routine = AsmRoutine(**item)
                    self._routines[routine.name] = routine
            except Exception as e:
                logger.warning(f"Failed to load routines: {e}")

        if self.modules_file.exists():
            try:
                data = json.loads(self.modules_file.read_text())
                for item in data:
                    mod = GameModule(**item)
                    self._modules[mod.id] = mod
            except Exception as e:
                logger.warning(f"Failed to load modules: {e}")

        # Load embeddings from disk
        self._load_embeddings()

    def _load_embeddings(self):
        """Load embeddings from disk."""
        if not self.embeddings_dir.exists():
            return

        emb_count = 0
        for emb_file in self.embeddings_dir.glob("*.json"):
            try:
                data = json.loads(emb_file.read_text())
                emb_id = data.get("id")
                embedding = data.get("embedding")
                if emb_id and embedding:
                    self._embeddings[emb_id] = embedding
                    emb_count += 1
            except Exception as e:
                logger.debug(f"Failed to load embedding {emb_file}: {e}")

        if emb_count > 0:
            logger.info(f"Loaded {emb_count} embeddings from disk")

    def _save_knowledge(self):
        """Save knowledge to disk."""
        try:
            symbols_data = [s.to_dict() for s in self._symbols.values()]
            self.symbols_file.write_text(json.dumps(symbols_data, indent=2))

            routines_data = [r.to_dict() for r in self._routines.values()]
            self.routines_file.write_text(json.dumps(routines_data, indent=2))

            modules_data = [asdict(m) for m in self._modules.values()]
            self.modules_file.write_text(json.dumps(modules_data, indent=2))

            logger.info(f"Saved {len(self._symbols)} symbols, {len(self._routines)} routines")
        except Exception as e:
            logger.error(f"Failed to save knowledge: {e}")

    async def build_from_source(
        self,
        source_path: Optional[Path] = None,
        generate_embeddings: bool = True,
        deep_analysis: bool = True,
    ) -> dict[str, int]:
        """Build knowledge base from disassembly source.

        Args:
            source_path: Path to usdasm directory.
            generate_embeddings: Whether to generate semantic embeddings.
            deep_analysis: Whether to use LLM for deep analysis.

        Returns:
            Statistics about extracted knowledge.
        """
        source = Path(source_path) if source_path else self.source_path

        if not source.exists():
            raise ValueError(f"Source path not found: {source}")

        logger.info(f"Building ALTTP knowledge base from {source}")
        start_time = time.time()

        # Step 1: Extract WRAM symbols
        wram_file = source / "wram.asm"
        if wram_file.exists():
            await self._extract_wram_symbols(wram_file)

        # Step 2: Extract register definitions
        registers_file = source / "registers.asm"
        if registers_file.exists():
            await self._extract_register_symbols(registers_file)

        # Step 3: Extract routines from bank files
        for bank_file in sorted(source.glob("bank_*.asm")):
            await self._extract_routines(bank_file)

        # Step 4: Extract game modules
        await self._extract_game_modules(source / "bank_00.asm")

        # Step 5: Build cross-references
        self._build_cross_references()

        # Step 6: Generate embeddings
        if generate_embeddings:
            await self._generate_all_embeddings()

        # Step 7: Deep analysis with Gemini
        if deep_analysis:
            await self._perform_deep_analysis()

        # Save results
        self._save_knowledge()

        elapsed = time.time() - start_time
        stats = {
            "symbols": len(self._symbols),
            "routines": len(self._routines),
            "modules": len(self._modules),
            "embeddings": len(self._embeddings),
            "elapsed_seconds": int(elapsed),
        }

        logger.info(f"Knowledge base built: {stats}")
        return stats

    async def _extract_wram_symbols(self, wram_file: Path):
        """Extract WRAM symbols from wram.asm."""
        logger.info(f"Extracting WRAM symbols from {wram_file.name}")

        content = wram_file.read_text(errors="ignore")
        lines = content.split("\n")

        current_description = []

        for i, line in enumerate(lines):
            # Collect comment lines as description
            if line.strip().startswith(";"):
                comment = line.strip()[1:].strip()
                if comment and not comment.startswith("=") and not comment.startswith("-"):
                    current_description.append(comment)
                continue

            # Match symbol definitions: NAME = $7EXXXX
            match = re.match(r"^(\w+)\s*=\s*\$([0-9A-Fa-f]+)", line)
            if match:
                name = match.group(1)
                address = f"${match.group(2)}"

                # Determine category from address
                addr_val = int(match.group(2), 16)
                if addr_val >= 0x7E0000:
                    addr_val -= 0x7E0000

                category = "wram"
                for (start, end), cat in self.WRAM_CATEGORIES.items():
                    if start <= addr_val <= end:
                        category = f"wram_{cat}"
                        break

                # Build description from accumulated comments
                description = " ".join(current_description[-5:]) if current_description else ""

                symbol = AsmSymbol(
                    name=name,
                    address=address,
                    category=category,
                    description=description,
                    file_path=str(wram_file),
                    line_number=i + 1,
                )

                self._symbols[symbol.id] = symbol
                current_description = []

        logger.info(f"Extracted {len([s for s in self._symbols.values() if 'wram' in s.category])} WRAM symbols")

    async def _extract_register_symbols(self, registers_file: Path):
        """Extract hardware register definitions."""
        logger.info(f"Extracting registers from {registers_file.name}")

        content = registers_file.read_text(errors="ignore")
        lines = content.split("\n")

        for i, line in enumerate(lines):
            match = re.match(r"^(\w+)\s*=\s*\$([0-9A-Fa-f]+)", line)
            if match:
                name = match.group(1)
                address = f"${match.group(2)}"

                symbol = AsmSymbol(
                    name=name,
                    address=address,
                    category="register",
                    file_path=str(registers_file),
                    line_number=i + 1,
                )

                self._symbols[symbol.id] = symbol

        logger.info(f"Extracted {len([s for s in self._symbols.values() if s.category == 'register'])} registers")

    async def _extract_routines(self, bank_file: Path):
        """Extract routine definitions from a bank file."""
        bank_name = bank_file.stem  # e.g., "bank_00"

        content = bank_file.read_text(errors="ignore")
        lines = content.split("\n")

        current_routine: Optional[dict] = None
        routine_lines: list[str] = []

        for i, line in enumerate(lines):
            # Match routine label: RoutineName:
            if match := re.match(r"^(\w+):(?:\s*;(.*))?$", line):
                # Save previous routine
                if current_routine:
                    routine = AsmRoutine(
                        name=current_routine["name"],
                        address=current_routine.get("address", ""),
                        bank=bank_name,
                        file_path=str(bank_file),
                        line_start=current_routine["line_start"],
                        line_end=i,
                        code="\n".join(routine_lines[:50]),  # First 50 lines
                        description=current_routine.get("description", ""),
                    )
                    self._routines[routine.name] = routine

                # Start new routine
                current_routine = {
                    "name": match.group(1),
                    "line_start": i + 1,
                    "description": match.group(2) if match.group(2) else "",
                }
                routine_lines = []

            # Track routine content
            elif current_routine:
                routine_lines.append(line)

                # Extract address from instruction prefix
                if addr_match := re.match(r"\s*#_([0-9A-Fa-f]+):", line):
                    if not current_routine.get("address"):
                        current_routine["address"] = f"${addr_match.group(1)}"

        # Save last routine
        if current_routine:
            routine = AsmRoutine(
                name=current_routine["name"],
                address=current_routine.get("address", ""),
                bank=bank_name,
                file_path=str(bank_file),
                line_start=current_routine["line_start"],
                line_end=len(lines),
                code="\n".join(routine_lines[:50]),
            )
            self._routines[routine.name] = routine

        logger.debug(f"Extracted routines from {bank_name}")

    async def _extract_game_modules(self, bank_00: Path):
        """Extract game module definitions from bank_00."""
        if not bank_00.exists():
            return

        content = bank_00.read_text(errors="ignore")

        # Find module definitions in the pool
        module_names = re.findall(r"db (Module\w+)>>0", content)

        for i, name in enumerate(module_names):
            # Extract module number from name
            if match := re.match(r"Module([0-9A-Fa-f]+)_(\w+)", name):
                mod_id = int(match.group(1), 16)
                mod_name = match.group(2)

                self._modules[mod_id] = GameModule(
                    id=mod_id,
                    name=name,
                    description=mod_name.replace("_", " "),
                )

        logger.info(f"Extracted {len(self._modules)} game modules")

    def _build_cross_references(self):
        """Build cross-references between symbols and routines."""
        logger.info("Building cross-references...")

        for routine in self._routines.values():
            # Find JSR/JSL calls
            for match in re.finditer(r"(?:JSR|JSL)\.?\w?\s+(\w+)", routine.code):
                called = match.group(1)
                if called not in routine.calls:
                    routine.calls.append(called)

                # Update called routine's called_by
                if called in self._routines:
                    if routine.name not in self._routines[called].called_by:
                        self._routines[called].called_by.append(routine.name)

            # Find memory accesses
            for match in re.finditer(r"(?:LDA|STA|STZ|INC|DEC)\.?\w?\s+(\$[0-9A-Fa-f]+|\w+)", routine.code):
                addr = match.group(1)
                if addr not in routine.memory_access:
                    routine.memory_access.append(addr)

        logger.info("Cross-references built")

    async def _generate_all_embeddings(self):
        """Generate embeddings for all symbols and routines."""
        logger.info("Generating embeddings...")

        if not self._orchestrator or not self._embedding_manager:
            logger.warning("No orchestrator available for embeddings")
            return

        symbol_items = []
        for symbol in self._symbols.values():
            text = f"{symbol.name}: {symbol.description}" if symbol.description else symbol.name
            symbol_items.append((symbol.id, text))

        await self._embedding_manager.generate_embeddings(
            symbol_items,
            kb_name="alttp_symbols",
        )

        # Generate for key routines
        routine_items = []
        for routine in list(self._routines.values())[:100]:  # Top 100
            text = f"{routine.name}: {routine.description}" if routine.description else routine.name
            routine_items.append((f"routine:{routine.name}", text))

        if routine_items:
            await self._embedding_manager.generate_embeddings(
                routine_items,
                kb_name="alttp_routines",
            )

        self._load_embeddings()

        logger.info(f"Generated {len(self._embeddings)} embeddings")

    async def _perform_deep_analysis(self):
        """Use Gemini for deep semantic analysis of key structures."""
        logger.info("Performing deep analysis with Gemini...")

        if not self._orchestrator:
            return

        # Analyze game modules
        modules_text = "\n".join([
            f"Module {m.id:02X}: {m.name} - {m.description}"
            for m in self._modules.values()
        ])

        try:
            result = await self._orchestrator.generate(
                prompt=f"""Analyze these ALTTP game modules and explain the game state machine:

{modules_text}

Explain:
1. What each module represents in the game flow
2. How modules transition between each other
3. Key modules for gameplay vs menus vs cutscenes
4. The significance of the module numbering""",
                tier=TaskTier.RESEARCH,
                provider=Provider.GEMINI,
            )

            if result.content:
                # Save analysis
                analysis_file = self.kb_dir / "module_analysis.md"
                analysis_file.write_text(f"# ALTTP Game Module Analysis\n\n{result.content}")
                logger.info("Game module analysis complete")

        except Exception as e:
            logger.warning(f"Module analysis failed: {e}")

        # Analyze WRAM layout
        wram_symbols = [s for s in self._symbols.values() if "wram" in s.category][:50]
        wram_text = "\n".join([
            f"{s.name} = {s.address}: {s.description}"
            for s in wram_symbols
        ])

        try:
            result = await self._orchestrator.generate(
                prompt=f"""Analyze these ALTTP WRAM (Work RAM) symbols and explain the memory layout:

{wram_text}

Explain:
1. How Link's state is represented in memory
2. Important game state variables
3. The relationship between related variables
4. Memory regions for different subsystems""",
                tier=TaskTier.RESEARCH,
                provider=Provider.GEMINI,
            )

            if result.content:
                analysis_file = self.kb_dir / "wram_analysis.md"
                analysis_file.write_text(f"# ALTTP WRAM Analysis\n\n{result.content}")
                logger.info("WRAM analysis complete")

        except Exception as e:
            logger.warning(f"WRAM analysis failed: {e}")

    async def search(
        self,
        query: str,
        limit: int = 10,
        category_filter: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Semantic search over the knowledge base.

        Args:
            query: Search query.
            limit: Maximum results.
            category_filter: Optional category to filter by.

        Returns:
            List of matching results with scores.
        """
        if not self._orchestrator:
            await self.setup()

        # Generate query embedding
        try:
            query_embedding = await self._orchestrator.embed(
                query,
                provider=self._embedding_provider,
                model=self._embedding_model,
            )
            if not query_embedding:
                return []
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return []

        # Search embeddings
        results = []

        for emb_id, embedding in self._embeddings.items():
            # Apply category filter
            if category_filter:
                if category_filter not in emb_id:
                    continue

            # Calculate similarity
            score = self._cosine_similarity(query_embedding, embedding)

            # Get the actual item
            if emb_id.startswith("routine:"):
                name = emb_id[8:]
                if name in self._routines:
                    item = self._routines[name]
                    results.append({
                        "type": "routine",
                        "name": item.name,
                        "address": item.address,
                        "bank": item.bank,
                        "description": item.description,
                        "score": score,
                    })
            elif ":" in emb_id:
                if emb_id in self._symbols:
                    item = self._symbols[emb_id]
                    results.append({
                        "type": "symbol",
                        "name": item.name,
                        "address": item.address,
                        "category": item.category,
                        "description": item.description,
                        "score": score,
                    })

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity."""
        import math

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    async def analyze_routine(self, routine_name: str) -> str:
        """Deep analysis of a specific routine using Gemini.

        Args:
            routine_name: Name of routine to analyze.

        Returns:
            Detailed analysis.
        """
        if routine_name not in self._routines:
            return f"Routine '{routine_name}' not found"

        routine = self._routines[routine_name]

        if not self._orchestrator:
            await self.setup()

        # Get related symbols
        related_symbols = []
        for addr in routine.memory_access[:10]:
            for sym in self._symbols.values():
                if sym.address == addr or sym.name == addr:
                    related_symbols.append(f"{sym.name} ({sym.address}): {sym.description}")
                    break

        prompt = f"""Analyze this 65816 assembly routine from A Link to the Past:

ROUTINE: {routine.name}
ADDRESS: {routine.address}
BANK: {routine.bank}

CODE:
{routine.code}

CALLS: {', '.join(routine.calls[:10])}
CALLED BY: {', '.join(routine.called_by[:10])}

RELATED MEMORY:
{chr(10).join(related_symbols)}

Provide:
1. Plain English explanation of what this routine does
2. Its role in the game (graphics, gameplay, AI, etc.)
3. Key algorithms or patterns used
4. Important side effects (memory writes, state changes)
5. How it fits into the larger game loop"""

        try:
            result = await self._orchestrator.generate(
                prompt=prompt,
                tier=TaskTier.REASONING,
                provider=Provider.GEMINI,
            )
            return result.content
        except Exception as e:
            return f"Analysis failed: {e}"

    async def explain_symbol(self, symbol_name: str) -> str:
        """Get detailed explanation of a symbol.

        Args:
            symbol_name: Symbol name to explain.

        Returns:
            Detailed explanation.
        """
        # Find symbol
        symbol = None
        for s in self._symbols.values():
            if s.name == symbol_name:
                symbol = s
                break

        if not symbol:
            return f"Symbol '{symbol_name}' not found"

        if not self._orchestrator:
            await self.setup()

        prompt = f"""Explain this ALTTP memory address:

NAME: {symbol.name}
ADDRESS: {symbol.address}
CATEGORY: {symbol.category}
DESCRIPTION: {symbol.description}

For this SNES/65816 memory address, explain:
1. What game state or data it represents
2. How it's typically used in the game
3. Valid values and their meanings
4. Related symbols or memory regions
5. Common patterns for reading/writing this address"""

        try:
            result = await self._orchestrator.generate(
                prompt=prompt,
                tier=TaskTier.FAST,
                provider=Provider.GEMINI,
            )
            return result.content
        except Exception as e:
            return f"Explanation failed: {e}"

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the knowledge base."""
        return {
            "total_symbols": len(self._symbols),
            "total_routines": len(self._routines),
            "total_modules": len(self._modules),
            "embeddings_count": len(self._embeddings),
        }
