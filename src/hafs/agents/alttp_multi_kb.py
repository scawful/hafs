"""ALTTP Multi-Knowledge-Base Manager.

Manages multiple separate knowledge bases for different ALTTP sources:
- usdasm: US version disassembly (Kan's work)
- jpdasm: Japanese version disassembly
- gigaleak: Original Nintendo source code
- mathonnapkins: Readable refactor version
- jaredbrian: Another readable version

Enables cross-referencing between KBs without symbol duplication.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from hafs.agents.base import BaseAgent
from hafs.agents.alttp_knowledge import ALTTPKnowledgeBase, AsmSymbol, AsmRoutine
from hafs.core.orchestrator_v2 import UnifiedOrchestrator, TaskTier, Provider

logger = logging.getLogger(__name__)


class KnowledgeBaseType(Enum):
    """Types of ALTTP knowledge bases."""

    USDASM = "usdasm"           # US version disassembly (Kan)
    JPDASM = "jpdasm"           # JP version disassembly
    GIGALEAK = "gigaleak"       # Original Nintendo source
    MATHONNAPKINS = "mon"       # MathOnNapkins readable version
    JAREDBRIAN = "jb"          # JaredBrian version


@dataclass
class CrossReference:
    """Cross-reference between symbols in different KBs."""

    source_kb: str
    source_symbol: str
    target_kb: str
    target_symbol: str
    confidence: float = 1.0
    match_type: str = "exact"  # exact, address, semantic
    notes: str = ""


@dataclass
class UnifiedSymbol:
    """A symbol with references across multiple KBs."""

    canonical_name: str
    address: str
    category: str
    description: str

    # References in each KB
    kb_refs: dict[str, str] = field(default_factory=dict)  # kb_name -> symbol_name

    # Metadata
    tags: list[str] = field(default_factory=list)
    embedding_id: Optional[str] = None


class ALTTPMultiKBManager(BaseAgent):
    """Manager for multiple ALTTP knowledge bases.

    Maintains separate, clean knowledge bases for each source while
    enabling cross-referencing and unified search.

    Example:
        manager = ALTTPMultiKBManager()
        await manager.setup()

        # Build specific KB
        await manager.build_kb(KnowledgeBaseType.USDASM)
        await manager.build_kb(KnowledgeBaseType.GIGALEAK)

        # Cross-reference
        await manager.build_cross_references()

        # Unified search across all KBs
        results = await manager.search_all("Link's position")

        # Compare symbol across KBs
        comparison = await manager.compare_symbol("POSX")
    """

    # Source paths for each KB type
    SOURCE_PATHS = {
        KnowledgeBaseType.USDASM: Path.home() / "Code" / "usdasm",
        KnowledgeBaseType.JPDASM: Path.home() / "Code" / "alttp-gigaleak" / "DISASM" / "jpdasm",
        KnowledgeBaseType.GIGALEAK: Path.home() / "Code" / "alttp-gigaleak" / "1. ゼルダの伝説神々のトライフォース" / "日本_Ver3" / "asm",
    }

    def __init__(self):
        super().__init__(
            "ALTTPMultiKBManager",
            "Manage multiple ALTTP knowledge bases with cross-referencing."
        )

        # Storage
        self.kb_root = self.context_root / "knowledge" / "alttp"
        self.kb_root.mkdir(parents=True, exist_ok=True)

        # Individual knowledge bases
        self._kbs: dict[KnowledgeBaseType, ALTTPKnowledgeBase] = {}

        # Cross-references
        self._cross_refs: list[CrossReference] = []
        self._unified_symbols: dict[str, UnifiedSymbol] = {}

        self.cross_refs_file = self.kb_root / "cross_references.json"
        self.unified_file = self.kb_root / "unified_symbols.json"

        # Orchestrator
        self._orchestrator: Optional[UnifiedOrchestrator] = None

    async def setup(self):
        """Initialize the multi-KB manager."""
        await super().setup()

        self._orchestrator = UnifiedOrchestrator()
        await self._orchestrator.initialize()

        self._load_cross_refs()

        logger.info("ALTTPMultiKBManager ready")

    def _load_cross_refs(self):
        """Load cross-references from disk."""
        if self.cross_refs_file.exists():
            try:
                data = json.loads(self.cross_refs_file.read_text())
                self._cross_refs = [
                    CrossReference(**ref) for ref in data.get("refs", [])
                ]
            except Exception as e:
                logger.warning(f"Failed to load cross-refs: {e}")

        if self.unified_file.exists():
            try:
                data = json.loads(self.unified_file.read_text())
                for item in data:
                    sym = UnifiedSymbol(**item)
                    self._unified_symbols[sym.canonical_name] = sym
            except Exception as e:
                logger.warning(f"Failed to load unified symbols: {e}")

    def _save_cross_refs(self):
        """Save cross-references to disk."""
        try:
            data = {
                "refs": [
                    {
                        "source_kb": r.source_kb,
                        "source_symbol": r.source_symbol,
                        "target_kb": r.target_kb,
                        "target_symbol": r.target_symbol,
                        "confidence": r.confidence,
                        "match_type": r.match_type,
                        "notes": r.notes,
                    }
                    for r in self._cross_refs
                ],
                "updated": datetime.now().isoformat(),
            }
            self.cross_refs_file.write_text(json.dumps(data, indent=2))

            unified_data = [
                {
                    "canonical_name": s.canonical_name,
                    "address": s.address,
                    "category": s.category,
                    "description": s.description,
                    "kb_refs": s.kb_refs,
                    "tags": s.tags,
                }
                for s in self._unified_symbols.values()
            ]
            self.unified_file.write_text(json.dumps(unified_data, indent=2))

        except Exception as e:
            logger.error(f"Failed to save cross-refs: {e}")

    def _get_kb_path(self, kb_type: KnowledgeBaseType) -> Path:
        """Get storage path for a KB."""
        return self.kb_root / kb_type.value

    async def get_kb(self, kb_type: KnowledgeBaseType) -> ALTTPKnowledgeBase:
        """Get or create a knowledge base.

        Args:
            kb_type: Type of KB to get.

        Returns:
            The knowledge base instance.
        """
        if kb_type not in self._kbs:
            kb_path = self._get_kb_path(kb_type)
            kb = ALTTPKnowledgeBase(source_path=self.SOURCE_PATHS.get(kb_type))

            # Override storage location
            kb.kb_dir = kb_path
            kb.kb_dir.mkdir(parents=True, exist_ok=True)
            kb.symbols_file = kb_path / "symbols.json"
            kb.routines_file = kb_path / "routines.json"
            kb.modules_file = kb_path / "modules.json"
            kb.embeddings_dir = kb_path / "embeddings"
            kb.embeddings_dir.mkdir(parents=True, exist_ok=True)

            await kb.setup()
            self._kbs[kb_type] = kb

        return self._kbs[kb_type]

    async def build_kb(
        self,
        kb_type: KnowledgeBaseType,
        generate_embeddings: bool = True,
        deep_analysis: bool = False,
    ) -> dict[str, int]:
        """Build a specific knowledge base.

        Args:
            kb_type: Type of KB to build.
            generate_embeddings: Whether to generate embeddings.
            deep_analysis: Whether to run deep LLM analysis.

        Returns:
            Build statistics.
        """
        source_path = self.SOURCE_PATHS.get(kb_type)
        if not source_path or not source_path.exists():
            raise ValueError(f"Source path not found for {kb_type}: {source_path}")

        logger.info(f"Building KB: {kb_type.value} from {source_path}")

        kb = await self.get_kb(kb_type)

        # For gigaleak, we need custom extraction
        if kb_type == KnowledgeBaseType.GIGALEAK:
            return await self._build_gigaleak_kb(kb, source_path, generate_embeddings)

        return await kb.build_from_source(
            source_path=source_path,
            generate_embeddings=generate_embeddings,
            deep_analysis=deep_analysis,
        )

    async def _build_gigaleak_kb(
        self,
        kb: ALTTPKnowledgeBase,
        source_path: Path,
        generate_embeddings: bool,
    ) -> dict[str, int]:
        """Build KB from original Nintendo gigaleak source.

        The gigaleak uses a different assembler syntax with:
        - GLB (global) and EXT (external) declarations
        - Japanese comments
        - Different label naming conventions
        """
        logger.info("Building gigaleak KB with custom extraction...")

        # Extract symbols from gigaleak ASM files
        for asm_file in source_path.glob("*.asm"):
            await self._extract_gigaleak_file(kb, asm_file)

        # Build cross-refs
        kb._build_cross_references()

        # Generate embeddings
        if generate_embeddings:
            await kb._generate_all_embeddings()

        kb._save_knowledge()

        return {
            "symbols": len(kb._symbols),
            "routines": len(kb._routines),
            "modules": len(kb._modules),
            "embeddings": len(kb._embeddings),
        }

    async def _extract_gigaleak_file(self, kb: ALTTPKnowledgeBase, asm_file: Path):
        """Extract symbols from a gigaleak ASM file."""
        try:
            # Try multiple encodings for Japanese text
            for encoding in ["shift_jis", "utf-8", "cp932", "latin-1"]:
                try:
                    content = asm_file.read_text(encoding=encoding)
                    break
                except:
                    continue
            else:
                content = asm_file.read_text(errors="ignore")

            lines = content.split("\n")
            current_routine = None

            for i, line in enumerate(lines):
                # Match GLB (global) declarations
                if match := re.match(r"\s*GLB\s+(\w+)", line):
                    name = match.group(1)
                    kb._symbols[f"global:{name}"] = AsmSymbol(
                        name=name,
                        address="",
                        category="global_export",
                        file_path=str(asm_file),
                        line_number=i + 1,
                    )

                # Match EXT (external) declarations
                elif match := re.match(r"\s*EXT\s+(\w+)", line):
                    name = match.group(1)
                    kb._symbols[f"external:{name}"] = AsmSymbol(
                        name=name,
                        address="",
                        category="external_import",
                        file_path=str(asm_file),
                        line_number=i + 1,
                    )

                # Match label definitions: LABEL EQU $ or LABEL:
                elif match := re.match(r"^(\w+)\s+EQU\s+\$", line):
                    name = match.group(1)
                    current_routine = name

                    # Get context lines
                    context = "\n".join(lines[max(0, i-2):min(len(lines), i+10)])

                    kb._routines[name] = AsmRoutine(
                        name=name,
                        address="",
                        bank=asm_file.stem,
                        file_path=str(asm_file),
                        line_start=i + 1,
                        line_end=i + 20,
                        code=context,
                    )

                # Match EQU constants
                elif match := re.match(r"^(\w+)\s+EQU\s+([0-9A-Fa-fHh]+)", line):
                    name = match.group(1)
                    value = match.group(2)

                    kb._symbols[f"constant:{name}"] = AsmSymbol(
                        name=name,
                        address=value,
                        category="constant",
                        file_path=str(asm_file),
                        line_number=i + 1,
                    )

        except Exception as e:
            logger.warning(f"Failed to extract {asm_file}: {e}")

    async def build_cross_references(self):
        """Build cross-references between all loaded KBs.

        Matches symbols by:
        1. Exact address match
        2. Similar name patterns
        3. Semantic similarity (using embeddings)
        """
        logger.info("Building cross-references between KBs...")

        kb_list = list(self._kbs.items())

        for i, (kb_type1, kb1) in enumerate(kb_list):
            for kb_type2, kb2 in kb_list[i + 1:]:
                await self._cross_ref_kbs(kb_type1, kb1, kb_type2, kb2)

        # Build unified symbol index
        await self._build_unified_index()

        self._save_cross_refs()
        logger.info(f"Built {len(self._cross_refs)} cross-references, {len(self._unified_symbols)} unified symbols")

    async def _cross_ref_kbs(
        self,
        type1: KnowledgeBaseType,
        kb1: ALTTPKnowledgeBase,
        type2: KnowledgeBaseType,
        kb2: ALTTPKnowledgeBase,
    ):
        """Cross-reference two knowledge bases."""
        # Match by address
        addr_map1 = {s.address: s for s in kb1._symbols.values() if s.address}
        addr_map2 = {s.address: s for s in kb2._symbols.values() if s.address}

        for addr, sym1 in addr_map1.items():
            if addr in addr_map2:
                sym2 = addr_map2[addr]
                self._cross_refs.append(CrossReference(
                    source_kb=type1.value,
                    source_symbol=sym1.name,
                    target_kb=type2.value,
                    target_symbol=sym2.name,
                    confidence=1.0,
                    match_type="address",
                ))

        # Match by name similarity
        import difflib

        names1 = {s.name.lower(): s for s in kb1._symbols.values()}
        names2 = {s.name.lower(): s for s in kb2._symbols.values()}

        for name1, sym1 in names1.items():
            matches = difflib.get_close_matches(name1, names2.keys(), n=1, cutoff=0.8)
            if matches:
                name2 = matches[0]
                sym2 = names2[name2]

                # Avoid duplicates
                if not any(
                    r.source_symbol == sym1.name and r.target_symbol == sym2.name
                    for r in self._cross_refs
                ):
                    self._cross_refs.append(CrossReference(
                        source_kb=type1.value,
                        source_symbol=sym1.name,
                        target_kb=type2.value,
                        target_symbol=sym2.name,
                        confidence=0.8,
                        match_type="name_similarity",
                    ))

    async def _build_unified_index(self):
        """Build unified symbol index from cross-references."""
        # Group cross-refs by connected symbols
        symbol_groups: dict[str, set[tuple[str, str]]] = {}  # canonical -> set of (kb, name)

        for ref in self._cross_refs:
            # Use address as canonical key if available
            key = None

            # Find the symbol to get address
            for kb_type, kb in self._kbs.items():
                if kb_type.value == ref.source_kb:
                    for sym in kb._symbols.values():
                        if sym.name == ref.source_symbol and sym.address:
                            key = sym.address
                            break

            if not key:
                key = ref.source_symbol.lower()

            if key not in symbol_groups:
                symbol_groups[key] = set()

            symbol_groups[key].add((ref.source_kb, ref.source_symbol))
            symbol_groups[key].add((ref.target_kb, ref.target_symbol))

        # Create unified symbols
        for key, refs in symbol_groups.items():
            # Get best description
            description = ""
            category = ""

            for kb_name, sym_name in refs:
                for kb_type, kb in self._kbs.items():
                    if kb_type.value == kb_name:
                        for sym in kb._symbols.values():
                            if sym.name == sym_name:
                                if sym.description and len(sym.description) > len(description):
                                    description = sym.description
                                if sym.category:
                                    category = sym.category
                                break

            unified = UnifiedSymbol(
                canonical_name=list(refs)[0][1],  # Use first symbol name
                address=key if key.startswith("$") else "",
                category=category,
                description=description,
                kb_refs={kb: name for kb, name in refs},
            )

            self._unified_symbols[unified.canonical_name] = unified

    async def search_all(
        self,
        query: str,
        limit: int = 20,
    ) -> dict[str, list[dict[str, Any]]]:
        """Search across all loaded knowledge bases.

        Args:
            query: Search query.
            limit: Max results per KB.

        Returns:
            Dict mapping KB names to search results.
        """
        results = {}

        for kb_type, kb in self._kbs.items():
            try:
                kb_results = await kb.search(query, limit=limit)
                results[kb_type.value] = kb_results
            except Exception as e:
                logger.warning(f"Search failed for {kb_type}: {e}")
                results[kb_type.value] = []

        return results

    async def compare_symbol(
        self,
        symbol_name: str,
    ) -> dict[str, Any]:
        """Compare a symbol across all KBs.

        Args:
            symbol_name: Symbol name to compare.

        Returns:
            Comparison data from each KB.
        """
        comparison = {
            "symbol": symbol_name,
            "found_in": {},
            "cross_refs": [],
        }

        # Find in each KB
        for kb_type, kb in self._kbs.items():
            for sym in kb._symbols.values():
                if sym.name.lower() == symbol_name.lower():
                    comparison["found_in"][kb_type.value] = {
                        "name": sym.name,
                        "address": sym.address,
                        "category": sym.category,
                        "description": sym.description,
                    }
                    break

        # Find cross-references
        for ref in self._cross_refs:
            if (
                ref.source_symbol.lower() == symbol_name.lower()
                or ref.target_symbol.lower() == symbol_name.lower()
            ):
                comparison["cross_refs"].append({
                    "source": f"{ref.source_kb}:{ref.source_symbol}",
                    "target": f"{ref.target_kb}:{ref.target_symbol}",
                    "confidence": ref.confidence,
                    "match_type": ref.match_type,
                })

        return comparison

    async def deep_research(
        self,
        topic: str,
        use_gemini: bool = True,
    ) -> str:
        """Perform deep research on a topic using all KBs.

        Args:
            topic: Research topic.
            use_gemini: Whether to use Gemini for analysis.

        Returns:
            Research analysis.
        """
        if not self._orchestrator:
            await self.setup()

        # Gather context from all KBs
        context_parts = []

        for kb_type, kb in self._kbs.items():
            # Search this KB
            results = await kb.search(topic, limit=5)

            if results:
                context_parts.append(f"\n## {kb_type.value} Knowledge Base\n")
                for r in results:
                    context_parts.append(f"- {r['name']} ({r['address']}): {r.get('description', '')[:100]}")

        context = "\n".join(context_parts)

        prompt = f"""Research the following topic across multiple ALTTP knowledge bases:

TOPIC: {topic}

CONTEXT FROM KNOWLEDGE BASES:
{context}

Provide:
1. Summary of what this topic relates to in ALTTP
2. How the different sources (usdasm, gigaleak, etc.) describe it
3. Key memory addresses and routines involved
4. Technical implementation details
5. How this knowledge could be used for ROM hacking"""

        try:
            result = await self._orchestrator.generate(
                prompt=prompt,
                tier=TaskTier.RESEARCH,
                provider=Provider.GEMINI if use_gemini else None,
            )
            return result.content
        except Exception as e:
            return f"Research failed: {e}"

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics for all KBs."""
        stats = {
            "knowledge_bases": {},
            "total_symbols": 0,
            "total_routines": 0,
            "cross_references": len(self._cross_refs),
            "unified_symbols": len(self._unified_symbols),
        }

        for kb_type, kb in self._kbs.items():
            kb_stats = kb.get_statistics()
            stats["knowledge_bases"][kb_type.value] = kb_stats
            stats["total_symbols"] += kb_stats["total_symbols"]
            stats["total_routines"] += kb_stats["total_routines"]

        return stats

    async def run_task(self, task: str = "stats") -> dict[str, Any]:
        """Run a multi-KB task.

        Args:
            task: Task to run:
                - "build:TYPE" - Build specific KB
                - "cross_ref" - Build cross-references
                - "search:QUERY" - Search all KBs
                - "compare:SYMBOL" - Compare symbol
                - "research:TOPIC" - Deep research
                - "stats" - Get statistics

        Returns:
            Task result.
        """
        if task == "stats":
            return self.get_statistics()

        elif task.startswith("build:"):
            kb_name = task[6:].strip().lower()
            try:
                kb_type = KnowledgeBaseType(kb_name)
                return await self.build_kb(kb_type)
            except ValueError:
                return {"error": f"Unknown KB type: {kb_name}"}

        elif task == "cross_ref":
            await self.build_cross_references()
            return {
                "cross_references": len(self._cross_refs),
                "unified_symbols": len(self._unified_symbols),
            }

        elif task.startswith("search:"):
            query = task[7:].strip()
            return await self.search_all(query)

        elif task.startswith("compare:"):
            symbol = task[8:].strip()
            return await self.compare_symbol(symbol)

        elif task.startswith("research:"):
            topic = task[9:].strip()
            analysis = await self.deep_research(topic)
            return {"topic": topic, "analysis": analysis}

        else:
            return {
                "error": "Unknown task",
                "usage": [
                    "build:usdasm|jpdasm|gigaleak",
                    "cross_ref",
                    "search:QUERY",
                    "compare:SYMBOL",
                    "research:TOPIC",
                    "stats",
                ],
            }


# Need to import re at module level
import re
