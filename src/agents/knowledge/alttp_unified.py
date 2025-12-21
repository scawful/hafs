"""ALTTP Unified Knowledge Base System.

Provides:
- Batch embeddings with checkpointing (resume interrupted builds)
- Oracle-of-Secrets ROM hack knowledge base
- Cross-referencing between hack and vanilla sources
- Unified search across all knowledge bases
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from agents.core import BaseAgent
from hafs.core.embeddings import BatchEmbeddingManager
from hafs.core.orchestrator_v2 import UnifiedOrchestrator, TaskTier, Provider

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class HackModification:
    """Tracks a modification in the ROM hack vs vanilla."""

    address: str
    hack_symbol: str
    vanilla_symbol: Optional[str]
    modification_type: str  # new, override, extend, hook
    file_path: str
    description: str = ""
    bank: Optional[str] = None


@dataclass
class UnifiedSearchResult:
    """Search result from unified search."""

    kb_name: str
    item_type: str  # symbol, routine, modification
    name: str
    address: str
    description: str
    score: float
    is_hack_modification: bool = False
    vanilla_equivalent: Optional[str] = None


# ============================================================================
# Oracle of Secrets Knowledge Base
# ============================================================================

class OracleOfSecretsKB(BaseAgent):
    """Knowledge base for Oracle-of-Secrets ROM hack.

    Tracks:
    - Custom symbols and WRAM usage
    - Modified routines and hooks
    - New content (sprites, items, dungeons)
    - Bank allocations
    - Cross-references to vanilla code

    Example:
        kb = OracleOfSecretsKB()
        await kb.setup()
        await kb.build()

        # Find what's modified
        mods = await kb.get_modifications()

        # Compare with vanilla
        comparison = await kb.compare_to_vanilla("SprY")
    """

    SOURCE_PATH = Path.home() / "Code" / "Oracle-of-Secrets"

    # Bank allocations from Oracle_main.asm
    BANK_ALLOCATIONS = {
        0x20: "Expanded Music",
        0x28: "ZSCustomOverworld",
        0x2B: "Items",
        0x2C: "Dungeons",
        0x2D: "Menu",
        0x2E: "HUD",
        0x2F: "Expanded Messages",
        0x30: "Sprites",
        0x31: "Sprites",
        0x32: "Sprites",
        0x33: "Moosh Form",
        0x34: "Time System/Overlays",
        0x35: "Deku Link",
        0x36: "Zora Link",
        0x37: "Bunny Link",
        0x38: "Wolf Link",
        0x39: "Minish Link",
        0x3A: "Mask Routines",
        0x3B: "GBC Link",
        0x40: "LW World Map",
        0x41: "DW World Map",
    }

    def __init__(
        self,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        super().__init__(
            "OracleOfSecretsKB",
            "Knowledge base for Oracle-of-Secrets ROM hack."
        )

        self.kb_dir = self.context_root / "knowledge" / "oracle-of-secrets"
        self.kb_dir.mkdir(parents=True, exist_ok=True)

        # Storage files
        self.symbols_file = self.kb_dir / "symbols.json"
        self.routines_file = self.kb_dir / "routines.json"
        self.modifications_file = self.kb_dir / "modifications.json"
        self.banks_file = self.kb_dir / "banks.json"

        # In-memory data
        self._symbols: dict[str, dict] = {}
        self._routines: dict[str, dict] = {}
        self._modifications: list[HackModification] = []
        self._bank_contents: dict[int, list[str]] = {}

        # Orchestrator and embedding manager
        self._orchestrator: Optional[UnifiedOrchestrator] = None
        self._embedding_manager: Optional[BatchEmbeddingManager] = None
        self._embedding_provider = embedding_provider
        self._embedding_model = embedding_model

    async def setup(self):
        """Initialize the KB."""
        await super().setup()

        self._orchestrator = UnifiedOrchestrator()
        await self._orchestrator.initialize()

        self._embedding_manager = BatchEmbeddingManager(
            self.kb_dir,
            self._orchestrator,
            batch_size=50,
            embedding_provider=self._embedding_provider,
            embedding_model=self._embedding_model,
        )

        self._load_data()
        logger.info(f"OracleOfSecretsKB ready. {len(self._symbols)} symbols, {len(self._routines)} routines")

    def _load_data(self):
        """Load existing data from disk."""
        if self.symbols_file.exists():
            try:
                data = json.loads(self.symbols_file.read_text())
                if isinstance(data, list):
                    self._symbols = {item.get("name", f"Symbol_{i}"): item for i, item in enumerate(data)}
                else:
                    self._symbols = data
            except:
                pass

        if self.routines_file.exists():
            try:
                data = json.loads(self.routines_file.read_text())
                if isinstance(data, list):
                    # Convert list to dict
                    self._routines = {r.get("name", f"Routine_{i}"): r for i, r in enumerate(data)}
                else:
                    self._routines = data
            except:
                pass

        if self.modifications_file.exists():
            try:
                data = json.loads(self.modifications_file.read_text())
                self._modifications = [HackModification(**m) for m in data]
            except:
                pass

    def _save_data(self):
        """Save data to disk."""
        self.symbols_file.write_text(json.dumps(self._symbols, indent=2))
        self.routines_file.write_text(json.dumps(self._routines, indent=2))
        self.modifications_file.write_text(json.dumps(
            [asdict(m) for m in self._modifications], indent=2
        ))

    async def build(
        self,
        generate_embeddings: bool = True,
    ) -> dict[str, int]:
        """Build the knowledge base from source.

        Args:
            generate_embeddings: Whether to generate embeddings.

        Returns:
            Build statistics.
        """
        if not self.SOURCE_PATH.exists():
            raise ValueError(f"Oracle-of-Secrets not found at {self.SOURCE_PATH}")

        logger.info(f"Building Oracle-of-Secrets KB from {self.SOURCE_PATH}")

        # Extract from key files
        await self._extract_symbols(self.SOURCE_PATH / "Core" / "symbols.asm")
        await self._extract_symbols(self.SOURCE_PATH / "Core" / "sram.asm")
        await self._extract_patches(self.SOURCE_PATH / "Core" / "patches.asm")

        # Extract from content directories
        for subdir in ["Items", "Masks", "Dungeons", "Overworld", "Menu", "Sprites", "Music"]:
            content_dir = self.SOURCE_PATH / subdir
            if content_dir.exists():
                for asm_file in content_dir.rglob("*.asm"):
                    await self._extract_content_file(asm_file, subdir)

        # Identify modifications (hooks into vanilla code)
        await self._identify_modifications()

        self._save_data()

        # Generate embeddings
        if generate_embeddings and self._embedding_manager:
            items = []
            for sym_id, sym in self._symbols.items():
                text = f"{sym.get('name', sym_id)}: {sym.get('description', '')}"
                items.append((f"symbol:{sym_id}", text))

            for routine_name, routine in self._routines.items():
                text = f"{routine_name}: {routine.get('description', '')}"
                items.append((f"routine:{routine_name}", text))

            if items:
                await self._embedding_manager.generate_embeddings(
                    items, "oracle-of-secrets"
                )

        return {
            "symbols": len(self._symbols),
            "routines": len(self._routines),
            "modifications": len(self._modifications),
        }

    async def _extract_symbols(self, file_path: Path):
        """Extract symbols from an ASM file."""
        if not file_path.exists():
            return

        try:
            content = file_path.read_text(errors="ignore")
            lines = content.split("\n")

            current_desc = []

            for i, line in enumerate(lines):
                # Collect comments
                if line.strip().startswith(";"):
                    comment = line.strip()[1:].strip()
                    if comment and not comment.startswith("="):
                        current_desc.append(comment)
                    continue

                # Match skip definitions: SymbolName: skip N
                if match := re.match(r"(\w+):\s*skip\s+(\d+)", line):
                    name = match.group(1)
                    size = int(match.group(2))

                    self._symbols[name] = {
                        "name": name,
                        "size": size,
                        "category": "wram_custom",
                        "file": str(file_path),
                        "line": i + 1,
                        "description": " ".join(current_desc[-3:]),
                    }
                    current_desc = []

                # Match EQU definitions: SymbolName = $XXXX
                elif match := re.match(r"(\w+)\s*=\s*\$([0-9A-Fa-f]+)", line):
                    name = match.group(1)
                    address = f"${match.group(2)}"

                    self._symbols[name] = {
                        "name": name,
                        "address": address,
                        "category": "constant",
                        "file": str(file_path),
                        "line": i + 1,
                        "description": " ".join(current_desc[-3:]),
                    }
                    current_desc = []

        except Exception as e:
            logger.warning(f"Failed to extract symbols from {file_path}: {e}")

    async def _extract_patches(self, patches_file: Path):
        """Extract patches/hooks from patches.asm."""
        if not patches_file.exists():
            return

        try:
            content = patches_file.read_text(errors="ignore")
            lines = content.split("\n")

            current_address = None

            for i, line in enumerate(lines):
                # Match org directives: org $XXXXXX
                if match := re.match(r"\s*org\s+\$([0-9A-Fa-f]+)", line):
                    current_address = f"${match.group(1)}"

                # Match routine labels
                elif match := re.match(r"^(\w+):", line):
                    name = match.group(1)

                    self._routines[name] = {
                        "name": name,
                        "address": current_address or "",
                        "category": "patch",
                        "file": str(patches_file),
                        "line": i + 1,
                    }

                    # Check if this is a hook (JSL/JSR redirect)
                    if current_address:
                        self._modifications.append(HackModification(
                            address=current_address,
                            hack_symbol=name,
                            vanilla_symbol=None,  # Will be resolved later
                            modification_type="hook",
                            file_path=str(patches_file),
                        ))

        except Exception as e:
            logger.warning(f"Failed to extract patches: {e}")

    async def _extract_content_file(self, file_path: Path, category: str):
        """Extract routines from a content file."""
        try:
            content = file_path.read_text(errors="ignore")
            lines = content.split("\n")

            for i, line in enumerate(lines):
                # Match routine labels
                if match := re.match(r"^(\w+):", line):
                    name = match.group(1)

                    # Skip if already extracted
                    if name in self._routines:
                        continue

                    self._routines[name] = {
                        "name": name,
                        "category": category.lower(),
                        "file": str(file_path),
                        "line": i + 1,
                    }

        except Exception as e:
            logger.debug(f"Failed to extract from {file_path}: {e}")

    async def _identify_modifications(self):
        """Identify modifications that override/hook vanilla code."""
        # Look for pushpc/pullpc patterns which indicate vanilla modifications
        patches_file = self.SOURCE_PATH / "Core" / "patches.asm"

        if patches_file.exists():
            content = patches_file.read_text(errors="ignore")

            # Find org $XX:XXXX patterns that target vanilla ROM space
            for match in re.finditer(r"org\s+\$([0-9A-Fa-f]{6})", content):
                address = f"${match.group(1)}"
                addr_val = int(match.group(1), 16)

                # Vanilla ROM is typically banks $00-$1F
                if addr_val < 0x200000:
                    self._modifications.append(HackModification(
                        address=address,
                        hack_symbol="(patch)",
                        vanilla_symbol=None,
                        modification_type="override",
                        file_path=str(patches_file),
                    ))

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search the Oracle-of-Secrets knowledge base.

        Args:
            query: Search query.
            limit: Max results.

        Returns:
            Search results.
        """
        if not self._embedding_manager:
            return []

        # Get query embedding
        try:
            query_embedding = await self._orchestrator.embed(
                query,
                provider=self._embedding_provider,
                model=self._embedding_model,
            )
            if not query_embedding:
                return []
        except:
            return []

        results = []

        # Search symbols
        for sym_id, sym in self._symbols.items():
            emb = self._embedding_manager.get_embedding(f"symbol:{sym_id}")
            if emb:
                score = self._cosine_similarity(query_embedding, emb)
                results.append({
                    "type": "symbol",
                    "name": sym.get("name", sym_id),
                    "address": sym.get("address", ""),
                    "category": sym.get("category", ""),
                    "description": sym.get("description", ""),
                    "score": score,
                })

        # Search routines
        for routine_name, routine in self._routines.items():
            emb = self._embedding_manager.get_embedding(f"routine:{routine_name}")
            if emb:
                score = self._cosine_similarity(query_embedding, emb)
                results.append({
                    "type": "routine",
                    "name": routine_name,
                    "address": routine.get("address", ""),
                    "category": routine.get("category", ""),
                    "score": score,
                })

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

    def get_modifications(self) -> list[HackModification]:
        """Get all tracked modifications."""
        return self._modifications

    def get_bank_usage(self) -> dict[str, Any]:
        """Get bank allocation information."""
        return {
            "allocations": self.BANK_ALLOCATIONS,
            "custom_symbols_by_bank": {},  # Could be enhanced
        }

    async def run_task(self, task: str = "stats") -> dict[str, Any]:
        """Run a KB task."""
        if task == "build":
            return await self.build()

        elif task == "stats":
            return {
                "symbols": len(self._symbols),
                "routines": len(self._routines),
                "modifications": len(self._modifications),
            }

        elif task.startswith("search:"):
            query = task[7:].strip()
            return {"results": await self.search(query)}

        return {"error": "Unknown task"}


# ============================================================================
# Unified Knowledge Base System
# ============================================================================

class UnifiedALTTPKnowledge(BaseAgent):
    """Unified search across all ALTTP knowledge bases.

    Provides:
    - Unified search across vanilla (usdasm) and hack (Oracle-of-Secrets)
    - Cross-referencing between sources
    - Modification tracking
    - Comparative analysis

    Example:
        unified = UnifiedALTTPKnowledge()
        await unified.setup()

        # Unified search
        results = await unified.search("sprite animation")

        # Compare symbol
        comparison = await unified.compare("SprY")

        # Get hack modifications
        mods = await unified.get_hack_modifications()
    """

    def __init__(self):
        super().__init__(
            "UnifiedALTTPKnowledge",
            "Unified search and cross-referencing across all ALTTP knowledge bases."
        )

        self._vanilla_kb = None  # ALTTPKnowledgeBase
        self._hack_kb = None     # OracleOfSecretsKB
        self._orchestrator = None

        self.unified_dir = self.context_root / "knowledge" / "alttp_unified"
        self.unified_dir.mkdir(parents=True, exist_ok=True)

        self.cross_refs_file = self.unified_dir / "cross_references.json"
        self._cross_refs: list[dict] = []

    async def setup(self):
        """Initialize the unified knowledge system."""
        print("UnifiedALTTPKnowledge: setup started...", flush=True)
        await super().setup()
        print("UnifiedALTTPKnowledge: BaseAgent setup done.", flush=True)

        self._orchestrator = UnifiedOrchestrator()
        print("UnifiedALTTPKnowledge: Initializing Orchestrator...", flush=True)
        await self._orchestrator.initialize()
        print("UnifiedALTTPKnowledge: Orchestrator initialized.", flush=True)

        # Initialize vanilla KB
        from agents.knowledge.alttp import ALTTPKnowledgeBase
        self._vanilla_kb = ALTTPKnowledgeBase()
        print("UnifiedALTTPKnowledge: Setting up Vanilla KB...", flush=True)
        await self._vanilla_kb.setup()

        # Initialize hack KB
        self._hack_kb = OracleOfSecretsKB()
        print("UnifiedALTTPKnowledge: Setting up Hack KB...", flush=True)
        await self._hack_kb.setup()

        self._load_cross_refs()

        logger.info("UnifiedALTTPKnowledge ready")

    def _load_cross_refs(self):
        """Load cross-references from disk."""
        if self.cross_refs_file.exists():
            try:
                self._cross_refs = json.loads(self.cross_refs_file.read_text())
            except:
                pass

    def _save_cross_refs(self):
        """Save cross-references to disk."""
        self.cross_refs_file.write_text(json.dumps(self._cross_refs, indent=2))

    async def build_all(
        self,
        generate_embeddings: bool = True,
    ) -> dict[str, Any]:
        """Build all knowledge bases.

        Args:
            generate_embeddings: Whether to generate embeddings.

        Returns:
            Combined statistics.
        """
        stats = {}

        # Build vanilla KB
        logger.info("Building vanilla (usdasm) KB...")
        vanilla_stats = await self._vanilla_kb.build_from_source(
            generate_embeddings=generate_embeddings,
            deep_analysis=False,
        )
        stats["vanilla"] = vanilla_stats

        # Build hack KB
        logger.info("Building Oracle-of-Secrets KB...")
        hack_stats = await self._hack_kb.build(
            generate_embeddings=generate_embeddings,
        )
        stats["hack"] = hack_stats

        # Build cross-references
        await self._build_cross_references()
        stats["cross_refs"] = len(self._cross_refs)

        return stats

    async def _build_cross_references(self):
        """Build cross-references between vanilla and hack."""
        logger.info("Building cross-references...")

        self._cross_refs = []

        # Match by symbol name
        hack_symbols = set(self._hack_kb._symbols.keys())
        vanilla_symbols = {s.name: s for s in self._vanilla_kb._symbols.values()}

        for hack_name in hack_symbols:
            if hack_name in vanilla_symbols:
                vanilla = vanilla_symbols[hack_name]
                hack = self._hack_kb._symbols[hack_name]

                self._cross_refs.append({
                    "hack_symbol": hack_name,
                    "vanilla_symbol": hack_name,
                    "match_type": "name_exact",
                    "hack_address": hack.get("address", ""),
                    "vanilla_address": vanilla.address,
                    "is_override": hack.get("address") != vanilla.address,
                })

        # Match by address
        hack_addrs = {
            s.get("address"): name
            for name, s in self._hack_kb._symbols.items()
            if s.get("address")
        }
        vanilla_addrs = {s.address: s.name for s in self._vanilla_kb._symbols.values()}

        for addr, hack_name in hack_addrs.items():
            if addr in vanilla_addrs:
                vanilla_name = vanilla_addrs[addr]
                if hack_name != vanilla_name:
                    self._cross_refs.append({
                        "hack_symbol": hack_name,
                        "vanilla_symbol": vanilla_name,
                        "address": addr,
                        "match_type": "address",
                    })

        self._save_cross_refs()
        logger.info(f"Built {len(self._cross_refs)} cross-references")

    async def search(
        self,
        query: str,
        limit: int = 20,
        sources: Optional[list[str]] = None,
    ) -> list[UnifiedSearchResult]:
        """Unified search across all knowledge bases.

        Args:
            query: Search query.
            limit: Max total results.
            sources: Optional list of sources to search ("vanilla", "hack").

        Returns:
            Unified search results.
        """
        sources = sources or ["vanilla", "hack"]
        all_results = []

        # Search vanilla
        if "vanilla" in sources and self._vanilla_kb:
            try:
                vanilla_results = await self._vanilla_kb.search(query, limit=limit)
                for r in vanilla_results:
                    # Check if there's a hack modification
                    is_modified = any(
                        xr["vanilla_symbol"] == r["name"]
                        for xr in self._cross_refs
                    )

                    all_results.append(UnifiedSearchResult(
                        kb_name="vanilla",
                        item_type=r.get("type", "symbol"),
                        name=r["name"],
                        address=r.get("address", ""),
                        description=r.get("description", ""),
                        score=r.get("score", 0.0),
                        is_hack_modification=is_modified,
                    ))
            except Exception as e:
                logger.warning(f"Vanilla search failed: {e}")

        # Search hack
        if "hack" in sources and self._hack_kb:
            try:
                hack_results = await self._hack_kb.search(query, limit=limit)
                for r in hack_results:
                    # Find vanilla equivalent
                    vanilla_eq = None
                    for xr in self._cross_refs:
                        if xr["hack_symbol"] == r["name"]:
                            vanilla_eq = xr.get("vanilla_symbol")
                            break

                    all_results.append(UnifiedSearchResult(
                        kb_name="hack",
                        item_type=r.get("type", "symbol"),
                        name=r["name"],
                        address=r.get("address", ""),
                        description=r.get("description", ""),
                        score=r.get("score", 0.0),
                        is_hack_modification=True,
                        vanilla_equivalent=vanilla_eq,
                    ))
            except Exception as e:
                logger.warning(f"Hack search failed: {e}")

        # Sort by score and deduplicate
        all_results.sort(key=lambda x: x.score, reverse=True)

        # Deduplicate by name (prefer hack version if exists)
        seen = set()
        deduped = []
        for r in all_results:
            if r.name not in seen:
                seen.add(r.name)
                deduped.append(r)

        return deduped[:limit]

    async def compare(self, symbol_name: str) -> dict[str, Any]:
        """Compare a symbol between vanilla and hack.

        Args:
            symbol_name: Symbol name to compare.

        Returns:
            Comparison data.
        """
        result = {
            "symbol": symbol_name,
            "vanilla": None,
            "hack": None,
            "cross_refs": [],
            "is_modified": False,
        }

        # Find in vanilla
        for sym in self._vanilla_kb._symbols.values():
            if sym.name.lower() == symbol_name.lower():
                result["vanilla"] = {
                    "name": sym.name,
                    "address": sym.address,
                    "category": sym.category,
                    "description": sym.description,
                }
                break

        # Find in hack
        for name, sym in self._hack_kb._symbols.items():
            if name.lower() == symbol_name.lower():
                result["hack"] = {
                    "name": name,
                    "address": sym.get("address", ""),
                    "category": sym.get("category", ""),
                    "description": sym.get("description", ""),
                }
                break

        # Find cross-references
        for xr in self._cross_refs:
            if (
                xr.get("hack_symbol", "").lower() == symbol_name.lower()
                or xr.get("vanilla_symbol", "").lower() == symbol_name.lower()
            ):
                result["cross_refs"].append(xr)

        result["is_modified"] = len(result["cross_refs"]) > 0 or (
            result["vanilla"] and result["hack"]
        )

        return result

    async def get_hack_modifications(self) -> list[dict[str, Any]]:
        """Get all hack modifications with vanilla context.

        Returns:
            List of modifications with context.
        """
        mods = self._hack_kb.get_modifications()

        enriched = []
        for mod in mods:
            # Try to find vanilla symbol at same address
            vanilla_sym = None
            for sym in self._vanilla_kb._symbols.values():
                if sym.address == mod.address:
                    vanilla_sym = sym.name
                    break

            enriched.append({
                "address": mod.address,
                "hack_symbol": mod.hack_symbol,
                "vanilla_symbol": vanilla_sym or mod.vanilla_symbol,
                "type": mod.modification_type,
                "file": mod.file_path,
                "description": mod.description,
            })

        return enriched

    def get_statistics(self) -> dict[str, Any]:
        """Get unified statistics."""
        return {
            "vanilla": self._vanilla_kb.get_statistics() if self._vanilla_kb else {},
            "hack": {
                "symbols": len(self._hack_kb._symbols),
                "routines": len(self._hack_kb._routines),
                "modifications": len(self._hack_kb._modifications),
            } if self._hack_kb else {},
            "cross_references": len(self._cross_refs),
        }
