"""Enhanced ALTTP Embedding Builder.

Generates rich, contextual embeddings for ALTTP knowledge bases with:
- Relationship context (calls, called_by, references)
- Code pattern recognition
- Memory access patterns
- Category and bank information
- Cross-reference integration

These enhanced embeddings improve semantic search accuracy for:
- Finding related routines by functionality
- Discovering memory patterns
- Understanding code flow
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from hafs.core.embeddings import BatchEmbeddingManager
from hafs.core.orchestrator_v2 import UnifiedOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class EnrichedEmbeddingItem:
    """An item ready for enhanced embedding generation."""

    id: str
    primary_text: str
    context_text: str
    category: str
    metadata: dict[str, Any]

    @property
    def full_text(self) -> str:
        """Combine primary and context text for embedding."""
        parts = [self.primary_text]
        if self.context_text:
            parts.append(self.context_text)
        return "\n".join(parts)


class ALTTPEmbeddingBuilder:
    """Enhanced embedding generator for ALTTP knowledge bases.

    Generates embeddings with rich context including:
    - Symbol/routine relationships
    - Code patterns and memory access
    - Game state context
    - Cross-references to vanilla/hack versions
    """

    # Code pattern categories for richer embeddings
    CODE_PATTERNS = {
        "input_handling": ["CheckJoypadInput", "GetJoypad", "ButtonPress", "ButtonHold"],
        "sprite_logic": ["Sprite_", "Enemy_", "NPC_", "Prep", "Draw"],
        "dungeon": ["Dungeon_", "Room_", "Door_", "Chest_", "Underworld"],
        "overworld": ["Overworld_", "Module_", "Map_", "Entrance_"],
        "item_logic": ["Item_", "Equipment_", "Inventory_", "GetItem"],
        "animation": ["Animate", "Animation_", "Frame", "Pose"],
        "physics": ["Move", "Velocity", "Collision", "Position", "Speed"],
        "graphics": ["Draw", "Render", "Tile", "BG", "OAM", "DMA"],
        "sound": ["Sound_", "Music_", "SFX_", "Audio_"],
        "save_load": ["Save", "Load", "SRAM", "Backup"],
    }

    # WRAM region context for memory symbols
    WRAM_REGIONS = {
        (0x0000, 0x00FF): ("direct_page", "Fast access direct page variables"),
        (0x0100, 0x01FF): ("stack", "Stack area"),
        (0x0200, 0x02FF): ("oam_buffer", "OAM sprite buffer"),
        (0x0300, 0x03FF): ("link_state", "Link's state and properties"),
        (0x0400, 0x04FF): ("objects", "Interactive objects"),
        (0x0500, 0x05FF): ("sprites", "Enemy and NPC sprites"),
        (0x0C00, 0x0CFF): ("pushable", "Pushable blocks"),
        (0x0D00, 0x0DFF): ("ancillae", "Projectiles and effects"),
        (0x0E00, 0x0FFF): ("misc", "Miscellaneous game state"),
        (0x1000, 0x12FF): ("tilemap", "BG tilemap buffer"),
        (0xF300, 0xF4FF): ("sram_mirror", "SRAM mirror (save data)"),
    }

    def __init__(
        self,
        kb_dir: Path,
        orchestrator: Optional[UnifiedOrchestrator] = None,
    ):
        self.kb_dir = kb_dir
        self.embeddings_dir = kb_dir / "embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        self._orchestrator = orchestrator
        self._embedding_manager: Optional[BatchEmbeddingManager] = None

    async def setup(self):
        """Initialize the embedding builder."""
        if self._orchestrator is None:
            self._orchestrator = UnifiedOrchestrator()
        await self._orchestrator.initialize()

        self._embedding_manager = BatchEmbeddingManager(
            kb_dir=self.kb_dir,
            orchestrator=self._orchestrator,
        )

    def _get_wram_context(self, address: str) -> tuple[str, str]:
        """Get context for a WRAM address."""
        try:
            # Parse address (handle $7EXXXX or $XXXX format)
            addr_str = address.replace("$", "").replace("7E", "").replace("7F", "")
            addr = int(addr_str, 16)

            for (start, end), (category, description) in self.WRAM_REGIONS.items():
                if start <= addr <= end:
                    return category, description

            return "unknown", "Unknown WRAM region"
        except (ValueError, AttributeError):
            return "unknown", "Unknown WRAM region"

    def _get_code_pattern_category(self, name: str) -> list[str]:
        """Identify code patterns from routine name."""
        patterns = []
        for category, keywords in self.CODE_PATTERNS.items():
            for keyword in keywords:
                if keyword.lower() in name.lower():
                    patterns.append(category)
                    break
        return patterns

    def _build_relationship_context(
        self,
        calls: list[str],
        called_by: list[str],
        references: list[str],
    ) -> str:
        """Build relationship context text."""
        parts = []

        if calls:
            # Limit to most important calls
            important_calls = calls[:10]
            parts.append(f"Calls: {', '.join(important_calls)}")

        if called_by:
            important_callers = called_by[:10]
            parts.append(f"Called by: {', '.join(important_callers)}")

        if references:
            important_refs = references[:10]
            parts.append(f"References: {', '.join(important_refs)}")

        return "; ".join(parts) if parts else ""

    def _build_memory_context(self, memory_access: list[str]) -> str:
        """Build memory access context text."""
        if not memory_access:
            return ""

        # Group by access type
        regions = set()
        for addr in memory_access[:20]:
            category, _ = self._get_wram_context(addr)
            if category != "unknown":
                regions.add(category)

        if regions:
            return f"Accesses memory: {', '.join(sorted(regions))}"
        return ""

    def enrich_symbol(
        self,
        symbol_id: str,
        name: str,
        address: str,
        category: str,
        description: str = "",
        references: Optional[list[str]] = None,
        referenced_by: Optional[list[str]] = None,
        bank: Optional[str] = None,
    ) -> EnrichedEmbeddingItem:
        """Create enriched embedding item for a symbol."""
        # Primary text: name and description
        primary_parts = [name]
        if description:
            primary_parts.append(description)
        primary_text = ": ".join(primary_parts)

        # Context text: relationships and metadata
        context_parts = []

        # Add category context
        if category:
            context_parts.append(f"Type: {category}")

        # Add WRAM region context for memory symbols
        if "wram" in category.lower():
            region_cat, region_desc = self._get_wram_context(address)
            context_parts.append(f"Region: {region_desc} ({region_cat})")

        # Add bank info
        if bank:
            context_parts.append(f"Bank: {bank}")

        # Add relationships
        rel_context = self._build_relationship_context(
            [],
            referenced_by or [],
            references or [],
        )
        if rel_context:
            context_parts.append(rel_context)

        return EnrichedEmbeddingItem(
            id=symbol_id,
            primary_text=primary_text,
            context_text="; ".join(context_parts),
            category=category,
            metadata={
                "address": address,
                "bank": bank,
                "references": references or [],
                "referenced_by": referenced_by or [],
            },
        )

    def enrich_routine(
        self,
        routine_name: str,
        address: str,
        bank: str,
        description: str = "",
        calls: Optional[list[str]] = None,
        called_by: Optional[list[str]] = None,
        memory_access: Optional[list[str]] = None,
        code_snippet: str = "",
    ) -> EnrichedEmbeddingItem:
        """Create enriched embedding item for a routine."""
        # Primary text: name and description
        primary_parts = [routine_name]
        if description:
            primary_parts.append(description)
        primary_text = ": ".join(primary_parts)

        # Context text: patterns, relationships, memory access
        context_parts = []

        # Add code pattern categories
        patterns = self._get_code_pattern_category(routine_name)
        if patterns:
            context_parts.append(f"Categories: {', '.join(patterns)}")

        # Add bank location
        context_parts.append(f"Bank: {bank}")

        # Add relationship context
        rel_context = self._build_relationship_context(
            calls or [],
            called_by or [],
            [],
        )
        if rel_context:
            context_parts.append(rel_context)

        # Add memory access context
        mem_context = self._build_memory_context(memory_access or [])
        if mem_context:
            context_parts.append(mem_context)

        # Add code pattern hints from snippet
        if code_snippet:
            # Extract key instructions
            key_ops = set()
            for op in ["PHK", "PLB", "REP", "SEP", "XBA", "RTL", "RTS"]:
                if op in code_snippet:
                    key_ops.add(op)
            if key_ops:
                context_parts.append(f"Uses: {', '.join(sorted(key_ops))}")

        return EnrichedEmbeddingItem(
            id=f"routine:{routine_name}",
            primary_text=primary_text,
            context_text="; ".join(context_parts),
            category="routine",
            metadata={
                "address": address,
                "bank": bank,
                "patterns": patterns,
                "calls": calls or [],
                "called_by": called_by or [],
            },
        )

    async def generate_embeddings(
        self,
        items: list[EnrichedEmbeddingItem],
        kb_name: str = "alttp",
        batch_size: int = 50,
        progress_callback: Optional[callable] = None,
    ) -> dict[str, int]:
        """Generate embeddings for enriched items.

        Args:
            items: List of enriched items to embed.
            kb_name: Knowledge base name for logging.
            batch_size: Items per batch.
            progress_callback: Optional progress callback.

        Returns:
            Statistics about generated embeddings.
        """
        if not self._embedding_manager:
            await self.setup()

        # Convert to (id, text) tuples for BatchEmbeddingManager
        embedding_items = [
            (item.id, item.full_text)
            for item in items
        ]

        logger.info(f"Generating {len(embedding_items)} enriched embeddings for {kb_name}")

        created = await self._embedding_manager.generate_embeddings(
            embedding_items,
            kb_name=kb_name,
            batch_size=batch_size,
            progress_callback=progress_callback,
        )

        # Also save enriched metadata
        for item in items:
            metadata_file = self.embeddings_dir / f"{item.id.replace(':', '_')}_meta.json"
            try:
                metadata = {
                    "id": item.id,
                    "primary_text": item.primary_text,
                    "context_text": item.context_text,
                    "category": item.category,
                    "metadata": item.metadata,
                    "generated_at": datetime.now().isoformat(),
                }
                metadata_file.write_text(json.dumps(metadata, indent=2))
            except Exception as e:
                logger.debug(f"Failed to save metadata for {item.id}: {e}")

        return {
            "total": len(items),
            "created": created,
            "kb_name": kb_name,
        }

    async def generate_code_pattern_embeddings(
        self,
        routines: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Generate embeddings for code patterns.

        Groups routines by pattern and creates pattern-level embeddings.
        """
        if not self._embedding_manager:
            await self.setup()

        # Group routines by pattern
        pattern_routines: dict[str, list[str]] = {
            category: [] for category in self.CODE_PATTERNS
        }

        for routine in routines:
            name = routine.get("name", "")
            patterns = self._get_code_pattern_category(name)
            for pattern in patterns:
                pattern_routines[pattern].append(name)

        # Create pattern embeddings
        pattern_items = []
        for pattern, routine_names in pattern_routines.items():
            if not routine_names:
                continue

            # Create pattern description
            keywords = self.CODE_PATTERNS[pattern]
            sample_routines = routine_names[:10]

            text = f"Code pattern: {pattern}\n"
            text += f"Keywords: {', '.join(keywords)}\n"
            text += f"Example routines: {', '.join(sample_routines)}\n"
            text += f"Total routines: {len(routine_names)}"

            pattern_items.append((f"pattern:{pattern}", text))

        if pattern_items:
            created = await self._embedding_manager.generate_embeddings(
                pattern_items,
                kb_name="alttp_patterns",
            )
            return {"patterns": len(pattern_items), "created": created}

        return {"patterns": 0, "created": 0}

    async def generate_relationship_embeddings(
        self,
        routines: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Generate embeddings for routine relationships.

        Creates embeddings for call graphs and clusters.
        """
        if not self._embedding_manager:
            await self.setup()

        # Find highly connected routines (hubs)
        call_counts: dict[str, int] = {}
        for routine in routines:
            name = routine.get("name", "")
            called_by = routine.get("called_by", [])
            call_counts[name] = len(called_by)

        # Sort by call count
        hubs = sorted(call_counts.items(), key=lambda x: x[1], reverse=True)[:50]

        # Create hub embeddings
        hub_items = []
        routine_map = {r.get("name"): r for r in routines}

        for hub_name, call_count in hubs:
            routine = routine_map.get(hub_name, {})
            calls = routine.get("calls", [])
            called_by = routine.get("called_by", [])

            text = f"Hub routine: {hub_name}\n"
            text += f"Called by {call_count} routines\n"
            if called_by[:5]:
                text += f"Callers: {', '.join(called_by[:5])}\n"
            if calls[:5]:
                text += f"Calls: {', '.join(calls[:5])}"

            hub_items.append((f"hub:{hub_name}", text))

        if hub_items:
            created = await self._embedding_manager.generate_embeddings(
                hub_items,
                kb_name="alttp_hubs",
            )
            return {"hubs": len(hub_items), "created": created}

        return {"hubs": 0, "created": 0}


async def enhance_alttp_kb(
    kb_dir: Path,
    symbols: list[dict[str, Any]],
    routines: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run full enhanced embedding generation for ALTTP KB.

    Args:
        kb_dir: Knowledge base directory.
        symbols: List of symbol dictionaries.
        routines: List of routine dictionaries.

    Returns:
        Statistics about generated embeddings.
    """
    builder = ALTTPEmbeddingBuilder(kb_dir)
    await builder.setup()

    stats = {}

    # Generate enriched symbol embeddings
    symbol_items = []
    for sym in symbols:
        item = builder.enrich_symbol(
            symbol_id=sym.get("id", f"symbol:{sym.get('name', '')}"),
            name=sym.get("name", ""),
            address=sym.get("address", ""),
            category=sym.get("category", ""),
            description=sym.get("description", ""),
            references=sym.get("references", []),
            referenced_by=sym.get("referenced_by", []),
            bank=sym.get("bank"),
        )
        symbol_items.append(item)

    if symbol_items:
        result = await builder.generate_embeddings(symbol_items, kb_name="alttp_symbols_enriched")
        stats["symbols"] = result

    # Generate enriched routine embeddings
    routine_items = []
    for routine in routines:
        item = builder.enrich_routine(
            routine_name=routine.get("name", ""),
            address=routine.get("address", ""),
            bank=routine.get("bank", ""),
            description=routine.get("description", ""),
            calls=routine.get("calls", []),
            called_by=routine.get("called_by", []),
            memory_access=routine.get("memory_access", []),
            code_snippet=routine.get("code", "")[:500],
        )
        routine_items.append(item)

    if routine_items:
        result = await builder.generate_embeddings(routine_items, kb_name="alttp_routines_enriched")
        stats["routines"] = result

    # Generate pattern embeddings
    pattern_result = await builder.generate_code_pattern_embeddings(routines)
    stats["patterns"] = pattern_result

    # Generate relationship/hub embeddings
    hub_result = await builder.generate_relationship_embeddings(routines)
    stats["hubs"] = hub_result

    logger.info(f"Enhanced ALTTP embeddings complete: {stats}")
    return stats
