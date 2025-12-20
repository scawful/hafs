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
from collections import Counter, defaultdict
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
    WRAM_REGION_DESCRIPTIONS = {
        category: description for _, (category, description) in WRAM_REGIONS.items()
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

    def _parse_wram_address(self, address: str) -> Optional[int]:
        """Parse a WRAM address into a 16-bit offset, if possible."""
        if not address:
            return None

        addr_str = (
            str(address)
            .strip()
            .replace("$", "")
            .replace("0x", "")
            .replace(":", "")
            .replace(" ", "")
        )
        if not addr_str:
            return None

        try:
            value = int(addr_str, 16)
        except ValueError:
            match = re.search(r"([0-9A-Fa-f]{4,6})", addr_str)
            if not match:
                return None
            value = int(match.group(1), 16)

        return value & 0xFFFF

    def _get_wram_context(self, address: str) -> tuple[str, str]:
        """Get context for a WRAM address."""
        try:
            addr = self._parse_wram_address(address)
            if addr is None:
                return "unknown", "Unknown WRAM region"

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
        calls_label: str = "Calls",
        called_by_label: str = "Called by",
        references_label: str = "References",
    ) -> str:
        """Build relationship context text."""
        parts = []

        if calls:
            # Limit to most important calls
            important_calls = calls[:10]
            parts.append(f"{calls_label}: {', '.join(important_calls)}")

        if called_by:
            important_callers = called_by[:10]
            parts.append(f"{called_by_label}: {', '.join(important_callers)}")

        if references:
            important_refs = references[:10]
            parts.append(f"{references_label}: {', '.join(important_refs)}")

        return "; ".join(parts) if parts else ""

    def _build_memory_context(
        self,
        memory_access: list[str],
        symbol_lookup: Optional[dict[str, dict[str, Any]]] = None,
    ) -> str:
        """Build memory access context text."""
        if not memory_access:
            return ""

        region_counts: Counter[str] = Counter()
        labels: list[str] = []

        for entry in memory_access[:50]:
            if not entry:
                continue

            entry_str = str(entry).strip()

            # Detect raw address-like entries
            if re.match(r"^\$?[0-9A-Fa-f:]+$", entry_str):
                category, _ = self._get_wram_context(entry_str)
                if category != "unknown":
                    region_counts[category] += 1
                continue

            labels.append(entry_str)
            if symbol_lookup:
                symbol = symbol_lookup.get(entry_str)
                if symbol:
                    category, _ = self._get_wram_context(symbol.get("address", ""))
                    if category != "unknown":
                        region_counts[category] += 1

        parts = []
        if region_counts:
            region_summary = ", ".join(
                f"{name} ({count})" if count > 1 else name
                for name, count in region_counts.most_common(5)
            )
            parts.append(f"Memory regions: {region_summary}")

        if labels:
            unique_labels = list(dict.fromkeys(labels))[:10]
            parts.append(f"Memory symbols: {', '.join(unique_labels)}")

        return "; ".join(parts)

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
        semantic_tags: Optional[list[str]] = None,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
        code_context: str = "",
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
            called_by_label="Referenced by",
        )
        if rel_context:
            context_parts.append(rel_context)

        # Add semantic tags
        if semantic_tags:
            tags = sorted({t for t in semantic_tags if t})
            if tags:
                context_parts.append(f"Tags: {', '.join(tags)}")

        # Add source context
        if file_path:
            context_parts.append(f"Source: {Path(file_path).name}")
        if line_number:
            context_parts.append(f"Line: {line_number}")
        if code_context:
            snippet = " ".join(code_context.split())
            if len(snippet) > 160:
                snippet = f"{snippet[:157]}..."
            context_parts.append(f"Context: {snippet}")

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
                "semantic_tags": semantic_tags or [],
                "file_path": file_path,
                "line_number": line_number,
            },
        )

    def enrich_routine(
        self,
        routine_name: str,
        address: str,
        bank: str,
        description: str = "",
        purpose: str = "",
        complexity: str = "",
        calls: Optional[list[str]] = None,
        called_by: Optional[list[str]] = None,
        memory_access: Optional[list[str]] = None,
        code_snippet: str = "",
        file_path: str = "",
        line_start: Optional[int] = None,
        line_end: Optional[int] = None,
        symbol_lookup: Optional[dict[str, dict[str, Any]]] = None,
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

        if purpose:
            context_parts.append(f"Purpose: {purpose}")
        if complexity and complexity != "unknown":
            context_parts.append(f"Complexity: {complexity}")
        if file_path:
            location = Path(file_path).name
            if line_start and line_end:
                location = f"{location} (lines {line_start}-{line_end})"
            elif line_start:
                location = f"{location} (line {line_start})"
            context_parts.append(f"Source: {location}")

        # Add relationship context
        if calls or called_by:
            context_parts.append(
                f"Call graph: {len(calls or [])} calls, {len(called_by or [])} callers"
            )
        rel_context = self._build_relationship_context(
            calls or [],
            called_by or [],
            [],
        )
        if rel_context:
            context_parts.append(rel_context)

        # Add memory access context
        mem_context = self._build_memory_context(memory_access or [], symbol_lookup)
        if mem_context:
            context_parts.append(mem_context)

        # Add code pattern hints from snippet
        if code_snippet:
            branch_ops = re.findall(r"\b(BEQ|BNE|BMI|BPL|BCS|BCC|BRA|BVS|BVC|BRL)\b", code_snippet)
            jsr_count = len(re.findall(r"\bJSR\b", code_snippet))
            jsl_count = len(re.findall(r"\bJSL\b", code_snippet))

            # Extract key instructions
            key_ops = set()
            for op in ["PHK", "PLB", "REP", "SEP", "XBA", "RTL", "RTS"]:
                if op in code_snippet:
                    key_ops.add(op)
            if key_ops:
                context_parts.append(f"Uses: {', '.join(sorted(key_ops))}")
            if branch_ops:
                context_parts.append(f"Branches: {len(branch_ops)}")
            if jsr_count or jsl_count:
                context_parts.append(f"Subroutine calls: {jsr_count + jsl_count}")

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
                "purpose": purpose,
                "complexity": complexity,
                "file_path": file_path,
                "line_start": line_start,
                "line_end": line_end,
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

    async def generate_memory_region_embeddings(
        self,
        symbols: list[dict[str, Any]],
        routines: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, int]:
        """Generate embeddings for WRAM memory regions."""
        if not self._embedding_manager:
            await self.setup()

        region_symbols: dict[str, list[dict[str, Any]]] = defaultdict(list)
        symbol_lookup: dict[str, dict[str, Any]] = {}

        for symbol in symbols:
            category = str(symbol.get("category", "")).lower()
            if "wram" not in category:
                continue
            address = symbol.get("address", "")
            region, _ = self._get_wram_context(address)
            if region == "unknown":
                continue
            region_symbols[region].append(symbol)
            name = symbol.get("name", "")
            if name:
                symbol_lookup[name] = symbol

        region_routines: dict[str, set[str]] = defaultdict(set)
        if routines:
            for routine in routines:
                routine_name = routine.get("name", "")
                for access in routine.get("memory_access", []):
                    region = "unknown"
                    if access in symbol_lookup:
                        region, _ = self._get_wram_context(symbol_lookup[access].get("address", ""))
                    else:
                        region, _ = self._get_wram_context(access)
                    if region != "unknown" and routine_name:
                        region_routines[region].add(routine_name)

        region_items = []
        for region, region_syms in region_symbols.items():
            description = self.WRAM_REGION_DESCRIPTIONS.get(region, "Unknown WRAM region")
            symbol_names = [s.get("name", "") for s in region_syms if s.get("name")]
            sample_symbols = ", ".join(symbol_names[:15])
            routine_names = sorted(region_routines.get(region, []))
            sample_routines = ", ".join(routine_names[:10])

            text = f"WRAM region: {region}\n"
            text += f"Description: {description}\n"
            if symbol_names:
                text += f"Symbols: {sample_symbols}\n"
                text += f"Total symbols: {len(symbol_names)}\n"
            if routine_names:
                text += f"Example routines: {sample_routines}\n"
                text += f"Total routines: {len(routine_names)}"

            region_items.append((f"region:{region}", text))

        if region_items:
            created = await self._embedding_manager.generate_embeddings(
                region_items,
                kb_name="alttp_wram_regions",
            )
            return {"regions": len(region_items), "created": created}

        return {"regions": 0, "created": 0}

    async def generate_semantic_tag_embeddings(
        self,
        symbols: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Generate embeddings for semantic tags on symbols."""
        if not self._embedding_manager:
            await self.setup()

        tag_symbols: dict[str, list[str]] = defaultdict(list)
        for symbol in symbols:
            name = symbol.get("name", "")
            tags = symbol.get("semantic_tags") or []
            if not name or not tags:
                continue
            for tag in tags:
                if tag:
                    tag_symbols[tag].append(name)

        tag_items = []
        for tag, names in tag_symbols.items():
            sample = ", ".join(names[:20])
            text = f"Semantic tag: {tag}\n"
            text += f"Symbols: {sample}\n"
            text += f"Total symbols: {len(names)}"
            tag_items.append((f"tag:{tag}", text))

        if tag_items:
            created = await self._embedding_manager.generate_embeddings(
                tag_items,
                kb_name="alttp_symbol_tags",
            )
            return {"tags": len(tag_items), "created": created}

        return {"tags": 0, "created": 0}

    async def generate_bank_embeddings(
        self,
        routines: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Generate embeddings per ROM bank."""
        if not self._embedding_manager:
            await self.setup()

        bank_routines: dict[str, list[str]] = defaultdict(list)
        for routine in routines:
            bank = routine.get("bank")
            name = routine.get("name")
            if bank and name:
                bank_routines[str(bank)].append(name)

        bank_items = []
        for bank, names in bank_routines.items():
            sample = ", ".join(names[:20])
            text = f"ROM bank: {bank}\n"
            text += f"Routines: {sample}\n"
            text += f"Total routines: {len(names)}"
            bank_items.append((f"bank:{bank}", text))

        if bank_items:
            created = await self._embedding_manager.generate_embeddings(
                bank_items,
                kb_name="alttp_banks",
            )
            return {"banks": len(bank_items), "created": created}

        return {"banks": 0, "created": 0}

    async def generate_module_embeddings(
        self,
        modules: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Generate embeddings for game modules."""
        if not self._embedding_manager:
            await self.setup()

        module_items = []
        for module in modules:
            module_id = module.get("id")
            name = module.get("name", "")
            description = module.get("description", "")
            routines = module.get("routines", []) or []

            if isinstance(module_id, int):
                module_id_str = f"{module_id:02X}"
            else:
                module_id_str = str(module_id) if module_id is not None else "unknown"

            text = f"Game module {module_id_str}: {name}\n"
            if description:
                text += f"Description: {description}\n"
            if routines:
                sample = ", ".join(routines[:12])
                text += f"Routines: {sample}\n"
                text += f"Total routines: {len(routines)}"

            module_items.append((f"module:{module_id_str}", text))

        if module_items:
            created = await self._embedding_manager.generate_embeddings(
                module_items,
                kb_name="alttp_modules",
            )
            return {"modules": len(module_items), "created": created}

        return {"modules": 0, "created": 0}


async def enhance_alttp_kb(
    kb_dir: Path,
    symbols: list[dict[str, Any]],
    routines: list[dict[str, Any]],
    modules: Optional[list[dict[str, Any]]] = None,
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
    symbol_lookup = {sym.get("name", ""): sym for sym in symbols if sym.get("name")}
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
            semantic_tags=sym.get("semantic_tags", []),
            file_path=sym.get("file_path"),
            line_number=sym.get("line_number"),
            code_context=sym.get("code_context", ""),
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
            purpose=routine.get("purpose", ""),
            complexity=routine.get("complexity", ""),
            calls=routine.get("calls", []),
            called_by=routine.get("called_by", []),
            memory_access=routine.get("memory_access", []),
            code_snippet=routine.get("code", "")[:500],
            file_path=routine.get("file_path", ""),
            line_start=routine.get("line_start"),
            line_end=routine.get("line_end"),
            symbol_lookup=symbol_lookup,
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

    # Generate memory region embeddings
    region_result = await builder.generate_memory_region_embeddings(symbols, routines)
    stats["regions"] = region_result

    # Generate semantic tag embeddings
    tag_result = await builder.generate_semantic_tag_embeddings(symbols)
    stats["tags"] = tag_result

    # Generate bank embeddings
    bank_result = await builder.generate_bank_embeddings(routines)
    stats["banks"] = bank_result

    # Generate module embeddings
    if modules:
        module_result = await builder.generate_module_embeddings(modules)
        stats["modules"] = module_result

    logger.info(f"Enhanced ALTTP embeddings complete: {stats}")
    return stats
