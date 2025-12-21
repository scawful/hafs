"""Oracle of Secrets Knowledge Base.

Builds and manages a knowledge base from the Oracle of Secrets ROM hack source.
Tracks custom routines, symbols, and modifications to vanilla ALTTP.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agents.core import BaseAgent
from hafs.core.embeddings import BatchEmbeddingManager

logger = logging.getLogger(__name__)

ORACLE_KB_PATH = Path.home() / ".context" / "knowledge" / "oracle-of-secrets"
DEFAULT_SOURCE_PATH = Path.home() / "Code" / "Oracle-of-Secrets"


@dataclass
class OracleSymbol:
    """A symbol defined in Oracle of Secrets."""

    id: str
    name: str
    address: str = ""
    symbol_type: str = "label"  # label, constant, macro, variable
    file_path: str = ""
    line_number: int = 0
    description: str = ""
    category: str = ""  # core, sprite, dungeon, item, music, etc.
    vanilla_reference: Optional[str] = None  # Cross-reference to vanilla ALTTP

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OracleRoutine:
    """A routine/function defined in Oracle of Secrets."""

    id: str
    name: str
    address: str = ""
    bank: str = ""
    file_path: str = ""
    line_number: int = 0
    description: str = ""
    category: str = ""
    code_snippet: str = ""
    calls: list[str] = field(default_factory=list)
    called_by: list[str] = field(default_factory=list)
    is_hook: bool = False
    hooks_vanilla: Optional[str] = None  # Which vanilla routine this hooks
    memory_access: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OracleModification:
    """A modification to vanilla ALTTP."""

    id: str
    name: str
    modification_type: str  # hook, override, extend, new
    address: str = ""
    hack_symbol: str = ""
    vanilla_symbol: str = ""
    file_path: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OracleKnowledgeBase:
    """Knowledge base for Oracle of Secrets ROM hack."""

    def __init__(self, kb_path: Optional[Path] = None):
        self.kb_path = kb_path or ORACLE_KB_PATH
        self._symbols: dict[str, OracleSymbol] = {}
        self._routines: dict[str, OracleRoutine] = {}
        self._modifications: list[OracleModification] = []
        self._embeddings: dict[str, list[float]] = {}
        self.embeddings_dir = self.kb_path / "embeddings"

    async def setup(self):
        """Load existing knowledge from disk."""
        self.kb_path.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Load symbols
        symbols_file = self.kb_path / "symbols.json"
        if symbols_file.exists():
            try:
                data = json.loads(symbols_file.read_text())
                for s in data:
                    sym = OracleSymbol(**s)
                    self._symbols[sym.id] = sym
            except Exception as e:
                logger.error(f"Failed to load symbols: {e}")

        # Load routines
        routines_file = self.kb_path / "routines.json"
        if routines_file.exists():
            try:
                data = json.loads(routines_file.read_text())
                for r in data:
                    routine = OracleRoutine(**r)
                    self._routines[routine.id] = routine
            except Exception as e:
                logger.error(f"Failed to load routines: {e}")

        # Load modifications
        mods_file = self.kb_path / "modifications.json"
        if mods_file.exists():
            try:
                data = json.loads(mods_file.read_text())
                self._modifications = [OracleModification(**m) for m in data]
            except Exception as e:
                logger.error(f"Failed to load modifications: {e}")

        # Load embeddings
        for emb_file in self.embeddings_dir.glob("*.json"):
            try:
                data = json.loads(emb_file.read_text())
                if "id" in data and "embedding" in data:
                    self._embeddings[data["id"]] = data["embedding"]
            except Exception:
                pass

        logger.info(f"OracleKB loaded: {len(self._symbols)} symbols, "
                    f"{len(self._routines)} routines, "
                    f"{len(self._modifications)} modifications, "
                    f"{len(self._embeddings)} embeddings")

    def save(self):
        """Save knowledge to disk."""
        self.kb_path.mkdir(parents=True, exist_ok=True)

        # Save symbols
        symbols_file = self.kb_path / "symbols.json"
        symbols_file.write_text(json.dumps(
            [s.to_dict() for s in self._symbols.values()],
            indent=2
        ))

        # Save routines
        routines_file = self.kb_path / "routines.json"
        routines_file.write_text(json.dumps(
            [r.to_dict() for r in self._routines.values()],
            indent=2
        ))

        # Save modifications
        mods_file = self.kb_path / "modifications.json"
        mods_file.write_text(json.dumps(
            [m.to_dict() for m in self._modifications],
            indent=2
        ))

        logger.info(f"OracleKB saved: {len(self._symbols)} symbols, "
                    f"{len(self._routines)} routines")

    async def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search Oracle of Secrets knowledge."""
        results = []
        query_lower = query.lower()

        # Search symbols
        for name, symbol in self._symbols.items():
            if query_lower in name.lower() or query_lower in symbol.description.lower():
                results.append({
                    "type": "symbol",
                    "name": name,
                    "address": symbol.address,
                    "category": symbol.category,
                    "description": symbol.description,
                    "score": 1.0 if query_lower == name.lower() else 0.5
                })

        # Search routines
        for name, routine in self._routines.items():
            if query_lower in name.lower() or query_lower in routine.description.lower():
                results.append({
                    "type": "routine",
                    "name": name,
                    "address": routine.address,
                    "category": routine.category,
                    "description": routine.description,
                    "is_hook": routine.is_hook,
                    "score": 1.0 if query_lower == name.lower() else 0.5
                })

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def get_modifications_by_type(self, mod_type: str) -> list[OracleModification]:
        """Get modifications by type."""
        return [m for m in self._modifications if m.modification_type == mod_type]

    def get_statistics(self) -> dict[str, Any]:
        """Get KB statistics."""
        mod_types = {}
        for m in self._modifications:
            t = m.modification_type
            mod_types[t] = mod_types.get(t, 0) + 1

        categories = {}
        for s in self._symbols.values():
            c = s.category or "uncategorized"
            categories[c] = categories.get(c, 0) + 1

        return {
            "symbols": len(self._symbols),
            "routines": len(self._routines),
            "modifications": len(self._modifications),
            "embeddings": len(self._embeddings),
            "modification_types": mod_types,
            "categories": categories,
        }


class OracleKBBuilder(BaseAgent):
    """Builds Oracle of Secrets knowledge base from source."""

    # Patterns for extracting ASM elements
    LABEL_PATTERN = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*):')
    EQU_PATTERN = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\$[0-9A-Fa-f]+|\d+)')
    DEFINE_PATTERN = re.compile(r'^!([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)')
    MACRO_PATTERN = re.compile(r'^macro\s+([A-Za-z_][A-Za-z0-9_]*)')
    JSL_PATTERN = re.compile(r'\bJSL\s+([A-Za-z_][A-Za-z0-9_]*)')
    JSR_PATTERN = re.compile(r'\bJSR\s+([A-Za-z_][A-Za-z0-9_]*)')
    ORG_PATTERN = re.compile(r'^org\s+\$?([0-9A-Fa-f:]+)')
    INLINE_COMMENT = re.compile(r';\s*(.+)$')  # Inline comment after code
    BLOCK_COMMENT = re.compile(r'^;\s*(.+)$')  # Full line comment
    MEMORY_ACCESS_PATTERN = re.compile(
        r"\$7[EF][0-9A-Fa-f]{4}|\$[0-9A-Fa-f]{2}:[0-9A-Fa-f]{4}|\$21[0-9A-Fa-f]{2}|\$42[0-9A-Fa-f]{2}"
    )

    # Category detection based on file path
    CATEGORY_MAP = {
        "Core": "core",
        "Dungeons": "dungeon",
        "Items": "item",
        "Music": "music",
        "Masks": "mask",
        "Sprites": "sprite",
        "Scripts": "script",
        "Graphics": "graphics",
        "Overworld": "overworld",
    }

    def __init__(self):
        super().__init__(
            "OracleKBBuilder",
            "Build Oracle of Secrets knowledge base from source code."
        )
        self._kb = OracleKnowledgeBase()
        self._vanilla_kb = None
        self._source_path: Optional[Path] = None
        self._embedding_manager: Optional[BatchEmbeddingManager] = None

    async def setup(self):
        await super().setup()
        await self._kb.setup()
        if self.orchestrator:
            self._embedding_manager = BatchEmbeddingManager(
                kb_dir=self._kb.kb_path,
                orchestrator=self.orchestrator,
            )

        # Try to load vanilla KB for cross-referencing
        try:
            from agents.knowledge.alttp import ALTTPKnowledgeBase
            self._vanilla_kb = ALTTPKnowledgeBase()
            await self._vanilla_kb.setup()
            logger.info("Vanilla ALTTP KB loaded for cross-referencing")
        except Exception as e:
            logger.warning(f"Could not load vanilla KB: {e}")

    def _normalize_org_address(self, raw: str) -> tuple[str, str]:
        """Normalize ORG address and return (address, bank)."""
        cleaned = raw.strip().replace("$", "")
        if not cleaned:
            return "", ""

        if ":" in cleaned:
            bank_raw, offset_raw = cleaned.split(":", 1)
            try:
                bank_val = int(bank_raw, 16)
                offset_val = int(offset_raw, 16)
                return f"${bank_val:02X}:{offset_val:04X}", f"{bank_val:02X}"
            except ValueError:
                return f"${cleaned}", ""

        try:
            value = int(cleaned, 16)
        except ValueError:
            return f"${cleaned}", ""

        if value > 0xFFFF:
            bank_val = (value >> 16) & 0xFF
            offset_val = value & 0xFFFF
            return f"${bank_val:02X}:{offset_val:04X}", f"{bank_val:02X}"

        return f"${value:04X}", ""

    def _address_to_int(self, address: str) -> Optional[int]:
        """Convert address string to int for comparisons."""
        if not address:
            return None
        cleaned = address.strip().replace("$", "")
        if not cleaned:
            return None

        if ":" in cleaned:
            bank_raw, offset_raw = cleaned.split(":", 1)
            try:
                return (int(bank_raw, 16) << 16) | int(offset_raw, 16)
            except ValueError:
                return None

        try:
            return int(cleaned, 16)
        except ValueError:
            return None

    def _lookup_vanilla_by_address(self, address: str) -> Optional[str]:
        """Find vanilla symbol id by address."""
        if not self._vanilla_kb or not address:
            return None

        target = self._address_to_int(address)
        if target is None:
            return None

        for sym in self._vanilla_kb._symbols.values():
            if not sym.address:
                continue
            if self._address_to_int(sym.address) == target:
                return sym.id
        return None

    async def build_from_source(
        self,
        source_path: Optional[Path] = None,
        generate_embeddings: bool = True
    ) -> dict[str, Any]:
        """Build knowledge base from Oracle source.

        Args:
            source_path: Path to Oracle-of-Secrets repository
            generate_embeddings: Whether to generate embeddings for symbols

        Returns:
            Build statistics
        """
        self._source_path = Path(source_path or DEFAULT_SOURCE_PATH)

        if not self._source_path.exists():
            raise ValueError(f"Source path not found: {self._source_path}")

        logger.info(f"Building Oracle KB from {self._source_path}")
        start_time = datetime.now()

        # Extract from ASM files
        asm_files = list(self._source_path.rglob("*.asm"))
        logger.info(f"Found {len(asm_files)} ASM files")

        for asm_file in asm_files:
            await self._extract_from_file(asm_file)

        # Build call graph
        self._build_call_graph()

        # Detect hooks and modifications
        await self._detect_modifications()

        # Cross-reference with vanilla
        if self._vanilla_kb:
            await self._cross_reference_vanilla()

        # Generate embeddings
        if generate_embeddings and self.orchestrator:
            await self._generate_embeddings()

        # Save
        self._kb.save()

        elapsed = (datetime.now() - start_time).total_seconds()
        stats = self._kb.get_statistics()
        stats["build_time_seconds"] = elapsed

        logger.info(f"Oracle KB build complete in {elapsed:.1f}s")
        return stats

    async def _extract_from_file(self, file_path: Path):
        """Extract symbols and routines from an ASM file."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            logger.debug(f"Could not read {file_path}: {e}")
            return

        relative_path = str(file_path.relative_to(self._source_path))
        category = self._detect_category(relative_path)

        lines = content.split('\n')
        current_address = ""
        current_bank = ""
        current_routine: Optional[OracleRoutine] = None
        routine_lines = []
        pending_comments: list[str] = []  # Track preceding comments

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                pending_comments = []  # Reset on blank line
                continue

            # Capture block comments (full line comments)
            if stripped.startswith(';'):
                comment_match = self.BLOCK_COMMENT.match(stripped)
                if comment_match:
                    comment_text = comment_match.group(1).strip()
                    # Skip separator lines like ";;;;;;;" or "======="
                    if not all(c in ';=-_#*' for c in comment_text):
                        pending_comments.append(comment_text)
                continue

            # Extract inline comment from current line
            inline_comment = ""
            inline_match = self.INLINE_COMMENT.search(stripped)
            if inline_match:
                inline_comment = inline_match.group(1).strip()

            # Track ORG address
            org_match = self.ORG_PATTERN.match(stripped)
            if org_match:
                current_address, current_bank = self._normalize_org_address(org_match.group(1))
                pending_comments = []
                continue

            # Build description from comments
            description = ""
            if pending_comments:
                description = " ".join(pending_comments)
            if inline_comment:
                if description:
                    description += " - " + inline_comment
                else:
                    description = inline_comment

            # Detect labels (potential routine starts)
            label_match = self.LABEL_PATTERN.match(stripped)
            if label_match:
                name = label_match.group(1)

                # Save previous routine
                if current_routine and routine_lines:
                    current_routine.code_snippet = '\n'.join(routine_lines[:20])
                    self._kb._routines[current_routine.id] = current_routine

                # Create new routine with description
                routine_id = f"oracle:{name}"
                current_routine = OracleRoutine(
                    id=routine_id,
                    name=name,
                    address=current_address,
                    bank=current_bank,
                    file_path=relative_path,
                    line_number=line_num,
                    category=category,
                    description=description,
                )
                routine_lines = [line]

                # Also add as symbol with description
                symbol_id = f"oracle:{name}"
                self._kb._symbols[symbol_id] = OracleSymbol(
                    id=symbol_id,
                    name=name,
                    address=current_address,
                    symbol_type="label",
                    file_path=relative_path,
                    line_number=line_num,
                    category=category,
                    description=description,
                )
                pending_comments = []
                continue

            # Detect EQU definitions
            equ_match = self.EQU_PATTERN.match(stripped)
            if equ_match:
                name = equ_match.group(1)
                value = equ_match.group(2)
                symbol_id = f"oracle:{name}"
                self._kb._symbols[symbol_id] = OracleSymbol(
                    id=symbol_id,
                    name=name,
                    address=value,
                    symbol_type="constant",
                    file_path=relative_path,
                    line_number=line_num,
                    category=category,
                    description=description,
                )
                pending_comments = []
                continue

            # Detect !define
            define_match = self.DEFINE_PATTERN.match(stripped)
            if define_match:
                name = define_match.group(1)
                value = define_match.group(2).strip()
                # Remove inline comment from value if present
                if ';' in value:
                    value = value.split(';')[0].strip()
                symbol_id = f"oracle:!{name}"
                self._kb._symbols[symbol_id] = OracleSymbol(
                    id=symbol_id,
                    name=f"!{name}",
                    address=value,
                    symbol_type="define",
                    file_path=relative_path,
                    line_number=line_num,
                    category=category,
                    description=description,
                )
                pending_comments = []
                continue

            # Detect macros
            macro_match = self.MACRO_PATTERN.match(stripped)
            if macro_match:
                name = macro_match.group(1)
                symbol_id = f"oracle:macro:{name}"
                self._kb._symbols[symbol_id] = OracleSymbol(
                    id=symbol_id,
                    name=name,
                    symbol_type="macro",
                    file_path=relative_path,
                    line_number=line_num,
                    category=category,
                    description=description,
                )
                pending_comments = []
                continue

            # Track routine calls
            if current_routine:
                routine_lines.append(line)

                for mem_match in self.MEMORY_ACCESS_PATTERN.finditer(stripped):
                    addr = mem_match.group(0)
                    if addr not in current_routine.memory_access:
                        current_routine.memory_access.append(addr)

                for jsl_match in self.JSL_PATTERN.finditer(stripped):
                    target = jsl_match.group(1)
                    if target not in current_routine.calls:
                        current_routine.calls.append(target)

                for jsr_match in self.JSR_PATTERN.finditer(stripped):
                    target = jsr_match.group(1)
                    if target not in current_routine.calls:
                        current_routine.calls.append(target)

        # Save last routine
        if current_routine and routine_lines:
            current_routine.code_snippet = '\n'.join(routine_lines[:20])
            self._kb._routines[current_routine.id] = current_routine

    def _detect_category(self, file_path: str) -> str:
        """Detect category from file path."""
        for folder, category in self.CATEGORY_MAP.items():
            if folder in file_path:
                return category
        return "other"

    def _build_call_graph(self):
        """Build called_by references."""
        for routine in self._kb._routines.values():
            for call_target in routine.calls:
                target_id = f"oracle:{call_target}"
                if target_id in self._kb._routines:
                    target = self._kb._routines[target_id]
                    if routine.name not in target.called_by:
                        target.called_by.append(routine.name)

    async def _detect_modifications(self):
        """Detect hooks and modifications to vanilla."""
        # Look for common hook patterns
        hook_patterns = [
            (r'org\s+\$([0-9A-Fa-f]+)', 'hook'),  # Direct org to vanilla address
            (r'JSL\s+([A-Za-z_]+)', 'call'),       # JSL to custom routine
        ]

        for routine in self._kb._routines.values():
            # Check if routine address is in vanilla ROM space
            if routine.address:
                addr = self._address_to_int(routine.address)
                if addr is None:
                    continue
                if 0x008000 <= addr <= 0x1FFFFF:  # Vanilla ROM range
                    routine.is_hook = True
                    self._kb._modifications.append(OracleModification(
                        id=f"mod:{routine.name}",
                        name=routine.name,
                        modification_type="hook",
                        address=routine.address,
                        hack_symbol=routine.name,
                        vanilla_symbol=self._lookup_vanilla_by_address(routine.address) or "",
                        file_path=routine.file_path,
                    ))

    async def _cross_reference_vanilla(self):
        """Cross-reference with vanilla ALTTP KB."""
        if not self._vanilla_kb:
            return

        for symbol in self._kb._symbols.values():
            # Check if symbol name matches vanilla
            if symbol.name in self._vanilla_kb._symbols:
                vanilla = self._vanilla_kb._symbols[symbol.name]
                symbol.vanilla_reference = vanilla.id

            # Check if address matches vanilla
            if symbol.address:
                for v_name, v_sym in self._vanilla_kb._symbols.items():
                    if v_sym.address == symbol.address:
                        symbol.vanilla_reference = v_sym.id
                        break

        # Fill vanilla symbols for modifications
        for mod in self._kb._modifications:
            if mod.vanilla_symbol:
                continue
            if mod.address:
                mod.vanilla_symbol = self._lookup_vanilla_by_address(mod.address) or ""

        logger.info(f"Cross-referenced {sum(1 for s in self._kb._symbols.values() if s.vanilla_reference)} symbols with vanilla")

    async def _generate_embeddings(self):
        """Generate embeddings for symbols with descriptions."""
        if not self.orchestrator or not self._embedding_manager:
            return
        items = []
        for symbol in self._kb._symbols.values():
            text = f"{symbol.name}"
            if symbol.description:
                text = f"{symbol.name}: {symbol.description}"
            elif symbol.category:
                text = f"{symbol.name} ({symbol.category})"
            items.append((symbol.id, text))

        await self._embedding_manager.generate_embeddings(
            items,
            kb_name="oracle_symbols",
        )

        self._kb._embeddings = {}
        for emb_file in self._kb.embeddings_dir.glob("*.json"):
            try:
                data = json.loads(emb_file.read_text())
                if "id" in data and "embedding" in data:
                    self._kb._embeddings[data["id"]] = data["embedding"]
            except Exception:
                continue

        logger.info("Embeddings refreshed: %s total", len(self._kb._embeddings))

    async def run_task(self, task: str = "help") -> dict[str, Any]:
        """Run KB builder task."""
        if task == "help":
            return {
                "usage": [
                    "build - Build KB from ~/Code/Oracle-of-Secrets",
                    "build:PATH - Build KB from specified path",
                    "stats - Get KB statistics",
                    "search:QUERY - Search the KB",
                ]
            }

        if task == "stats":
            return self._kb.get_statistics()

        if task.startswith("search:"):
            query = task[7:].strip()
            results = await self._kb.search(query)
            return {"results": results}

        if task.startswith("build"):
            path = None
            if ":" in task:
                path = Path(task.split(":", 1)[1].strip())
            return await self.build_from_source(path)

        return {"error": f"Unknown task: {task}"}
