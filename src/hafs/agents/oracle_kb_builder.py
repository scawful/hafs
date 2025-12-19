"""Oracle of Secrets Knowledge Base Builder.

Builds a knowledge base from the Oracle of Secrets ROM hack source code,
extracting custom routines, symbols, modifications, and cross-referencing
with vanilla ALTTP.

Usage:
    builder = OracleKBBuilder()
    await builder.setup()
    await builder.build_from_source(Path.home() / "Code" / "Oracle-of-Secrets")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from hafs.agents.base import BaseAgent

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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OracleRoutine:
    """A routine/function defined in Oracle of Secrets."""

    id: str
    name: str
    address: str = ""
    file_path: str = ""
    line_number: int = 0
    description: str = ""
    category: str = ""
    code_snippet: str = ""
    calls: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)
    is_hook: bool = False
    hooks_vanilla: Optional[str] = None  # Which vanilla routine this hooks

    def to_dict(self) -> Dict[str, Any]:
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OracleKnowledgeBase:
    """Knowledge base for Oracle of Secrets ROM hack."""

    def __init__(self, kb_path: Optional[Path] = None):
        self.kb_path = kb_path or ORACLE_KB_PATH
        self._symbols: Dict[str, OracleSymbol] = {}
        self._routines: Dict[str, OracleRoutine] = {}
        self._modifications: List[OracleModification] = []
        self._embeddings: Dict[str, List[float]] = {}
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

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
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

    def get_modifications_by_type(self, mod_type: str) -> List[OracleModification]:
        """Get modifications by type."""
        return [m for m in self._modifications if m.modification_type == mod_type]

    def get_statistics(self) -> Dict[str, Any]:
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
    ORG_PATTERN = re.compile(r'^org\s+(\$[0-9A-Fa-f]+)')

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

    async def setup(self):
        await super().setup()
        await self._kb.setup()

        # Try to load vanilla KB for cross-referencing
        try:
            from hafs.agents.alttp_knowledge import ALTTPKnowledgeBase
            self._vanilla_kb = ALTTPKnowledgeBase()
            await self._vanilla_kb.setup()
            logger.info("Vanilla ALTTP KB loaded for cross-referencing")
        except Exception as e:
            logger.warning(f"Could not load vanilla KB: {e}")

    async def build_from_source(
        self,
        source_path: Optional[Path] = None,
        generate_embeddings: bool = True
    ) -> Dict[str, Any]:
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
        current_routine: Optional[OracleRoutine] = None
        routine_lines = []

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith(';'):
                continue

            # Track ORG address
            org_match = self.ORG_PATTERN.match(stripped)
            if org_match:
                current_address = org_match.group(1)
                continue

            # Detect labels (potential routine starts)
            label_match = self.LABEL_PATTERN.match(stripped)
            if label_match:
                name = label_match.group(1)

                # Save previous routine
                if current_routine and routine_lines:
                    current_routine.code_snippet = '\n'.join(routine_lines[:20])
                    self._kb._routines[current_routine.id] = current_routine

                # Create new routine
                routine_id = f"oracle:{name}"
                current_routine = OracleRoutine(
                    id=routine_id,
                    name=name,
                    address=current_address,
                    file_path=relative_path,
                    line_number=line_num,
                    category=category,
                )
                routine_lines = [line]

                # Also add as symbol
                symbol_id = f"oracle:{name}"
                self._kb._symbols[symbol_id] = OracleSymbol(
                    id=symbol_id,
                    name=name,
                    address=current_address,
                    symbol_type="label",
                    file_path=relative_path,
                    line_number=line_num,
                    category=category,
                )
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
                )
                continue

            # Detect !define
            define_match = self.DEFINE_PATTERN.match(stripped)
            if define_match:
                name = define_match.group(1)
                value = define_match.group(2).strip()
                symbol_id = f"oracle:!{name}"
                self._kb._symbols[symbol_id] = OracleSymbol(
                    id=symbol_id,
                    name=f"!{name}",
                    address=value,
                    symbol_type="define",
                    file_path=relative_path,
                    line_number=line_num,
                    category=category,
                )
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
                )
                continue

            # Track routine calls
            if current_routine:
                routine_lines.append(line)

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
                try:
                    addr = int(routine.address.replace('$', ''), 16)
                    if 0x008000 <= addr <= 0x1FFFFF:  # Vanilla ROM range
                        routine.is_hook = True
                        self._kb._modifications.append(OracleModification(
                            id=f"mod:{routine.name}",
                            name=routine.name,
                            modification_type="hook",
                            address=routine.address,
                            hack_symbol=routine.name,
                            file_path=routine.file_path,
                        ))
                except ValueError:
                    pass

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

        logger.info(f"Cross-referenced {sum(1 for s in self._kb._symbols.values() if s.vanilla_reference)} symbols with vanilla")

    async def _generate_embeddings(self):
        """Generate embeddings for symbols with descriptions."""
        if not self.orchestrator:
            return

        count = 0
        for symbol in self._kb._symbols.values():
            if symbol.id in self._kb._embeddings:
                continue

            text = f"{symbol.name}"
            if symbol.description:
                text = f"{symbol.name}: {symbol.description}"
            elif symbol.category:
                text = f"{symbol.name} ({symbol.category})"

            try:
                embedding = await self.orchestrator.embed(text)
                if embedding:
                    self._kb._embeddings[symbol.id] = embedding

                    # Save to disk
                    emb_file = self._kb.embeddings_dir / f"{hashlib.md5(symbol.id.encode()).hexdigest()[:12]}.json"
                    emb_file.write_text(json.dumps({
                        "id": symbol.id,
                        "text": text,
                        "embedding": embedding,
                    }))
                    count += 1

                    if count % 50 == 0:
                        logger.info(f"Generated {count} embeddings...")

                await asyncio.sleep(0.05)  # Rate limiting
            except Exception as e:
                logger.debug(f"Failed to generate embedding for {symbol.name}: {e}")

        logger.info(f"Generated {count} new embeddings")

    async def run_task(self, task: str = "help") -> Dict[str, Any]:
        """Run KB builder task.

        Tasks:
            help - Show usage
            build - Build KB from default source path
            build:PATH - Build KB from specified path
            stats - Get KB statistics
            search:QUERY - Search the KB
        """
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


# CLI entry point
async def main():
    """CLI entry point for Oracle KB building."""
    import sys

    builder = OracleKBBuilder()
    await builder.setup()

    if len(sys.argv) < 2:
        result = await builder.run_task("help")
    else:
        task = " ".join(sys.argv[1:])
        result = await builder.run_task(task)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
