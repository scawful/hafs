"""Gigaleak Knowledge Base - Original Nintendo ALTTP Source.

Extracts and indexes the original Nintendo source code from the gigaleak,
including Japanese comments, original symbol names, and developer notes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from agents.core import BaseAgent
from core.embeddings import BatchEmbeddingManager
from core.orchestrator_v2 import UnifiedOrchestrator, TaskTier, Provider

logger = logging.getLogger(__name__)


@dataclass
class NintendoSymbol:
    """Symbol from original Nintendo source."""

    id: str
    name: str
    symbol_type: str  # GLB (global), EXT (external), EQU (equate), label
    file_path: str
    line_number: int
    address: str = ""
    source_tag: str = ""
    japanese_comment: str = ""
    english_translation: str = ""
    related_usdasm_symbol: Optional[str] = None
    code_context: str = ""


@dataclass
class NintendoModule:
    """A source module from the gigaleak."""

    id: str
    filename: str
    description: str
    file_path: str = ""
    source_tag: str = ""
    symbols: list[str] = field(default_factory=list)
    externals: list[str] = field(default_factory=list)
    japanese_header: str = ""
    english_translation: str = ""


class GigaleakKB(BaseAgent):
    """Knowledge base for original Nintendo ALTTP source from gigaleak.

    The gigaleak contains:
    - Original Nintendo 65816 assembler syntax (GLB, EXT, ORG)
    - Japanese comments and variable names
    - Multiple versions (Japan Ver3, French, German, English PAL)
    - Developer debug code and notes

    Example:
        kb = GigaleakKB()
        await kb.setup()

        # Build from source
        await kb.build()

        # Search with translation
        results = await kb.search("player movement")

        # Get Japanese to English mapping
        translation = await kb.translate_symbol("PLMAIN")
    """

    # Source paths
    GIGALEAK_ROOT = Path.home() / "Code" / "alttp-gigaleak"
    PRIMARY_ROOT = GIGALEAK_ROOT / "1. ゼルダの伝説神々のトライフォース"
    JAPAN_VER3 = PRIMARY_ROOT / "日本_Ver3"

    # Source tag mapping for multi-version indexing
    SOURCE_TAG_MAP = {
        "日本_Ver3": "japan_ver3",
        "NES_Ver2": "nes_ver2",
        "フランス_NES": "french_nes",
        "フランス_PAL": "french_pal",
        "ドイツ_PAL": "german_pal",
        "ドイツ_PAL": "german_pal",
        "英語_PAL": "english_pal",
        "DISASM": "disasm",
        "jpdasm": "jpdasm",
    }

    # File categories
    FILE_CATEGORIES = {
        "zel_main": "Main game loop and initialization",
        "zel_play": "Player/Link routines",
        "zel_char": "Character/sprite handling",
        "zel_enmy": "Enemy AI and behavior",
        "zel_ram": "RAM/WRAM definitions",
        "zel_title": "Title screen and menus",
        "zel_gover": "Game over handling",
        "zel_ending": "Ending sequence",
        "zel_bgm": "Background music",
        "zel_snd": "Sound effects",
        "zel_gmap": "World map",
        "zel_dj": "Dungeon handling",
        "zel_gd": "Graphics/drawing",
        "zel_vma": "VRAM/DMA transfers",
    }

    def __init__(
        self,
        version: str = "full",
        source_roots: Optional[list[Path]] = None,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        super().__init__(
            "GigaleakKB",
            "Knowledge base for original Nintendo ALTTP source code from gigaleak."
        )

        self.version = version
        self.primary_root = self._resolve_primary_root()
        self.source_roots = self._resolve_source_roots(version, source_roots)
        self.source_path = self.source_roots[0] if self.source_roots else self.JAPAN_VER3 / "asm"
        self._embedding_provider = embedding_provider
        self._embedding_model = embedding_model

        # KB storage
        self.kb_dir = self.context_root / "knowledge" / "gigaleak"
        self.kb_dir.mkdir(parents=True, exist_ok=True)

        self.symbols_file = self.kb_dir / "symbols.json"
        self.modules_file = self.kb_dir / "modules.json"
        self.translations_file = self.kb_dir / "translations.json"
        storage_id = BatchEmbeddingManager.resolve_storage_id(
            self._embedding_provider,
            self._embedding_model,
        )
        self.embeddings_dir = BatchEmbeddingManager.resolve_embeddings_dir(
            self.kb_dir,
            storage_id,
        )
        self.embeddings_dir.mkdir(exist_ok=True)

        # In-memory data
        self._symbols: dict[str, NintendoSymbol] = {}
        self._modules: dict[str, NintendoModule] = {}
        self._translations: dict[str, str] = {}
        self._embeddings: dict[str, list[float]] = {}

        self._orchestrator: Optional[UnifiedOrchestrator] = None
        self._embedding_manager: Optional[BatchEmbeddingManager] = None

    def _resolve_primary_root(self) -> Path:
        """Resolve the primary gigaleak source root."""
        if self.GIGALEAK_ROOT.exists():
            candidates = [
                p for p in self.GIGALEAK_ROOT.iterdir()
                if p.is_dir() and p.name.startswith("1.")
            ]
            if candidates:
                return sorted(candidates, key=lambda p: p.name)[0]
        return self.PRIMARY_ROOT

    def _build_version_map(self, primary_root: Path) -> dict[str, Path]:
        """Build a map of version names/tags to ASM roots."""
        mapping: dict[str, Path] = {}
        if primary_root.exists():
            for entry in primary_root.iterdir():
                if not entry.is_dir():
                    continue
                asm_dir = entry / "asm"
                if not asm_dir.exists():
                    continue
                tag = self._derive_source_tag(entry) or entry.name
                mapping[tag] = asm_dir
                mapping[entry.name] = asm_dir

        disasm_jp = self.GIGALEAK_ROOT / "DISASM" / "jpdasm"
        if disasm_jp.exists():
            mapping["jpdasm"] = disasm_jp
            mapping["disasm"] = disasm_jp

        return mapping

    def _resolve_source_roots(
        self,
        version: str,
        source_roots: Optional[list[Path]],
    ) -> list[Path]:
        """Resolve source roots from version selection or explicit paths."""
        if source_roots:
            return [Path(p).expanduser() for p in source_roots]

        primary_root = self.primary_root or self._resolve_primary_root()
        version_map = self._build_version_map(primary_root)

        if version == "full":
            unique = list({p for p in version_map.values()})
            return sorted(unique, key=lambda p: str(p))

        if version_map:
            if version in version_map:
                return [version_map[version]]

            normalized = version.lower()
            for key, path in version_map.items():
                if key.lower() == normalized:
                    return [path]

            for key, path in version_map.items():
                if normalized in key.lower() or key.lower() in normalized:
                    return [path]

        # Fallback to Japan Ver3 if present, else first available
        if "japan_ver3" in version_map:
            return [version_map["japan_ver3"]]
        if "日本_Ver3" in version_map:
            return [version_map["日本_Ver3"]]
        if version_map:
            return [next(iter(version_map.values()))]

        return [self.JAPAN_VER3 / "asm"]

    def _collect_asm_files(self) -> list[tuple[Path, str]]:
        """Collect ASM files across all configured source roots."""
        files: list[tuple[Path, str]] = []
        for root in self.source_roots:
            if not root.exists():
                continue
            source_tag = self._derive_source_tag(root)
            for asm_file in root.rglob("*.asm"):
                if asm_file.is_file():
                    files.append((asm_file, source_tag))
        return files

    def _derive_source_tag(self, path: Path) -> str:
        """Derive a source tag from a path."""
        for part in path.parts:
            if part in self.SOURCE_TAG_MAP:
                return self.SOURCE_TAG_MAP[part]
        fallback = re.sub(r"[^a-z0-9]+", "_", path.name.lower()).strip("_")
        return fallback or "unknown"

    def _make_symbol_id(self, source_tag: str, filename: str, name: str) -> str:
        return f"{source_tag}:{filename}:{name}"

    def _make_module_id(self, source_tag: str, filename: str) -> str:
        return f"{source_tag}:{filename}"

    def _normalize_file_path(self, file_path: Path) -> str:
        try:
            return str(file_path.relative_to(self.GIGALEAK_ROOT))
        except ValueError:
            return str(file_path)

    def _parse_hex(self, value: str) -> Optional[int]:
        raw = value.strip()
        if not raw:
            return None

        raw = raw.split(";")[0].strip()
        if not raw:
            return None

        if raw.lower().endswith("h"):
            raw = raw[:-1]

        if raw.startswith("$"):
            raw = raw[1:]

        if raw.lower().startswith("0x"):
            raw = raw[2:]

        if ":" in raw:
            parts = raw.split(":")
            if len(parts) == 2 and all(re.fullmatch(r"[0-9A-Fa-f]+", p) for p in parts):
                return (int(parts[0], 16) << 16) | int(parts[1], 16)
            return None

        if re.fullmatch(r"[0-9A-Fa-f]+", raw):
            return int(raw, 16)

        if raw.isdigit():
            return int(raw, 10)

        return None

    def _format_address(self, value: int) -> str:
        if value <= 0xFFFF:
            return f"$00:{value:04X}"
        bank = (value >> 16) & 0xFF
        offset = value & 0xFFFF
        return f"${bank:02X}:{offset:04X}"

    def _merge_symbol(self, symbol_id: str, new_symbol: NintendoSymbol) -> None:
        existing = self._symbols.get(symbol_id)
        if not existing:
            self._symbols[symbol_id] = new_symbol
            return

        if new_symbol.symbol_type not in {"GLB", "EXT"} and existing.symbol_type in {"GLB", "EXT"}:
            existing.symbol_type = new_symbol.symbol_type

        if new_symbol.address and not existing.address:
            existing.address = new_symbol.address

        if new_symbol.japanese_comment and not existing.japanese_comment:
            existing.japanese_comment = new_symbol.japanese_comment

        if new_symbol.code_context and not existing.code_context:
            existing.code_context = new_symbol.code_context

        if new_symbol.file_path and not existing.file_path:
            existing.file_path = new_symbol.file_path

        if new_symbol.line_number and not existing.line_number:
            existing.line_number = new_symbol.line_number

        if new_symbol.source_tag and not existing.source_tag:
            existing.source_tag = new_symbol.source_tag

    def _extract_code_context(self, lines: list[str], start_index: int, max_lines: int = 6) -> str:
        snippet = []
        for line in lines[start_index:start_index + max_lines]:
            if line.strip().startswith(";"):
                continue
            snippet.append(line.rstrip())
        return "\n".join(snippet).strip()

    def _find_symbols_by_name(self, name: str) -> list[NintendoSymbol]:
        target = name.lower()
        return [sym for sym in self._symbols.values() if sym.name.lower() == target]

    async def setup(self):
        """Initialize the KB."""
        await super().setup()

        self._orchestrator = UnifiedOrchestrator()
        await self._orchestrator.initialize()
        self._embedding_manager = BatchEmbeddingManager(
            kb_dir=self.kb_dir,
            orchestrator=self._orchestrator,
            embedding_provider=self._embedding_provider,
            embedding_model=self._embedding_model,
        )

        self._load_data()

        logger.info(f"GigaleakKB ready. {len(self._symbols)} symbols, {len(self._modules)} modules")

    def _load_data(self):
        """Load existing data from disk."""
        if self.symbols_file.exists():
            try:
                data = json.loads(self.symbols_file.read_text())
                if isinstance(data, list):
                    items = [(item.get("id", item.get("name", "")), item) for item in data]
                else:
                    items = list(data.items())

                for key, sym_data in items:
                    sym_data = sym_data.copy()
                    sym_id = sym_data.get("id") or key
                    sym_name = sym_data.get("name") or key
                    if not sym_id:
                        continue
                    sym_data["id"] = sym_id
                    sym_data["name"] = sym_name
                    if not sym_data.get("source_tag") and sym_data.get("file_path"):
                        sym_data["source_tag"] = self._derive_source_tag(Path(sym_data["file_path"]))
                    symbol = NintendoSymbol(**sym_data)
                    self._symbols[symbol.id] = symbol
            except Exception as e:
                logger.warning(f"Failed to load symbols: {e}")

        if self.modules_file.exists():
            try:
                data = json.loads(self.modules_file.read_text())
                if isinstance(data, list):
                    items = [(item.get("id", item.get("filename", "")), item) for item in data]
                else:
                    items = list(data.items())

                for key, mod_data in items:
                    mod_data = mod_data.copy()
                    mod_id = mod_data.get("id") or key
                    if not mod_id:
                        continue
                    mod_data["id"] = mod_id
                    if not mod_data.get("source_tag") and mod_data.get("file_path"):
                        mod_data["source_tag"] = self._derive_source_tag(Path(mod_data["file_path"]))
                    module = NintendoModule(**mod_data)
                    self._modules[module.id] = module
            except Exception as e:
                logger.warning(f"Failed to load modules: {e}")

        if self.translations_file.exists():
            try:
                raw_translations = json.loads(self.translations_file.read_text())
                if isinstance(raw_translations, dict):
                    for key, value in raw_translations.items():
                        if key in self._symbols:
                            self._translations[key] = value
                            continue
                        matches = self._find_symbols_by_name(key)
                        if matches:
                            for sym in matches:
                                self._translations[sym.id] = value
                        else:
                            self._translations[key] = value
            except:
                pass

        if self._translations:
            for sym_id, translation in self._translations.items():
                if sym_id in self._symbols:
                    self._symbols[sym_id].english_translation = translation

        # Load embeddings
        for emb_file in self.embeddings_dir.glob("*.json"):
            try:
                data = json.loads(emb_file.read_text())
                if "id" in data and "embedding" in data:
                    self._embeddings[data["id"]] = data["embedding"]
            except:
                pass

    def _save_data(self):
        """Save data to disk."""
        self.symbols_file.write_text(json.dumps(
            {sym_id: asdict(sym) for sym_id, sym in self._symbols.items()},
            indent=2, ensure_ascii=False
        ))

        self.modules_file.write_text(json.dumps(
            {mod_id: asdict(mod) for mod_id, mod in self._modules.items()},
            indent=2, ensure_ascii=False
        ))

        self.translations_file.write_text(json.dumps(
            self._translations, indent=2, ensure_ascii=False
        ))

    async def build(
        self,
        generate_embeddings: bool = True,
        translate_japanese: bool = True,
        batch_size: int = 50,
    ) -> dict[str, int]:
        """Build knowledge base from gigaleak source.

        Args:
            generate_embeddings: Generate semantic embeddings.
            translate_japanese: Translate Japanese comments using Gemini 3.
            batch_size: Batch size for API calls.

        Returns:
            Build statistics.
        """
        if not self.source_roots or not any(root.exists() for root in self.source_roots):
            raise ValueError("Gigaleak sources not found in configured roots")

        roots_preview = ", ".join(str(root) for root in self.source_roots)
        logger.info(f"Building GigaleakKB from roots: {roots_preview}")

        # Find all ASM files
        asm_files = self._collect_asm_files()
        logger.info(f"Found {len(asm_files)} ASM files across {len(self.source_roots)} roots")

        for asm_file, source_tag in asm_files:
            await self._extract_file(asm_file, source_tag)

        # Translate Japanese comments
        if translate_japanese:
            await self._translate_japanese_comments(batch_size)

        # Generate embeddings
        if generate_embeddings:
            await self._generate_embeddings(batch_size)

        self._save_data()

        return {
            "symbols": len(self._symbols),
            "modules": len(self._modules),
            "translations": len(self._translations),
            "embeddings": len(self._embeddings),
        }

    async def _extract_file(self, file_path: Path, source_tag: str):
        """Extract symbols and structure from an ASM file."""
        filename = file_path.stem

        # Determine category
        category = "unknown"
        for prefix, desc in self.FILE_CATEGORIES.items():
            if filename.startswith(prefix):
                category = desc
                break

        try:
            content = file_path.read_text(encoding="shift_jis", errors="ignore")
        except:
            content = file_path.read_text(errors="ignore")

        lines = content.split("\n")

        # Extract module info
        module_id = self._make_module_id(source_tag, filename)
        module = NintendoModule(
            id=module_id,
            filename=filename,
            description=category,
            file_path=self._normalize_file_path(file_path),
            source_tag=source_tag,
        )

        # Extract header comment
        header_lines = []
        for line in lines[:20]:
            if line.strip().startswith(";"):
                header_lines.append(line.strip()[1:].strip())
        module.japanese_header = "\n".join(header_lines)

        current_japanese = []
        current_address = ""

        for i, line in enumerate(lines):
            stripped = line.strip()

            if not stripped:
                continue

            # Collect Japanese comments
            if stripped.startswith(";"):
                comment = stripped[1:].strip()
                # Check if contains Japanese
                if any(ord(c) > 127 for c in comment):
                    current_japanese.append(comment)
                continue

            # Track ORG address
            org_match = re.search(r"\bORG\s+([^\s;]+)", line, re.IGNORECASE)
            if org_match:
                org_value = org_match.group(1).strip()
                parsed = self._parse_hex(org_value)
                if parsed is not None:
                    current_address = self._format_address(parsed)
                continue

            # Match GLB (global) declarations
            if match := re.match(r"\s*GLB\s+(.+)", line, re.IGNORECASE):
                symbols = [s.strip() for s in match.group(1).split(",")]
                for sym in symbols:
                    sym = sym.split(";")[0].strip()
                    if sym and not sym.startswith(";"):
                        sym_id = self._make_symbol_id(source_tag, filename, sym)
                        symbol = NintendoSymbol(
                            id=sym_id,
                            name=sym,
                            symbol_type="GLB",
                            file_path=self._normalize_file_path(file_path),
                            line_number=i + 1,
                            source_tag=source_tag,
                            japanese_comment="\n".join(current_japanese[-3:]),
                        )
                        self._merge_symbol(sym_id, symbol)
                        module.symbols.append(sym_id)
                current_japanese = []

            # Match EXT (external) declarations
            elif match := re.match(r"\s*EXT\s+(.+)", line, re.IGNORECASE):
                symbols = [s.strip() for s in match.group(1).split(",")]
                for sym in symbols:
                    sym = sym.split(";")[0].strip()
                    if sym and not sym.startswith(";"):
                        sym_id = self._make_symbol_id(source_tag, filename, sym)
                        symbol = NintendoSymbol(
                            id=sym_id,
                            name=sym,
                            symbol_type="EXT",
                            file_path=self._normalize_file_path(file_path),
                            line_number=i + 1,
                            source_tag=source_tag,
                        )
                        self._merge_symbol(sym_id, symbol)
                        module.externals.append(sym_id)

            # Match EQU definitions
            elif match := re.match(r"(\w+)\s+EQU\s+(.+)", line, re.IGNORECASE):
                name = match.group(1)
                value = match.group(2).split(";")[0].strip()
                symbol_type = "EQU"
                address = ""

                if value == "$":
                    symbol_type = "label"
                    address = current_address
                else:
                    parsed = self._parse_hex(value)
                    if parsed is not None:
                        address = self._format_address(parsed)

                sym_id = self._make_symbol_id(source_tag, filename, name)
                symbol = NintendoSymbol(
                    id=sym_id,
                    name=name,
                    symbol_type=symbol_type,
                    file_path=self._normalize_file_path(file_path),
                    line_number=i + 1,
                    address=address,
                    source_tag=source_tag,
                    code_context=value,
                    japanese_comment="\n".join(current_japanese[-3:]),
                )
                self._merge_symbol(sym_id, symbol)
                module.symbols.append(sym_id)
                current_japanese = []

            # Match labels (routine definitions)
            elif match := re.match(r"^(\w+)\s*:", line):
                name = match.group(1)
                sym_id = self._make_symbol_id(source_tag, filename, name)
                symbol = NintendoSymbol(
                    id=sym_id,
                    name=name,
                    symbol_type="label",
                    file_path=self._normalize_file_path(file_path),
                    line_number=i + 1,
                    address=current_address,
                    source_tag=source_tag,
                    japanese_comment="\n".join(current_japanese[-3:]),
                    code_context=self._extract_code_context(lines, i),
                )
                self._merge_symbol(sym_id, symbol)
                module.symbols.append(sym_id)
                current_japanese = []

        self._modules[module.id] = module

    async def _translate_japanese_comments(self, batch_size: int):
        """Translate Japanese comments using Gemini 3."""
        logger.info("Translating Japanese comments with Gemini 3...")

        # Collect untranslated comments
        to_translate = []
        for sym_id, sym in self._symbols.items():
            if sym.japanese_comment and sym_id not in self._translations:
                to_translate.append((sym_id, sym.japanese_comment, sym.name))

        if not to_translate:
            logger.info("No new translations needed")
            return

        logger.info(f"Translating {len(to_translate)} Japanese comments")

        # Batch translate
        for i in range(0, len(to_translate), batch_size):
            batch = to_translate[i:i + batch_size]

            # Build translation prompt
            prompt = """Translate these Japanese comments from Nintendo's ALTTP source code to English.
These are 65816 assembly code comments from the 1991 SNES game "The Legend of Zelda: A Link to the Past".

Format: Return JSON with symbol name as key and English translation as value.

Japanese comments to translate:
"""
            for sym_id, comment, name in batch:
                prompt += f"\n{sym_id} ({name}): {comment}"

            prompt += "\n\nReturn only valid JSON, no markdown."

            try:
                result = await self._orchestrator.generate(
                    prompt=prompt,
                    tier=TaskTier.FAST,
                    provider=Provider.GEMINI,
                    model="gemini-3-flash-preview",
                )

                if result.content:
                    # Parse JSON response
                    try:
                        # Clean up markdown if present
                        content = result.content
                        if "```" in content:
                            content = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
                            if content:
                                content = content.group(1)

                        translations = json.loads(content)

                        # Update symbols
                        for key, translation in translations.items():
                            if key in self._symbols:
                                self._translations[key] = translation
                                self._symbols[key].english_translation = translation
                                continue

                            matches = self._find_symbols_by_name(key)
                            if matches:
                                for sym in matches:
                                    self._translations[sym.id] = translation
                                    sym.english_translation = translation
                            else:
                                self._translations[key] = translation

                    except json.JSONDecodeError:
                        logger.warning("Failed to parse translation response")

            except Exception as e:
                logger.warning(f"Translation batch failed: {e}")

            await asyncio.sleep(0.5)  # Rate limiting

        logger.info(f"Translated {len(self._translations)} comments")

    async def _generate_embeddings(self, batch_size: int):
        """Generate embeddings for symbols."""
        logger.info("Generating embeddings...")
        if not self._embedding_manager:
            logger.warning("No embedding manager available")
            return

        self._embedding_manager.batch_size = batch_size

        to_embed = []
        for sym_id, sym in self._symbols.items():
            text = sym.name
            if sym.source_tag:
                text += f" [{sym.source_tag}]"
            if sym.address:
                text += f" @ {sym.address}"
            if sym.english_translation:
                text += f": {sym.english_translation}"
            elif sym.japanese_comment:
                text += f": {sym.japanese_comment}"
            to_embed.append((sym_id, text))

        if not to_embed:
            logger.info("No symbols available for embeddings")
            return

        await self._embedding_manager.generate_embeddings(
            to_embed,
            kb_name="gigaleak_symbols",
        )

        self._embeddings = {}
        for emb_file in self.embeddings_dir.glob("*.json"):
            try:
                data = json.loads(emb_file.read_text())
                if "id" in data and "embedding" in data:
                    self._embeddings[data["id"]] = data["embedding"]
            except Exception:
                continue

        logger.info("Embeddings refreshed: %s total", len(self._embeddings))

    async def search(
        self,
        query: str,
        limit: int = 10,
        include_translation: bool = True,
    ) -> list[dict[str, Any]]:
        """Search the gigaleak knowledge base.

        Args:
            query: Search query (English or Japanese).
            limit: Max results.
            include_translation: Include translations in results.

        Returns:
            Search results.
        """
        if not self._orchestrator:
            await self.setup()

        # Get query embedding
        try:
            query_embedding = await self._orchestrator.embed(query)
            if not query_embedding:
                return []
        except:
            return []

        results = []

        for sym_id, embedding in self._embeddings.items():
            score = self._cosine_similarity(query_embedding, embedding)

            sym = self._symbols.get(sym_id)
            if not sym:
                continue

            result = {
                "id": sym_id,
                "name": sym.name,
                "type": sym.symbol_type,
                "file": Path(sym.file_path).name if sym.file_path else "",
                "source": sym.source_tag,
                "address": sym.address,
                "score": score,
            }

            if include_translation:
                result["japanese"] = sym.japanese_comment
                result["english"] = sym.english_translation or self._translations.get(sym_id, "")

            results.append(result)

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

    async def translate_symbol(self, symbol_name: str) -> dict[str, str]:
        """Get translation for a symbol.

        Args:
            symbol_name: Symbol to translate.

        Returns:
            Translation info.
        """
        sym = self._symbols.get(symbol_name)
        if not sym:
            matches = self._find_symbols_by_name(symbol_name)
            if len(matches) == 1:
                sym = matches[0]
            elif len(matches) > 1:
                return {
                    "symbol": symbol_name,
                    "error": "Multiple symbols found",
                    "matches": [match.id for match in matches],
                }

        if sym and sym.id in self._translations:
            return {
                "symbol": sym.id,
                "cached": True,
                "japanese": sym.japanese_comment,
                "english": self._translations[sym.id],
            }

        # Translate on demand
        if not sym or not sym.japanese_comment:
            return {"symbol": symbol_name, "error": "No Japanese comment found"}

        try:
            result = await self._orchestrator.generate(
                prompt=f"""Translate this Japanese comment from Nintendo's ALTTP source code:

Symbol: {symbol_name}
Japanese: {sym.japanese_comment}

This is from 1991 SNES game assembly code. Provide a clear English translation.""",
                tier=TaskTier.FAST,
                provider=Provider.GEMINI,
            )

            if result.content:
                self._translations[sym.id] = result.content
                sym.english_translation = result.content
                self._save_data()

                return {
                    "symbol": sym.id,
                    "cached": False,
                    "japanese": sym.japanese_comment,
                    "english": result.content,
                }

        except Exception as e:
            return {"symbol": symbol_name, "error": str(e)}

        return {"symbol": symbol_name, "error": "Translation failed"}

    async def cross_reference_usdasm(self, symbol_name: str) -> dict[str, Any]:
        """Find corresponding symbol in usdasm disassembly.

        Args:
            symbol_name: Gigaleak symbol name.

        Returns:
            Cross-reference info.
        """
        from agents.knowledge.alttp import ALTTPKnowledgeBase

        usdasm_kb = ALTTPKnowledgeBase()
        await usdasm_kb.setup()

        # Try exact match
        for sym in usdasm_kb._symbols.values():
            if sym.name.upper() == symbol_name.upper():
                return {
                    "gigaleak_symbol": symbol_name,
                    "usdasm_symbol": sym.name,
                    "match_type": "exact",
                    "address": sym.address,
                    "description": sym.description,
                }

        # Try partial match
        matches = []
        symbol_upper = symbol_name.upper()
        for sym in usdasm_kb._symbols.values():
            if symbol_upper in sym.name.upper() or sym.name.upper() in symbol_upper:
                matches.append({
                    "name": sym.name,
                    "address": sym.address,
                    "description": sym.description,
                })

        if matches:
            return {
                "gigaleak_symbol": symbol_name,
                "match_type": "partial",
                "candidates": matches[:5],
            }

        return {
            "gigaleak_symbol": symbol_name,
            "match_type": "none",
            "candidates": [],
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get KB statistics."""
        glb_count = sum(1 for s in self._symbols.values() if s.symbol_type == "GLB")
        ext_count = sum(1 for s in self._symbols.values() if s.symbol_type == "EXT")
        equ_count = sum(1 for s in self._symbols.values() if s.symbol_type == "EQU")

        source_counts: dict[str, int] = {}
        for sym in self._symbols.values():
            tag = sym.source_tag or "unknown"
            source_counts[tag] = source_counts.get(tag, 0) + 1

        return {
            "total_symbols": len(self._symbols),
            "global_symbols": glb_count,
            "external_refs": ext_count,
            "equates": equ_count,
            "modules": len(self._modules),
            "translations": len(self._translations),
            "embeddings": len(self._embeddings),
            "source_path": str(self.source_path),
            "source_roots": [str(root) for root in self.source_roots],
            "sources": source_counts,
        }
