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
from hafs.core.embeddings import BatchEmbeddingManager
from hafs.core.orchestrator_v2 import UnifiedOrchestrator, TaskTier, Provider

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

    def __init__(self, version: str = "full", source_roots: Optional[list[Path]] = None):
        super().__init__(
            "GigaleakKB",
            "Knowledge base for original Nintendo ALTTP source code from gigaleak."
        )

        self.version = version
        self.source_roots = self._resolve_source_roots(version, source_roots)
        self.source_path = self.source_roots[0] if self.source_roots else self.JAPAN_VER3 / "asm"

        # KB storage
        self.kb_dir = self.context_root / "knowledge" / "gigaleak"
        self.kb_dir.mkdir(parents=True, exist_ok=True)

        self.symbols_file = self.kb_dir / "symbols.json"
        self.modules_file = self.kb_dir / "modules.json"
        self.translations_file = self.kb_dir / "translations.json"
        self.embeddings_dir = self.kb_dir / "embeddings"
        self.embeddings_dir.mkdir(exist_ok=True)

        # In-memory data
        self._symbols: dict[str, NintendoSymbol] = {}
        self._modules: dict[str, NintendoModule] = {}
        self._translations: dict[str, str] = {}
        self._embeddings: dict[str, list[float]] = {}

        self._orchestrator: Optional[UnifiedOrchestrator] = None
        self._embedding_manager: Optional[BatchEmbeddingManager] = None

    async def setup(self):
        """Initialize the KB."""
        await super().setup()

        self._orchestrator = UnifiedOrchestrator()
        await self._orchestrator.initialize()
        self._embedding_manager = BatchEmbeddingManager(
            kb_dir=self.kb_dir,
            orchestrator=self._orchestrator,
        )

        self._load_data()

        logger.info(f"GigaleakKB ready. {len(self._symbols)} symbols, {len(self._modules)} modules")

    def _load_data(self):
        """Load existing data from disk."""
        if self.symbols_file.exists():
            try:
                data = json.loads(self.symbols_file.read_text())
                for name, sym_data in data.items():
                    self._symbols[name] = NintendoSymbol(**sym_data)
            except Exception as e:
                logger.warning(f"Failed to load symbols: {e}")

        if self.modules_file.exists():
            try:
                data = json.loads(self.modules_file.read_text())
                for name, mod_data in data.items():
                    self._modules[name] = NintendoModule(**mod_data)
            except Exception as e:
                logger.warning(f"Failed to load modules: {e}")

        if self.translations_file.exists():
            try:
                self._translations = json.loads(self.translations_file.read_text())
            except:
                pass

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
            {name: asdict(sym) for name, sym in self._symbols.items()},
            indent=2, ensure_ascii=False
        ))

        self.modules_file.write_text(json.dumps(
            {name: asdict(mod) for name, mod in self._modules.items()},
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
        if not self.source_path.exists():
            raise ValueError(f"Gigaleak source not found at {self.source_path}")

        logger.info(f"Building GigaleakKB from {self.source_path}")

        # Find all ASM files
        asm_files = list(self.source_path.glob("*.asm"))
        logger.info(f"Found {len(asm_files)} ASM files")

        for asm_file in asm_files:
            await self._extract_file(asm_file)

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

    async def _extract_file(self, file_path: Path):
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
        module = NintendoModule(
            filename=filename,
            description=category,
        )

        # Extract header comment
        header_lines = []
        for line in lines[:20]:
            if line.strip().startswith(";"):
                header_lines.append(line.strip()[1:].strip())
        module.japanese_header = "\n".join(header_lines)

        current_japanese = []

        for i, line in enumerate(lines):
            # Collect Japanese comments
            if line.strip().startswith(";"):
                comment = line.strip()[1:].strip()
                # Check if contains Japanese
                if any(ord(c) > 127 for c in comment):
                    current_japanese.append(comment)
                continue

            # Match GLB (global) declarations
            if match := re.match(r"\s*GLB\s+(.+)", line, re.IGNORECASE):
                symbols = [s.strip() for s in match.group(1).split(",")]
                for sym in symbols:
                    if sym and not sym.startswith(";"):
                        self._symbols[sym] = NintendoSymbol(
                            name=sym,
                            symbol_type="GLB",
                            file_path=str(file_path),
                            line_number=i + 1,
                            japanese_comment="\n".join(current_japanese[-3:]),
                        )
                        module.symbols.append(sym)
                current_japanese = []

            # Match EXT (external) declarations
            elif match := re.match(r"\s*EXT\s+(.+)", line, re.IGNORECASE):
                symbols = [s.strip() for s in match.group(1).split(",")]
                for sym in symbols:
                    if sym and not sym.startswith(";"):
                        if sym not in self._symbols:
                            self._symbols[sym] = NintendoSymbol(
                                name=sym,
                                symbol_type="EXT",
                                file_path=str(file_path),
                                line_number=i + 1,
                            )
                        module.externals.append(sym)

            # Match EQU definitions
            elif match := re.match(r"(\w+)\s+EQU\s+(.+)", line, re.IGNORECASE):
                name = match.group(1)
                value = match.group(2).strip()
                self._symbols[name] = NintendoSymbol(
                    name=name,
                    symbol_type="EQU",
                    file_path=str(file_path),
                    line_number=i + 1,
                    code_context=value,
                    japanese_comment="\n".join(current_japanese[-3:]),
                )
                current_japanese = []

            # Match labels (routine definitions)
            elif match := re.match(r"^(\w+)\s+EQU\s+\$", line):
                name = match.group(1)
                self._symbols[name] = NintendoSymbol(
                    name=name,
                    symbol_type="label",
                    file_path=str(file_path),
                    line_number=i + 1,
                    japanese_comment="\n".join(current_japanese[-3:]),
                )
                current_japanese = []

        self._modules[filename] = module

    async def _translate_japanese_comments(self, batch_size: int):
        """Translate Japanese comments using Gemini 3."""
        logger.info("Translating Japanese comments with Gemini 3...")

        # Collect untranslated comments
        to_translate = []
        for name, sym in self._symbols.items():
            if sym.japanese_comment and name not in self._translations:
                to_translate.append((name, sym.japanese_comment))

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
            for name, comment in batch:
                prompt += f"\n{name}: {comment}"

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
                        self._translations.update(translations)

                        # Update symbols
                        for name, translation in translations.items():
                            if name in self._symbols:
                                self._symbols[name].english_translation = translation

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
        for name, sym in self._symbols.items():
            text = name
            if sym.english_translation:
                text += f": {sym.english_translation}"
            elif sym.japanese_comment:
                text += f": {sym.japanese_comment}"
            to_embed.append((name, text))

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

        for name, embedding in self._embeddings.items():
            score = self._cosine_similarity(query_embedding, embedding)

            if name in self._symbols:
                sym = self._symbols[name]
                result = {
                    "name": name,
                    "type": sym.symbol_type,
                    "file": Path(sym.file_path).name if sym.file_path else "",
                    "score": score,
                }

                if include_translation:
                    result["japanese"] = sym.japanese_comment
                    result["english"] = sym.english_translation or self._translations.get(name, "")

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
        if symbol_name in self._translations:
            return {
                "symbol": symbol_name,
                "cached": True,
                "japanese": self._symbols.get(symbol_name, NintendoSymbol("", "", "")).japanese_comment,
                "english": self._translations[symbol_name],
            }

        # Translate on demand
        sym = self._symbols.get(symbol_name)
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
                self._translations[symbol_name] = result.content
                sym.english_translation = result.content
                self._save_data()

                return {
                    "symbol": symbol_name,
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

        return {
            "total_symbols": len(self._symbols),
            "global_symbols": glb_count,
            "external_refs": ext_count,
            "equates": equ_count,
            "modules": len(self._modules),
            "translations": len(self._translations),
            "embeddings": len(self._embeddings),
            "source_path": str(self.source_path),
        }
