"""Knowledge Base Enhancer.

Enriches the ALTTP and Oracle knowledge bases with:
- SNES hardware register documentation
- ROM hacking infrastructure (banks, hooks, pointers)
- Semantic tags for better clustering
- Cross-references between knowledge bases

Usage:
    enhancer = KBEnhancer()
    await enhancer.setup()
    result = await enhancer.enhance_all()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from hafs.agents.base import BaseAgent

logger = logging.getLogger(__name__)

ALTTP_KB_PATH = Path.home() / ".context" / "knowledge" / "alttp"
ORACLE_KB_PATH = Path.home() / ".context" / "knowledge" / "oracle-of-secrets"
DATA_PATH = Path(__file__).parent.parent / "data"


@dataclass
class EnhancementResult:
    """Result of a KB enhancement operation."""

    registers_updated: int = 0
    symbols_tagged: int = 0
    infrastructure_added: int = 0
    cross_refs_added: int = 0
    errors: List[str] = field(default_factory=list)


class KBEnhancer(BaseAgent):
    """Enhances knowledge bases with additional documentation and metadata."""

    # Semantic tag mappings based on symbol name patterns
    SEMANTIC_TAGS = {
        # Physics and movement
        r"(VEL|SPEED|POS|COORD|SUB[VXY]|RECOIL|KNOCK)": ["physics", "movement"],
        r"(WALK|RUN|DASH|JUMP|FALL|CLIMB|SWIM)": ["movement", "player_state"],

        # Animation
        r"(ANIM|FRAME|TIMER|COUNT|STEP)": ["animation", "timing"],
        r"(GFX|TILE|SPRITE|OAM)": ["graphics", "rendering"],

        # Combat
        r"(DAMAGE|HEALTH|HEART|MAGIC|ARMOR|ATTACK|HIT)": ["combat", "stats"],
        r"(SWORD|SHIELD|BOW|BOMB|ARROW|HOOK)": ["combat", "items"],

        # Game state
        r"(MODULE|SUBMODULE|STATE|MODE|FLAG)": ["game_state", "flow"],
        r"(SAVE|LOAD|SRAM|PROGRESS)": ["save_system", "persistence"],

        # Sprites and enemies
        r"(SPR\d|SPRITE|ENEMY|NPC|BOSS)": ["sprite", "ai"],
        r"(ANCILLA|PROJECTILE|EFFECT)": ["ancilla", "effects"],

        # Dungeon/Overworld
        r"(ROOM|DOOR|CHEST|KEY|DUNGEON)": ["dungeon", "objects"],
        r"(OVERWORLD|AREA|MAP|ENTRANCE)": ["overworld", "navigation"],

        # Audio
        r"(MUSIC|SFX|SOUND|AUDIO|SONG)": ["audio", "music"],

        # Input
        r"(JOY|PAD|BUTTON|INPUT|CTRL)": ["input", "controller"],

        # DMA/Hardware
        r"(DMA|HDMA|VRAM|CGRAM|OAM)": ["hardware", "dma"],
        r"(PPU|APU|CPU)": ["hardware", "registers"],
    }

    def __init__(self):
        super().__init__(
            "KBEnhancer",
            "Enhance knowledge bases with register docs, semantic tags, and cross-references."
        )
        self._register_docs: Dict[str, Any] = {}
        self._rom_hack_info: Dict[str, Any] = {}

    async def setup(self):
        await super().setup()

        # Load register documentation
        register_file = DATA_PATH / "snes_registers.json"
        if register_file.exists():
            try:
                data = json.loads(register_file.read_text())
                self._register_docs = data.get("registers", {})
                self._rom_hack_info = data.get("rom_hacking_infrastructure", {})
                logger.info(f"Loaded {len(self._register_docs)} register definitions")
            except Exception as e:
                logger.error(f"Failed to load register docs: {e}")

    async def enhance_alttp_kb(self) -> EnhancementResult:
        """Enhance the ALTTP knowledge base."""
        result = EnhancementResult()

        symbols_file = ALTTP_KB_PATH / "symbols.json"
        if not symbols_file.exists():
            result.errors.append("symbols.json not found")
            return result

        try:
            symbols = json.loads(symbols_file.read_text())
        except Exception as e:
            result.errors.append(f"Failed to load symbols: {e}")
            return result

        # Update register descriptions
        for symbol in symbols:
            name = symbol.get("name", "")
            category = symbol.get("category", "")

            # Apply register documentation
            if category == "register" and name in self._register_docs:
                reg_info = self._register_docs[name]
                if not symbol.get("description"):
                    symbol["description"] = reg_info.get("description", "")
                    result.registers_updated += 1

                # Add register category as semantic tag
                reg_cat = reg_info.get("category", "")
                if reg_cat:
                    tags = symbol.get("semantic_tags", [])
                    if reg_cat not in tags:
                        tags.append(reg_cat)
                    symbol["semantic_tags"] = tags

            # Apply semantic tags based on name patterns
            import re
            for pattern, tags in self.SEMANTIC_TAGS.items():
                if re.search(pattern, name, re.IGNORECASE):
                    existing_tags = symbol.get("semantic_tags", [])
                    for tag in tags:
                        if tag not in existing_tags:
                            existing_tags.append(tag)
                    symbol["semantic_tags"] = existing_tags
                    result.symbols_tagged += 1
                    break

        # Save updated symbols
        try:
            symbols_file.write_text(json.dumps(symbols, indent=2))
            logger.info(f"Updated {result.registers_updated} registers, tagged {result.symbols_tagged} symbols")
        except Exception as e:
            result.errors.append(f"Failed to save symbols: {e}")

        # Add ROM hacking infrastructure
        result.infrastructure_added = await self._add_rom_hack_infrastructure()

        return result

    async def _add_rom_hack_infrastructure(self) -> int:
        """Add ROM hacking infrastructure to KB."""
        if not self._rom_hack_info:
            return 0

        infra_file = ALTTP_KB_PATH / "rom_hack_info.json"

        try:
            infra_file.write_text(json.dumps(self._rom_hack_info, indent=2))
            logger.info(f"Saved ROM hacking infrastructure to {infra_file}")
            return len(self._rom_hack_info)
        except Exception as e:
            logger.error(f"Failed to save ROM hack info: {e}")
            return 0

    async def enhance_oracle_kb(self) -> EnhancementResult:
        """Enhance the Oracle knowledge base with cross-references."""
        result = EnhancementResult()

        symbols_file = ORACLE_KB_PATH / "symbols.json"
        if not symbols_file.exists():
            result.errors.append("Oracle symbols.json not found")
            return result

        try:
            symbols = json.loads(symbols_file.read_text())
        except Exception as e:
            result.errors.append(f"Failed to load Oracle symbols: {e}")
            return result

        # Load vanilla symbols for cross-reference
        vanilla_symbols = {}
        alttp_symbols_file = ALTTP_KB_PATH / "symbols.json"
        if alttp_symbols_file.exists():
            try:
                vanilla_data = json.loads(alttp_symbols_file.read_text())
                vanilla_symbols = {s["name"]: s for s in vanilla_data}
            except Exception:
                pass

        # Cross-reference Oracle symbols with vanilla
        for symbol in symbols:
            name = symbol.get("name", "")

            # Apply semantic tags
            import re
            for pattern, tags in self.SEMANTIC_TAGS.items():
                if re.search(pattern, name, re.IGNORECASE):
                    existing_tags = symbol.get("semantic_tags", [])
                    for tag in tags:
                        if tag not in existing_tags:
                            existing_tags.append(tag)
                    symbol["semantic_tags"] = existing_tags
                    result.symbols_tagged += 1
                    break

            # Check for vanilla reference
            if name in vanilla_symbols:
                vanilla = vanilla_symbols[name]
                symbol["vanilla_reference"] = {
                    "name": vanilla.get("name"),
                    "address": vanilla.get("address"),
                    "description": vanilla.get("description", "")[:200],
                }
                result.cross_refs_added += 1

        # Save updated Oracle symbols
        try:
            symbols_file.write_text(json.dumps(symbols, indent=2))
            logger.info(f"Tagged {result.symbols_tagged} Oracle symbols, added {result.cross_refs_added} cross-refs")
        except Exception as e:
            result.errors.append(f"Failed to save Oracle symbols: {e}")

        return result

    async def add_ancillae_documentation(self) -> int:
        """Add documentation for ancillae (projectile/effect) symbols.

        Note: In ALTTP KB, ancillae use SPR naming convention (SPR0_YL, etc.)
        These are actually ancilla slots, not regular sprites.
        """

        # Ancillae field documentation based on ALTTP disassembly
        # Format: field_suffix -> description
        ancillae_fields = {
            "_YL": "Y position low byte in room coordinates.",
            "_YH": "Y position high byte (screen-relative).",
            "_XL": "X position low byte in room coordinates.",
            "_XH": "X position high byte (screen-relative).",
            "_CHRM": "Character/tile number for OAM rendering.",
            "_GFX": "Graphics properties and animation frame.",
            "_STSUB": "Status/subtype. Determines variant behavior within type.",
            "_TYP": "Ancilla type ID (0x00=unused, 0x01-0x49 various projectiles/effects).",
            "_TMR": "Timer. Counts down for timed effects, explosions, sparkles.",
            "_SPD": "Speed value for movement calculations.",
            "_DIR": "Direction (0=up, 2=down, 4=left, 6=right, or 8-dir for diagonals).",
            "_YVEL": "Y velocity. Signed byte. Positive=down, negative=up.",
            "_XVEL": "X velocity. Signed byte. Positive=right, negative=left.",
            "_HGT": "Height/altitude above ground. Affects collision and shadows.",
            "_LYR": "Layer. 0=lower layer, 1=upper layer. Affects collision detection.",
            "_OAM": "OAM buffer index for sprite rendering.",
            "_LIFT": "Lift height for thrown/falling objects.",
        }

        # Type-specific documentation
        ancilla_types = {
            0x01: "Somaria platform block",
            0x02: "Fire rod shot",
            0x03: "Unused",
            0x04: "Beam hit (Master Sword)",
            0x05: "Boomerang",
            0x06: "Wall hit debris",
            0x07: "Bomb",
            0x08: "Door debris",
            0x09: "Arrow",
            0x0A: "Ice rod shot/block",
            0x0B: "Sword beam",
            0x0C: "Spin attack sparkle",
            0x0D: "Hookshot head",
            0x0E: "Hookshot chain",
            0x0F: "Somaria block",
            0x10: "Somaria block fizz",
            0x11: "Lamp flame",
            0x12: "Lamp flame trail",
            0x13: "Ether effect",
            0x14: "Bombos effect",
            0x15: "Powder poof",
            0x16: "Wall arrow hit",
            0x17: "Ice shot sparkle",
            0x18: "Hammer stars",
            0x19: "Shovel dirt",
            0x1A: "Ether bolt",
            0x1B: "Bombos bolt",
            0x1C: "Magic powder sprinkle",
            0x1D: "Bush/rock debris",
            0x1E: "Wall bomb debris",
            0x1F: "Splash",
            0x20: "Bird (flute)",
        }

        symbols_file = ALTTP_KB_PATH / "symbols.json"
        if not symbols_file.exists():
            return 0

        try:
            symbols = json.loads(symbols_file.read_text())
        except Exception:
            return 0

        updated = 0
        import re
        for symbol in symbols:
            name = symbol.get("name", "")
            category = symbol.get("category", "")

            if category == "wram_ancillae":
                # Match SPR pattern: SPR0_YL, SPRA_XH, etc.
                match = re.match(r"SPR([0-9A-F])(_[A-Z]+)", name)
                if match:
                    slot = match.group(1)
                    field = match.group(2)

                    if field in ancillae_fields and not symbol.get("description"):
                        desc = f"Ancilla slot {slot}: {ancillae_fields[field]}"
                        symbol["description"] = desc

                        # Add semantic tags
                        tags = symbol.get("semantic_tags", [])
                        for tag in ["ancilla", "projectile", "effects"]:
                            if tag not in tags:
                                tags.append(tag)
                        symbol["semantic_tags"] = tags
                        updated += 1

        if updated > 0:
            try:
                symbols_file.write_text(json.dumps(symbols, indent=2))
                logger.info(f"Added documentation to {updated} ancillae symbols")
            except Exception as e:
                logger.error(f"Failed to save ancillae docs: {e}")

        return updated

    async def enhance_all(self) -> Dict[str, Any]:
        """Run all enhancements."""
        results = {
            "alttp": None,
            "oracle": None,
            "ancillae": 0,
        }

        logger.info("Starting KB enhancement...")

        # Enhance ALTTP KB
        alttp_result = await self.enhance_alttp_kb()
        results["alttp"] = {
            "registers_updated": alttp_result.registers_updated,
            "symbols_tagged": alttp_result.symbols_tagged,
            "infrastructure_added": alttp_result.infrastructure_added,
            "errors": alttp_result.errors,
        }

        # Add ancillae documentation
        results["ancillae"] = await self.add_ancillae_documentation()

        # Enhance Oracle KB
        oracle_result = await self.enhance_oracle_kb()
        results["oracle"] = {
            "symbols_tagged": oracle_result.symbols_tagged,
            "cross_refs_added": oracle_result.cross_refs_added,
            "errors": oracle_result.errors,
        }

        logger.info("KB enhancement complete")
        return results

    async def run_task(self, task: str = "all") -> Dict[str, Any]:
        """Run enhancer task.

        Tasks:
            all - Run all enhancements
            registers - Update register documentation only
            tags - Apply semantic tags only
            ancillae - Document ancillae symbols
            oracle - Enhance Oracle KB with cross-refs
            stats - Show enhancement statistics
        """
        if task == "all":
            return await self.enhance_all()

        if task == "registers":
            result = await self.enhance_alttp_kb()
            return {"registers_updated": result.registers_updated}

        if task == "tags":
            result = await self.enhance_alttp_kb()
            return {"symbols_tagged": result.symbols_tagged}

        if task == "ancillae":
            count = await self.add_ancillae_documentation()
            return {"ancillae_documented": count}

        if task == "oracle":
            result = await self.enhance_oracle_kb()
            return {
                "symbols_tagged": result.symbols_tagged,
                "cross_refs_added": result.cross_refs_added,
            }

        if task == "stats":
            return {
                "register_docs_available": len(self._register_docs),
                "rom_hack_sections": list(self._rom_hack_info.keys()),
                "semantic_tag_patterns": len(self.SEMANTIC_TAGS),
            }

        return {"error": f"Unknown task: {task}"}


async def main():
    """CLI entry point."""
    import sys

    enhancer = KBEnhancer()
    await enhancer.setup()

    task = sys.argv[1] if len(sys.argv) > 1 else "all"
    result = await enhancer.run_task(task)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
