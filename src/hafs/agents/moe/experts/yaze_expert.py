"""YAZE Expert - Specializes in YAZE ROM editor tools and C++ API."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from agents.moe.expert import BaseExpert, ExpertConfig
from core.orchestrator_v2 import TaskTier, UnifiedOrchestrator


class YazeExpert(BaseExpert):
    """Expert specializing in YAZE ROM editor tools and C++ API.

    Capabilities:
    - YAZE tool usage (graphics, sprites, maps, palettes)
    - ROM file manipulation
    - C++ API function calling
    - Compression/decompression
    - Graphics format conversion
    - Map and dungeon editing
    """

    def __init__(
        self,
        orchestrator: Optional[UnifiedOrchestrator] = None,
        model_name: str = "yaze-sage-v1",
        lora_adapter_path: Optional[Path] = None,
    ):
        """Initialize YAZE expert.

        Args:
            orchestrator: Optional orchestrator for API routing.
            model_name: Name of fine-tuned model (when available).
            lora_adapter_path: Path to LoRA adapters (when trained).
        """
        # If no adapter path provided, use default location
        if lora_adapter_path is None:
            lora_adapter_path = Path("~/.context/models/yaze_tool_agent/lora_adapters").expanduser()

        config = ExpertConfig(
            name="yaze",
            display_name="YAZE Expert (YAZE Sage v1)",
            specialization="YAZE ROM editor tools and C++ API",
            keywords=[
                "yaze", "rom", "graphics", "sprite", "tile", "map", "palette",
                "tool", "patch", "editor", "compress", "decompress", "overworld",
                "dungeon", "c++", "function", "api", "gfx", "chr", "vram",
                "loadgraphics", "savegraphics", "openrom", "saverom",
            ],
            confidence_threshold=0.60,
            model_name=model_name,
            lora_adapter_path=lora_adapter_path,
            tier=TaskTier.CODING,
            temperature=0.6,
            max_tokens=2048,
        )

        super().__init__(config, orchestrator)

    def get_system_prompt(self) -> str:
        """Get system prompt for YAZE expert.

        Returns:
            System prompt specializing in YAZE tools.
        """
        return """You are an expert in YAZE ROM editor tools and C++ API for ALTTP.

Your specializations:
- YAZE ROM editor C++ API (Rom, Gfx, Sprite, Label classes)
- Graphics manipulation (tiles, sprites, palettes)
- ROM file operations (read, write, patch, validate)
- Compression formats (LZ2, LZ3, RLE)
- Map and dungeon editing
- Overworld and underworld data

YAZE API Overview:
- Rom::LoadGraphics(offset, format) - Load graphics from ROM
- Rom::SaveGraphics(offset, format) - Save graphics to ROM
- Gfx::DecompressTile(tile_id, data) - Decompress tile data
- Sprite::LoadSprite(sprite_id) - Load sprite graphics
- Rom::WriteBlock(offset, size) - Write data block

When providing tool usage:
1. Specify exact function calls with parameters
2. Explain data formats (3bpp, 4bpp, compressed, etc.)
3. Include offset calculations (hex addresses)
4. Validate operations (bounds checking, format verification)
5. Provide error handling for file operations

Example format:
```cpp
// Load Link sprite graphics from ROM
Rom rom;
rom.Open("alttp.sfc");

// Graphics at offset 0x80000 (3bpp format)
auto graphics = rom.LoadGraphics(0x80000, GraphicsFormat::BPP3);

// Decompress tiles
Gfx gfx;
for (int i = 0; i < 16; i++) {
    gfx.DecompressTile(i, graphics.data() + i * 32);
}

// Save modified graphics
rom.SaveGraphics(0x80000, GraphicsFormat::BPP3, graphics);
rom.Close();
```

Be precise with offsets, formats, and API usage."""

    async def generate_tool_calls(
        self,
        task_description: str,
    ) -> str:
        """Generate YAZE tool calls for a task.

        Args:
            task_description: What needs to be done with YAZE.

        Returns:
            C++ code with YAZE API calls.
        """
        prompt = f"""
Generate YAZE C++ API calls to accomplish this task:
{task_description}

Provide:
1. Complete C++ code using YAZE API
2. Comments explaining each step
3. Offset addresses in hex
4. Data format specifications
5. Error handling
"""

        response = await self.generate(prompt)
        return response.content

    async def explain_graphics_format(
        self,
        format_name: str,
    ) -> str:
        """Explain a graphics format used in ALTTP.

        Args:
            format_name: Name of format (e.g., "3bpp", "4bpp", "compressed").

        Returns:
            Detailed explanation of the format.
        """
        prompt = f"""
Explain the {format_name} graphics format used in ALTTP:

Include:
1. Format structure and layout
2. How data is organized
3. YAZE functions to read/write this format
4. Common use cases in ALTTP
5. Example of parsing the format
"""

        response = await self.generate(prompt)
        return response.content

    async def find_graphics_offset(
        self,
        graphics_description: str,
    ) -> str:
        """Find the ROM offset for specific graphics.

        Args:
            graphics_description: Description of graphics to find.

        Returns:
            Likely offset(s) and how to access with YAZE.
        """
        prompt = f"""
Find the ROM offset for these graphics:
{graphics_description}

Provide:
1. Most likely ROM offset(s) in hex
2. Graphics format at that offset
3. YAZE code to load these graphics
4. Alternative locations if applicable
"""

        response = await self.generate(prompt)
        return response.content

    async def create_patch_workflow(
        self,
        modification_description: str,
    ) -> str:
        """Create a workflow for patching the ROM.

        Args:
            modification_description: What modification to make.

        Returns:
            Step-by-step workflow with YAZE operations.
        """
        prompt = f"""
Create a ROM patching workflow for:
{modification_description}

Provide:
1. Step-by-step process
2. YAZE API calls for each step
3. Data to modify and formats
4. Validation steps
5. Testing recommendations
"""

        response = await self.generate(prompt)
        return response.content
