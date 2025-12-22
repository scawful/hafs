#!/usr/bin/env python3
"""Test zelda3 generator extraction."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.training.generators.zelda3_generator import Zelda3DisasmGenerator


async def main():
    print("Testing Zelda3DisasmGenerator...")
    print()

    gen = Zelda3DisasmGenerator(use_template_variation=True)
    await gen.setup()

    print("Extracting source items...")
    items = await gen.extract_source_items()

    print(f"Found {len(items)} source items")
    print()

    if items:
        print("First 5 items:")
        for i, item in enumerate(items[:5], 1):
            print(f"{i}. {item.name} ({item.file_path})")
            print(f"   Lines: {len(item.code.split(chr(10)))}")
            print(f"   Address: {item.address}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
