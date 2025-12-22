#!/usr/bin/env python3
"""Test cross-domain sample generation."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.training.cross_domain import CrossDomainGenerator
from agents.training.generators.zelda3_generator import Zelda3DisasmGenerator


async def main():
    """Test cross-domain generation."""
    print("=" * 80)
    print("Cross-Domain Sample Generation Test")
    print("=" * 80)
    print()

    # Initialize generators
    print("Setting up generators...")
    zelda3_gen = Zelda3DisasmGenerator(use_template_variation=True)
    await zelda3_gen.setup()

    cross_gen = CrossDomainGenerator()
    await cross_gen.setup()

    # Extract some ASM items
    print("Extracting vanilla ASM routines...")
    asm_items = await zelda3_gen.extract_source_items()
    print(f"Found {len(asm_items)} vanilla routines")
    print()

    if len(asm_items) >= 2:
        # Test ASM+Oracle-style generation (simulated - just use two ASM items)
        print("Testing cross-domain generation...")
        print(f"Primary: {asm_items[0].name}")
        print(f"Secondary: {asm_items[1].name}")
        print()

        sample = await cross_gen.generate_asm_oracle_pair(
            asm_items[0],
            asm_items[1],
        )

        if sample:
            print("✓ Successfully generated cross-domain sample!")
            print()
            print(f"Domain: {sample.domain}")
            print(f"Instruction: {sample.instruction[:200]}...")
            print()
            print(f"Input: {sample.input[:200]}...")
            print()
            print(f"Output: {sample.output[:300]}...")
            print()
        else:
            print("✗ Failed to generate sample")
    else:
        print("Not enough ASM items for testing")

    print("=" * 80)
    print("Cross-domain generation test complete")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
