#!/usr/bin/env python3
"""Test minimal setup to find where it hangs."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


async def test_setup():
    """Test setup of each component."""
    from agents.training.curator import DataCurator
    from agents.training.generators.gigaleak_generator import GigaleakDataGenerator
    from agents.training.generators.error_generator import ErrorSampleGenerator

    print("Step 1: Create DataCurator")
    curator = DataCurator()

    print("Step 2: Setup DataCurator")
    await curator.setup()
    print("✓ Curator ready")

    print("Step 3: Create GigaleakDataGenerator")
    gigaleak_gen = GigaleakDataGenerator()

    print("Step 4: Setup GigaleakDataGenerator")
    await gigaleak_gen.setup()
    print("✓ Gigaleak ready")

    print("Step 5: Create ErrorSampleGenerator")
    error_gen = ErrorSampleGenerator()

    print("Step 6: Setup ErrorSampleGenerator")
    await error_gen.setup()
    print("✓ Error gen ready")

    print("\n✓✓ ALL SETUPS SUCCESSFUL")


if __name__ == "__main__":
    asyncio.run(test_setup())
