#!/usr/bin/env python3
"""Test script for ALTTP Unified Knowledge Base with batch embeddings.

This script:
1. Tests the batch embedding manager with checkpointing
2. Generates embeddings for vanilla (usdasm) and Oracle-of-Secrets KBs
3. Tests unified search across all knowledge bases
4. Demonstrates cross-referencing between vanilla and hack

Usage:
    # Quick test (extraction only)
    python scripts/test_alttp_unified_kb.py --quick

    # Full test with embeddings
    python scripts/test_alttp_unified_kb.py --full

    # Run specific test
    python scripts/test_alttp_unified_kb.py --test search

    # Continue from checkpoint
    python scripts/test_alttp_unified_kb.py --continue-embeddings
"""

import asyncio
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# Add hafs to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.knowledge.alttp import ALTTPKnowledgeBase
from agents.knowledge.alttp_unified import (
    UnifiedALTTPKnowledge,
    OracleOfSecretsKB,
    BatchEmbeddingManager,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_stats(stats: dict, title: str = "Statistics"):
    """Print statistics in a formatted way."""
    print(f"\n{title}:")
    print("-" * 40)
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")


async def test_vanilla_kb(generate_embeddings: bool = False):
    """Test the vanilla (usdasm) knowledge base."""
    print_header("Testing Vanilla ALTTP KB (usdasm)")

    kb = ALTTPKnowledgeBase()
    await kb.setup()

    # Get current stats
    stats = kb.get_statistics()
    print_stats(stats, "Current KB Statistics")

    if stats["total_symbols"] == 0:
        print("\nBuilding knowledge base from usdasm...")
        build_stats = await kb.build_from_source(
            generate_embeddings=generate_embeddings,
            deep_analysis=False,  # Skip deep analysis for now
        )
        print_stats(build_stats, "Build Statistics")
    else:
        print(f"\nKB already has {stats['total_symbols']} symbols loaded")

    return kb


async def test_oracle_kb(generate_embeddings: bool = False):
    """Test the Oracle-of-Secrets knowledge base."""
    print_header("Testing Oracle-of-Secrets KB")

    kb = OracleOfSecretsKB()
    await kb.setup()

    # Check if already built
    stats = await kb.run_task("stats")
    print_stats(stats, "Current KB Statistics")

    if stats.get("symbols", 0) == 0:
        print("\nBuilding Oracle-of-Secrets KB...")
        build_stats = await kb.build(generate_embeddings=generate_embeddings)
        print_stats(build_stats, "Build Statistics")
    else:
        print(f"\nKB already has {stats['symbols']} symbols loaded")

    # Show some modifications
    mods = kb.get_modifications()
    if mods:
        print(f"\nSample Modifications ({len(mods)} total):")
        for mod in mods[:5]:
            print(f"  {mod.address}: {mod.hack_symbol} ({mod.modification_type})")

    # Show bank allocations
    bank_usage = kb.get_bank_usage()
    print("\nBank Allocations:")
    for bank, purpose in list(bank_usage["allocations"].items())[:8]:
        print(f"  ${bank:02X}: {purpose}")

    return kb


async def test_unified_kb(generate_embeddings: bool = False):
    """Test the unified knowledge base system."""
    print_header("Testing Unified ALTTP Knowledge Base")

    unified = UnifiedALTTPKnowledge()
    await unified.setup()

    # Get statistics
    stats = unified.get_statistics()
    print_stats(stats, "Unified KB Statistics")

    # Test cross-references
    print("\nCross-References:")
    for xr in unified._cross_refs[:5]:
        print(f"  {xr.get('hack_symbol', '?')} <-> {xr.get('vanilla_symbol', '?')} ({xr.get('match_type', '?')})")

    if not unified._cross_refs:
        print("  (No cross-references built yet - run with --full to build)")

    return unified


async def test_embeddings_generation():
    """Test batch embeddings generation with checkpointing."""
    print_header("Testing Batch Embeddings with Checkpointing")

    unified = UnifiedALTTPKnowledge()
    await unified.setup()

    # Build all KBs with embeddings
    print("Building all KBs with embeddings (this may take a while)...")
    print("Progress will be checkpointed - can resume if interrupted\n")

    def progress_callback(processed, total):
        pct = (processed / total) * 100 if total > 0 else 0
        print(f"\r  Progress: {processed}/{total} ({pct:.1f}%)    ", end="", flush=True)

    # Build vanilla KB embeddings
    print("\n1. Vanilla KB embeddings:")
    if unified._vanilla_kb:
        # Get items for embedding
        items = []
        for sym in unified._vanilla_kb._symbols.values():
            text = f"{sym.name}: {sym.description}" if sym.description else sym.name
            items.append((sym.id, text))

        print(f"   Items to embed: {len(items)}")

        # Note: The actual embedding generation happens through the KB's build method
        # This is just showing the count

    # Build Oracle KB embeddings
    print("\n2. Oracle-of-Secrets KB embeddings:")
    if unified._hack_kb:
        items = []
        for name, sym in unified._hack_kb._symbols.items():
            text = f"{name}: {sym.get('description', '')}"
            items.append((f"symbol:{name}", text))
        print(f"   Items to embed: {len(items)}")

    print("\nTo generate embeddings, run: --full")


async def test_search():
    """Test unified search functionality."""
    print_header("Testing Unified Search")

    unified = UnifiedALTTPKnowledge()
    await unified.setup()

    # Test queries
    queries = [
        "Link's position",
        "sprite animation",
        "dungeon room",
        "music",
        "save game",
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)

        try:
            results = await unified.search(query, limit=5)
            if results:
                for r in results:
                    mod_flag = " [MODIFIED]" if r.is_hack_modification else ""
                    equiv = f" -> vanilla: {r.vanilla_equivalent}" if r.vanilla_equivalent else ""
                    print(f"  [{r.kb_name}] {r.name} ({r.item_type}){mod_flag}{equiv}")
                    print(f"         Score: {r.score:.3f}, Addr: {r.address}")
            else:
                print("  No results (embeddings may not be generated)")
        except Exception as e:
            print(f"  Error: {e}")


async def test_comparison():
    """Test symbol comparison between vanilla and hack."""
    print_header("Testing Symbol Comparison")

    unified = UnifiedALTTPKnowledge()
    await unified.setup()

    # Test comparisons
    symbols_to_compare = ["POSX", "POSY", "MODE", "SprY", "LINKDO"]

    for symbol in symbols_to_compare:
        print(f"\nComparing: {symbol}")
        print("-" * 40)

        try:
            comparison = await unified.compare(symbol)

            if comparison.get("vanilla"):
                v = comparison["vanilla"]
                print(f"  Vanilla: {v['name']} @ {v['address']}")
                print(f"           {v.get('description', '')[:60]}")

            if comparison.get("hack"):
                h = comparison["hack"]
                print(f"  Hack:    {h['name']} @ {h.get('address', 'N/A')}")
                print(f"           {h.get('description', '')[:60]}")

            if comparison.get("cross_refs"):
                print(f"  Cross-refs: {len(comparison['cross_refs'])}")
                for xr in comparison["cross_refs"][:2]:
                    print(f"    - {xr}")

            if comparison.get("is_modified"):
                print("  *** MODIFIED IN HACK ***")
            elif not comparison.get("vanilla") and not comparison.get("hack"):
                print("  (Symbol not found in either KB)")

        except Exception as e:
            print(f"  Error: {e}")


async def test_modifications():
    """Show hack modifications with vanilla context."""
    print_header("Hack Modifications")

    unified = UnifiedALTTPKnowledge()
    await unified.setup()

    try:
        mods = await unified.get_hack_modifications()

        print(f"Total modifications: {len(mods)}\n")

        # Group by type
        by_type = {}
        for mod in mods:
            mod_type = mod.get("type", "unknown")
            if mod_type not in by_type:
                by_type[mod_type] = []
            by_type[mod_type].append(mod)

        for mod_type, items in by_type.items():
            print(f"\n{mod_type.upper()} ({len(items)}):")
            for item in items[:5]:
                vanilla = item.get("vanilla_symbol", "")
                vanilla_str = f" (vanilla: {vanilla})" if vanilla else ""
                print(f"  {item['address']}: {item['hack_symbol']}{vanilla_str}")

    except Exception as e:
        print(f"Error: {e}")


async def run_full_build():
    """Run full build with embeddings."""
    print_header("Full Knowledge Base Build with Embeddings")

    unified = UnifiedALTTPKnowledge()
    await unified.setup()

    print("Starting full build...")
    print("This will generate embeddings using Gemini.")
    print("Progress is checkpointed - safe to interrupt and resume.\n")

    try:
        stats = await unified.build_all(generate_embeddings=True)
        print_stats(stats, "Build Complete")
    except Exception as e:
        logger.error(f"Build failed: {e}")
        print(f"\nBuild interrupted: {e}")
        print("Run again with --continue-embeddings to resume from checkpoint")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test ALTTP Unified Knowledge Base"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test (extraction only, no embeddings)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full test with embeddings generation"
    )
    parser.add_argument(
        "--test",
        choices=["vanilla", "oracle", "unified", "search", "compare", "mods", "embeddings"],
        help="Run specific test"
    )
    parser.add_argument(
        "--continue-embeddings",
        action="store_true",
        help="Continue embeddings from checkpoint"
    )

    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"#  ALTTP Unified Knowledge Base Test")
    print(f"#  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    if args.test:
        # Run specific test
        if args.test == "vanilla":
            await test_vanilla_kb()
        elif args.test == "oracle":
            await test_oracle_kb()
        elif args.test == "unified":
            await test_unified_kb()
        elif args.test == "search":
            await test_search()
        elif args.test == "compare":
            await test_comparison()
        elif args.test == "mods":
            await test_modifications()
        elif args.test == "embeddings":
            await test_embeddings_generation()

    elif args.full or args.continue_embeddings:
        # Full build with embeddings
        await run_full_build()

    else:
        # Quick test (default)
        print("\nRunning quick test (no embeddings)...")
        print("Use --full for full build with embeddings\n")

        await test_vanilla_kb(generate_embeddings=False)
        await test_oracle_kb(generate_embeddings=False)
        await test_unified_kb(generate_embeddings=False)
        await test_comparison()
        await test_modifications()

    print(f"\n{'='*60}")
    print("Test complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
