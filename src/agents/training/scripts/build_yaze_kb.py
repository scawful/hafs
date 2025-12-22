#!/usr/bin/env python3
"""Build YAZE Knowledge Base.

Creates embeddings and reports for YAZE C++ codebase to support:
1. Semantic search of YAZE functions/classes
2. Tool-calling training data generation
3. AI agent tool integration via MCP

Extracts:
- Function signatures (ROM manipulation, graphics, compression)
- Class definitions (Rom, Gfx, Sprite, Label, etc.)
- Tool APIs (OpenROM, SaveROM, DecompressGraphics, etc.)

Generates:
- Embeddings index at ~/.context/knowledge/yaze/
- Tool catalog report
- API reference documentation

Usage:
    python -m agents.training.scripts.build_yaze_kb
    python -m agents.training.scripts.build_yaze_kb --rebuild
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def extract_yaze_symbols(yaze_path: Path) -> dict[str, Any]:
    """Extract symbols from YAZE C++ codebase.

    Uses CppDataGenerator to parse source and extract:
    - Functions
    - Classes
    - Methods

    Args:
        yaze_path: Path to YAZE repository

    Returns:
        Dictionary of symbols:
        {
            "symbol_name": {
                "kind": "function|class|method",
                "signature": "return_type name(params)",
                "file_path": "relative/path",
                "namespace": "namespace",
                "class_name": "ClassName",
                "code": "source code"
            }
        }
    """
    logger.info(f"Extracting symbols from YAZE at: {yaze_path}")

    from agents.training.generators.cpp_generator import CppDataGenerator

    generator = CppDataGenerator(yaze_path=yaze_path)
    await generator.setup()

    # Extract all source items
    items = await generator.extract_source_items()
    logger.info(f"Extracted {len(items)} C++ code units from YAZE")

    # Convert to symbol dictionary
    symbols = {}
    for item in items:
        symbol_id = f"{item.kind}:{item.name}"
        if item.class_name:
            symbol_id = f"{item.class_name}::{item.name}"

        symbols[symbol_id] = {
            "name": item.name,
            "kind": item.kind,
            "signature": item.signature,
            "file_path": item.file_path,
            "namespace": item.namespace,
            "class_name": item.class_name,
            "code": item.code,
            "docstring": item.docstring,
        }

    return symbols


async def generate_embeddings(symbols: dict[str, Any], output_dir: Path) -> None:
    """Generate embeddings for YAZE symbols.

    Creates semantic index for:
    - Function/method search
    - Cross-reference to ALTTP
    - Tool discovery

    Args:
        symbols: Dictionary of YAZE symbols
        output_dir: Output directory for embeddings
    """
    logger.info(f"Generating embeddings for {len(symbols)} symbols...")

    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Initialize orchestrator for embeddings
    from core.orchestrator_v2 import UnifiedOrchestrator

    orchestrator = UnifiedOrchestrator()

    # Generate embeddings
    embedding_index = {}
    for i, (symbol_id, symbol) in enumerate(symbols.items()):
        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i+1}/{len(symbols)} embeddings generated")

        # Create text for embedding
        text_parts = [
            f"{symbol['kind']}: {symbol['name']}",
            f"Signature: {symbol['signature']}",
        ]
        if symbol.get("docstring"):
            text_parts.append(f"Description: {symbol['docstring']}")
        if symbol.get("namespace"):
            text_parts.append(f"Namespace: {symbol['namespace']}")

        text = "\n".join(text_parts)

        try:
            # Generate embedding
            embedding = await orchestrator.embed(text)

            # Save to file
            embedding_file = embeddings_dir / f"{symbol_id.replace('::', '_').replace(':', '_')}.json"
            with open(embedding_file, "w") as f:
                json.dump(
                    {
                        "id": symbol_id,
                        "text": text,
                        "embedding": embedding,
                        "metadata": {
                            "kind": symbol["kind"],
                            "name": symbol["name"],
                            "signature": symbol["signature"],
                            "file_path": symbol["file_path"],
                        },
                    },
                    f,
                )

            # Add to index
            embedding_index[symbol_id] = str(embedding_file.name)

        except Exception as e:
            logger.error(f"Failed to generate embedding for {symbol_id}: {e}")

    # Save embedding index
    index_path = output_dir / "embedding_index.json"
    with open(index_path, "w") as f:
        json.dump(embedding_index, f, indent=2)

    logger.info(f"✓ Generated {len(embedding_index)} embeddings")
    logger.info(f"✓ Embedding index saved to: {index_path}")


async def generate_tool_catalog(symbols: dict[str, Any], output_dir: Path) -> None:
    """Generate tool catalog report.

    Creates comprehensive documentation of YAZE tools with:
    - Function signatures
    - Usage examples
    - Categories (ROM I/O, Graphics, Compression, etc.)

    Args:
        symbols: Dictionary of YAZE symbols
        output_dir: Output directory for reports
    """
    logger.info("Generating tool catalog...")

    # Categorize tools
    categories = {
        "ROM I/O": [],
        "Graphics": [],
        "Compression": [],
        "Sprites": [],
        "Labels": [],
        "Emulation": [],
        "Debugging": [],
        "Other": [],
    }

    for symbol_id, symbol in symbols.items():
        name = symbol["name"].lower()
        code = symbol["code"].lower()

        # Categorize based on name/code
        if any(
            keyword in name
            for keyword in ["rom", "load", "save", "read", "write", "open"]
        ):
            categories["ROM I/O"].append(symbol)
        elif any(keyword in name for keyword in ["gfx", "graphic", "tile", "palette"]):
            categories["Graphics"].append(symbol)
        elif any(keyword in name for keyword in ["compress", "decompress", "lz"]):
            categories["Compression"].append(symbol)
        elif "sprite" in name:
            categories["Sprites"].append(symbol)
        elif "label" in name:
            categories["Labels"].append(symbol)
        elif any(keyword in name for keyword in ["emu", "step", "run", "cpu"]):
            categories["Emulation"].append(symbol)
        elif any(keyword in name for keyword in ["debug", "breakpoint", "watch"]):
            categories["Debugging"].append(symbol)
        else:
            categories["Other"].append(symbol)

    # Generate markdown report
    report_lines = [
        "# YAZE Tool Catalog",
        "",
        "Comprehensive catalog of YAZE ROM editor tools for ALTTP hacking.",
        "",
        f"**Total tools**: {len(symbols)}",
        "",
    ]

    for category, tools in categories.items():
        if not tools:
            continue

        report_lines.extend(
            [
                f"## {category}",
                "",
                f"**Count**: {len(tools)} tools",
                "",
            ]
        )

        for tool in tools[:20]:  # Limit to first 20 per category
            report_lines.extend(
                [
                    f"### `{tool['name']}`",
                    "",
                    f"**Signature**: `{tool['signature']}`",
                    "",
                ]
            )

            if tool.get("docstring"):
                report_lines.extend([f"**Description**: {tool['docstring']}", ""])

            report_lines.extend(
                [
                    f"**File**: `{tool['file_path']}`",
                    "",
                    "```cpp",
                    tool["code"][:500],  # First 500 chars
                    "```",
                    "",
                ]
            )

    # Save report
    reports_dir = Path.home() / ".context" / "reports" / "yaze"
    reports_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = reports_dir / "tool_catalog.md"
    with open(catalog_path, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"✓ Tool catalog saved to: {catalog_path}")


async def build_yaze_kb(yaze_path: Path, rebuild: bool = False) -> None:
    """Build complete YAZE knowledge base.

    Creates:
    1. symbols.json with all functions/classes
    2. embeddings/ directory with semantic index
    3. embedding_index.json mapping symbols to embeddings
    4. reports/yaze/tool_catalog.md

    Args:
        yaze_path: Path to YAZE repository
        rebuild: If True, rebuild from scratch
    """
    output_dir = Path.home() / ".context" / "knowledge" / "yaze"
    output_dir.mkdir(parents=True, exist_ok=True)

    symbols_path = output_dir / "symbols.json"

    # Check if already exists
    if symbols_path.exists() and not rebuild:
        logger.info(f"YAZE KB already exists at: {output_dir}")
        logger.info("Use --rebuild to regenerate")
        return

    logger.info("=" * 80)
    logger.info("BUILDING YAZE KNOWLEDGE BASE")
    logger.info(f"Source: {yaze_path}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 80)

    # Step 1: Extract symbols
    symbols = await extract_yaze_symbols(yaze_path)

    # Save symbols.json
    with open(symbols_path, "w") as f:
        json.dump(symbols, f, indent=2)
    logger.info(f"✓ Saved {len(symbols)} symbols to: {symbols_path}")

    # Step 2: Generate embeddings
    await generate_embeddings(symbols, output_dir)

    # Step 3: Generate tool catalog
    await generate_tool_catalog(symbols, output_dir)

    logger.info("=" * 80)
    logger.info("YAZE KNOWLEDGE BASE COMPLETE")
    logger.info(f"Symbols: {len(symbols)}")
    logger.info(f"Location: {output_dir}")
    logger.info("=" * 80)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build YAZE knowledge base")
    parser.add_argument(
        "--yaze-path",
        type=Path,
        default=Path.home() / "Code" / "yaze",
        help="Path to YAZE repository (default: ~/Code/yaze)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild KB from scratch (even if exists)",
    )

    args = parser.parse_args()

    if not args.yaze_path.exists():
        logger.error(f"YAZE path not found: {args.yaze_path}")
        logger.error("Please specify correct path with --yaze-path")
        return 1

    await build_yaze_kb(args.yaze_path, rebuild=args.rebuild)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
