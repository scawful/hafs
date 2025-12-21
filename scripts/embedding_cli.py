#!/usr/bin/env python3
"""Embedding Service CLI.

Manage background embedding generation, project indexing, and cross-references.

Usage:
    # Check status
    python scripts/embedding_cli.py status

    # List projects
    python scripts/embedding_cli.py list

    # Add a project
    python scripts/embedding_cli.py add alttp ~/Code/usdasm --type asm_disassembly
    python scripts/embedding_cli.py add oracle ~/Code/Oracle-of-Secrets --type rom_hack

    # Start indexing
    python scripts/embedding_cli.py index alttp

    # Watch progress
    python scripts/embedding_cli.py watch alttp

    # Cross-reference
    python scripts/embedding_cli.py xref alttp oracle
    python scripts/embedding_cli.py xref alttp oracle --query SprY
    python scripts/embedding_cli.py xref alttp oracle --semantic

    # Run background service
    python scripts/embedding_cli.py serve
"""

import asyncio
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hafs.services.embedding_service import (
    EmbeddingService,
    ProjectConfig,
    ProjectType,
)


def print_table(headers: list[str], rows: list[list[str]]):
    """Print a simple table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Print header
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    for row in rows:
        print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


async def cmd_status(args, service: EmbeddingService):
    """Show service status."""
    status = await service.get_status()

    print(f"\n=== Embedding Service Status ===\n")
    print(f"Service Running: {status.get('service_running', False)}")
    print(f"Queue: {status.get('queue', [])}")

    projects = status.get("projects", {})
    if projects:
        print(f"\nProjects ({len(projects)}):")
        for name, progress in projects.items():
            total = progress.get("total_items", 0)
            processed = progress.get("processed_items", 0)
            pct = (processed / total * 100) if total > 0 else 0
            status_str = progress.get("status", "unknown")
            print(f"  {name}: {processed}/{total} ({pct:.1f}%) - {status_str}")
    else:
        print("\nNo projects registered.")


async def cmd_list(args, service: EmbeddingService):
    """List registered projects."""
    projects = service.get_projects()

    if not projects:
        print("No projects registered.")
        print("Use 'add' command to add a project.")
        return

    print(f"\n=== Registered Projects ({len(projects)}) ===\n")

    headers = ["Name", "Type", "Path", "Provider", "Model", "Enabled", "Priority"]
    rows = []
    for p in projects:
        rows.append([
            p.name,
            p.project_type.value,
            str(p.path)[:50] + ("..." if len(str(p.path)) > 50 else ""),
            p.embedding_provider or "-",
            p.embedding_model or "-",
            "Yes" if p.enabled else "No",
            str(p.priority),
        ])

    print_table(headers, rows)


async def cmd_add(args, service: EmbeddingService):
    """Add a new project."""
    try:
        project_type = ProjectType(args.type)
    except ValueError:
        print(f"Invalid type: {args.type}")
        print(f"Valid types: {[t.value for t in ProjectType]}")
        return

    config = ProjectConfig(
        name=args.name,
        path=str(Path(args.path).expanduser().absolute()),
        project_type=project_type,
        description=args.description or "",
        embedding_provider=args.provider,
        embedding_model=args.model,
    )

    await service.add_project(config)
    print(f"Added project: {config.name}")
    print(f"  Type: {config.project_type.value}")
    print(f"  Path: {config.path}")


async def cmd_remove(args, service: EmbeddingService):
    """Remove a project."""
    if await service.remove_project(args.name):
        print(f"Removed project: {args.name}")
    else:
        print(f"Project not found: {args.name}")


async def cmd_index(args, service: EmbeddingService):
    """Queue a project for indexing."""
    if await service.queue_indexing(args.name):
        print(f"Queued {args.name} for indexing")
        print("Use 'watch' command to monitor progress")
    else:
        print(f"Project not found: {args.name}")


async def cmd_watch(args, service: EmbeddingService):
    """Watch indexing progress."""
    print(f"\n=== Watching {args.name} ===")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            progress = await service.get_progress_async(args.name)
            if not progress:
                print(f"Project not found: {args.name}")
                break

            total = progress.total_items
            processed = progress.processed_items
            failed = progress.failed_items
            pct = (processed / total * 100) if total > 0 else 0

            status_line = (
                f"\r[{progress.status.upper():10}] "
                f"{processed:>6}/{total:<6} ({pct:5.1f}%) "
                f"| Rate: {progress.rate_items_per_min:.1f}/min "
                f"| ETA: {progress.estimated_remaining_mins:.1f}min "
                f"| Current: {progress.current_item[:30]:<30}"
            )
            print(status_line, end="", flush=True)

            if progress.status in ("completed", "failed"):
                print()
                print(f"\nCompleted: {processed}, Failed: {failed}")
                break

            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\n\nStopped watching.")


async def cmd_xref(args, service: EmbeddingService):
    """Find cross-references between projects."""
    print(f"\n=== Cross-References: {args.source} <-> {args.target} ===\n")

    if args.semantic:
        source_provider = args.source_provider or args.provider
        target_provider = args.target_provider or args.provider
        source_model = args.source_model or args.model
        target_model = args.target_model or args.model
        refs = await service.semantic_cross_reference(
            args.source, args.target,
            threshold=args.threshold or 0.7,
            source_provider=source_provider,
            source_model=source_model,
            target_provider=target_provider,
            target_model=target_model,
        )
    else:
        refs = await service.cross_reference(
            args.source, args.target,
            query=args.query
        )

    if not refs:
        print("No cross-references found.")
        return

    print(f"Found {len(refs)} cross-references:\n")

    headers = ["Source Symbol", "Target Symbol", "Match Type", "Confidence"]
    rows = []
    for ref in refs[:50]:  # Limit to 50
        rows.append([
            ref.source_symbol,
            ref.target_symbol,
            ref.match_type,
            f"{ref.confidence:.2f}",
        ])

    print_table(headers, rows)


async def cmd_stores(args, service: EmbeddingService):
    """List embedding stores for projects."""
    from hafs.core.embeddings import BatchEmbeddingManager

    if args.project:
        config = service.resolve_project(args.project)
        if not config:
            print(f"Unknown project: {args.project}")
            return
        projects = [config]
    else:
        projects = [p for p in service.get_projects() if p.enabled]

    if not projects:
        print("No projects configured.")
        return

    for config in projects:
        root = service.get_embedding_root(config.name)
        print(f"\n{config.name} ({config.project_type.value})")
        print(f"  root: {root}")
        if not root or not root.exists():
            print("  No embedding root found.")
            continue

        stores = []
        default_index = root / "embedding_index.json"
        if default_index.exists():
            stores.append((None, default_index))

        for index_file in sorted(root.glob("embedding_index_*.json")):
            storage_id = index_file.stem.replace("embedding_index_", "", 1)
            stores.append((storage_id, index_file))

        if not stores:
            print("  No embedding stores found.")
            continue

        for storage_id, index_file in stores:
            store_dir = BatchEmbeddingManager.resolve_embeddings_dir(root, storage_id)
            count = 0
            try:
                data = json.loads(index_file.read_text())
                count = len(data) if isinstance(data, dict) else 0
            except Exception:
                pass

            provider = "-"
            model = "-"
            if store_dir.exists():
                for emb_file in store_dir.glob("*.json"):
                    try:
                        emb = json.loads(emb_file.read_text())
                        provider = emb.get("embedding_provider") or provider
                        model = emb.get("embedding_model") or model
                        break
                    except Exception:
                        continue

            label = storage_id or "default"
            print(f"  {label}: {count} embeddings | provider={provider} model={model}")


async def cmd_serve(args, service: EmbeddingService):
    """Run the background service."""
    print("Starting embedding service...")
    print("Press Ctrl+C to stop\n")

    await service.start()

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping service...")
        await service.stop()
        print("Service stopped.")


async def cmd_quick_index(args, service: EmbeddingService):
    """Quick inline indexing without background service."""
    print(f"\n=== Quick Index: {args.name} ({args.batch_size} items) ===\n")

    projects = service.get_projects()
    config = next((p for p in projects if p.name == args.name), None)

    if not config:
        print(f"Project not found: {args.name}")
        return

    from hafs.core.orchestrator_v2 import UnifiedOrchestrator
    import hashlib

    orchestrator = UnifiedOrchestrator()
    await orchestrator.initialize()

    if config.project_type == ProjectType.ASM_DISASSEMBLY:
        from hafs.agents.alttp_knowledge import ALTTPKnowledgeBase

        kb = ALTTPKnowledgeBase(Path(config.path).expanduser())
        await kb.setup()

        existing = set(kb._embeddings.keys())
        missing = [s for s in kb._symbols.values() if s.id not in existing]

        print(f"Existing: {len(existing)}, Missing: {len(missing)}")
        print(f"Processing batch of {args.batch_size}...\n")

        generated = 0
        errors = 0

        for i, symbol in enumerate(missing[:args.batch_size]):
            text = f"{symbol.name}: {symbol.description}" if symbol.description else symbol.name

            try:
                embedding = await orchestrator.embed(text)
                if embedding:
                    emb_file = kb.embeddings_dir / f"{hashlib.md5(symbol.id.encode()).hexdigest()[:12]}.json"
                    emb_file.write_text(json.dumps({
                        "id": symbol.id,
                        "text": text,
                        "embedding": embedding,
                    }))
                    generated += 1
            except Exception as e:
                errors += 1

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{args.batch_size} (generated: {generated}, errors: {errors})")
                await asyncio.sleep(0.1)

        print(f"\nComplete: {generated} generated, {errors} errors")
        print(f"New total: {len(existing) + generated} embeddings")

    else:
        print(f"Quick index not implemented for type: {config.project_type.value}")


async def main():
    parser = argparse.ArgumentParser(
        description="Embedding Service CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Status
    subparsers.add_parser("status", help="Show service status")

    # List
    subparsers.add_parser("list", help="List registered projects")

    # Add
    add_p = subparsers.add_parser("add", help="Add a project")
    add_p.add_argument("name", help="Project name")
    add_p.add_argument("path", help="Path to project")
    add_p.add_argument("--type", "-t", default="asm_disassembly",
                      choices=["asm_disassembly", "rom_hack", "codebase", "documentation"])
    add_p.add_argument("--description", "-d", help="Project description")
    add_p.add_argument("--provider", help="Embedding provider for this project")
    add_p.add_argument("--model", help="Embedding model for this project")

    # Remove
    rm_p = subparsers.add_parser("remove", help="Remove a project")
    rm_p.add_argument("name", help="Project name")

    # Index
    idx_p = subparsers.add_parser("index", help="Queue project for indexing")
    idx_p.add_argument("name", help="Project name")

    # Watch
    watch_p = subparsers.add_parser("watch", help="Watch indexing progress")
    watch_p.add_argument("name", help="Project name")

    # Cross-reference
    xref_p = subparsers.add_parser("xref", help="Find cross-references")
    xref_p.add_argument("source", help="Source project")
    xref_p.add_argument("target", help="Target project")
    xref_p.add_argument("--query", "-q", help="Filter query")
    xref_p.add_argument("--semantic", "-s", action="store_true", help="Use semantic matching")
    xref_p.add_argument("--threshold", "-t", type=float, default=0.7, help="Similarity threshold")
    xref_p.add_argument("--provider", help="Provider for both projects")
    xref_p.add_argument("--model", help="Model for both projects")
    xref_p.add_argument("--source-provider", help="Provider for source project")
    xref_p.add_argument("--target-provider", help="Provider for target project")
    xref_p.add_argument("--source-model", help="Model for source project")
    xref_p.add_argument("--target-model", help="Model for target project")

    # Stores
    stores_p = subparsers.add_parser("stores", help="List embedding stores")
    stores_p.add_argument("project", nargs="?", help="Optional project name")

    # Serve
    subparsers.add_parser("serve", help="Run background service")

    # Quick index
    quick_p = subparsers.add_parser("quick", help="Quick inline indexing")
    quick_p.add_argument("name", help="Project name")
    quick_p.add_argument("--batch-size", "-b", type=int, default=100, help="Batch size")

    args = parser.parse_args()

    service = EmbeddingService()

    commands = {
        "status": cmd_status,
        "list": cmd_list,
        "add": cmd_add,
        "remove": cmd_remove,
        "index": cmd_index,
        "watch": cmd_watch,
        "xref": cmd_xref,
        "serve": cmd_serve,
        "quick": cmd_quick_index,
        "stores": cmd_stores,
    }

    await commands[args.command](args, service)


if __name__ == "__main__":
    asyncio.run(main())
