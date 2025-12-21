#!/usr/bin/env python3
"""Clean up duplicate and malformed embeddings.

Scans embedding directories and:
1. Removes duplicate IDs (keeps newest)
2. Fixes malformed embeddings (missing fields)
3. Reports statistics
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


KNOWLEDGE_DIR = Path.home() / ".context" / "knowledge"


def scan_embeddings(emb_dir: Path) -> dict[str, list[Path]]:
    """Scan embeddings and group by ID."""
    id_to_files: dict[str, list[Path]] = defaultdict(list)

    for emb_file in emb_dir.glob("*.json"):
        try:
            data = json.loads(emb_file.read_text())
            emb_id = data.get("id", "")
            if emb_id:
                id_to_files[emb_id].append(emb_file)
        except (json.JSONDecodeError, KeyError):
            pass

    return id_to_files


def find_duplicates(id_to_files: dict[str, list[Path]]) -> list[tuple[str, list[Path]]]:
    """Find IDs with multiple files."""
    return [(emb_id, files) for emb_id, files in id_to_files.items() if len(files) > 1]


def find_malformed(emb_dir: Path) -> list[tuple[Path, list[str]]]:
    """Find embeddings missing required fields."""
    required_fields = ["id", "text", "embedding"]
    malformed = []

    for emb_file in emb_dir.glob("*.json"):
        try:
            data = json.loads(emb_file.read_text())
            missing = [f for f in required_fields if f not in data]
            if missing:
                malformed.append((emb_file, missing))
        except json.JSONDecodeError as e:
            malformed.append((emb_file, [f"JSON error: {e}"]))

    return malformed


def has_embedding(file_path: Path) -> bool:
    """Check if a file contains an actual embedding vector."""
    try:
        data = json.loads(file_path.read_text())
        embedding = data.get("embedding", [])
        return isinstance(embedding, list) and len(embedding) > 0
    except (json.JSONDecodeError, KeyError):
        return False


def cleanup_duplicates(
    duplicates: list[tuple[str, list[Path]]],
    dry_run: bool = True
) -> int:
    """Remove duplicate embeddings, keeping the one with actual embeddings."""
    removed = 0

    for emb_id, files in duplicates:
        # Separate files with embeddings from metadata-only files
        with_embeddings = [f for f in files if has_embedding(f)]
        without_embeddings = [f for f in files if not has_embedding(f)]

        if with_embeddings:
            # Keep the file with embedding, remove meta files
            # If multiple have embeddings, keep newest
            with_embeddings.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            keep = with_embeddings[0]
            to_remove = without_embeddings + with_embeddings[1:]
        else:
            # No embeddings anywhere - keep newest, note the issue
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            keep = files[0]
            to_remove = files[1:]
            print(f"  WARNING: No embeddings for {emb_id}")

        print(f"  ID: {emb_id}")
        print(f"    Keep: {keep.name} (has_embedding: {has_embedding(keep)})")

        for f in to_remove:
            print(f"    Remove: {f.name}")
            if not dry_run:
                f.unlink()
            removed += 1

    return removed


def cleanup_malformed(
    malformed: list[tuple[Path, list[str]]],
    dry_run: bool = True
) -> int:
    """Remove malformed embeddings."""
    removed = 0

    for emb_file, issues in malformed:
        # Only remove _meta files that are clearly not embeddings
        if "_meta" in emb_file.name:
            print(f"  Remove meta file: {emb_file.name} (issues: {issues})")
            if not dry_run:
                emb_file.unlink()
            removed += 1
        else:
            print(f"  Malformed (skipping): {emb_file.name} (issues: {issues})")

    return removed


def main():
    parser = argparse.ArgumentParser(description="Clean up duplicate embeddings")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Show what would be done without making changes")
    parser.add_argument("--execute", action="store_true",
                       help="Actually perform the cleanup")
    parser.add_argument("--project", type=str, default=None,
                       help="Only clean specific project")
    args = parser.parse_args()

    dry_run = not args.execute

    if dry_run:
        print("DRY RUN - no files will be modified")
        print("Use --execute to actually remove files")
        print()

    total_duplicates = 0
    total_malformed = 0
    total_removed = 0

    # Scan each project
    for project_dir in KNOWLEDGE_DIR.iterdir():
        if not project_dir.is_dir():
            continue

        if args.project and project_dir.name != args.project:
            continue

        emb_dir = project_dir / "embeddings"
        if not emb_dir.exists():
            continue

        print(f"Project: {project_dir.name}")

        # Find duplicates
        id_to_files = scan_embeddings(emb_dir)
        duplicates = find_duplicates(id_to_files)

        if duplicates:
            print(f"\n  Found {len(duplicates)} duplicate IDs:")
            total_duplicates += len(duplicates)
            removed = cleanup_duplicates(duplicates, dry_run)
            total_removed += removed

        # Find malformed
        malformed = find_malformed(emb_dir)
        if malformed:
            print(f"\n  Found {len(malformed)} malformed embeddings:")
            total_malformed += len(malformed)
            removed = cleanup_malformed(malformed, dry_run)
            total_removed += removed

        if not duplicates and not malformed:
            print("  No issues found")

        print()

    print("=" * 50)
    print(f"Summary:")
    print(f"  Duplicate IDs found: {total_duplicates}")
    print(f"  Malformed files found: {total_malformed}")
    print(f"  Files {'to remove' if dry_run else 'removed'}: {total_removed}")

    if dry_run and total_removed > 0:
        print()
        print("Run with --execute to actually remove files")


if __name__ == "__main__":
    main()
