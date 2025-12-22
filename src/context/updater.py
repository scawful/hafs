"""Context Updater for the Context Engineering Pipeline.

Implements the Updater phase from AFS research:
- Refresh: Update stale context items from sources
- Synchronize: Keep context consistent with AFS state
- Age: Apply retention policies and move old items to historical

Based on "Everything is Context: Agentic File System Abstraction"
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from models.context import (
    ContextItem,
    ContextPriority,
    MemoryType,
)

logger = logging.getLogger(__name__)


@dataclass
class RetentionPolicy:
    """Retention policy for a memory type."""

    memory_type: MemoryType

    # Maximum age before archival (hours)
    max_age_hours: float = 168  # 1 week default

    # Maximum items to retain
    max_items: int = 100

    # Whether to archive or delete expired items
    archive_on_expire: bool = True

    # Compression threshold for archival (compress if > this many tokens)
    archive_compress_threshold: int = 500


@dataclass
class UpdaterConfig:
    """Configuration for context updater."""

    # How often to run update cycle (seconds)
    update_interval_seconds: float = 300  # 5 minutes

    # How often to check for file changes (seconds)
    sync_interval_seconds: float = 60  # 1 minute

    # Staleness threshold before refresh (hours)
    staleness_threshold_hours: float = 1.0

    # Enable file watching for real-time sync
    enable_file_watch: bool = True

    # Retention policies per type
    retention_policies: dict[MemoryType, RetentionPolicy] = field(
        default_factory=lambda: {
            MemoryType.SCRATCHPAD: RetentionPolicy(
                memory_type=MemoryType.SCRATCHPAD,
                max_age_hours=24,
                max_items=50,
                archive_on_expire=False,
            ),
            MemoryType.EPISODIC: RetentionPolicy(
                memory_type=MemoryType.EPISODIC,
                max_age_hours=72,  # 3 days
                max_items=200,
                archive_on_expire=True,
            ),
            MemoryType.FACT: RetentionPolicy(
                memory_type=MemoryType.FACT,
                max_age_hours=8760,  # 1 year
                max_items=500,
                archive_on_expire=True,
            ),
            MemoryType.EXPERIENTIAL: RetentionPolicy(
                memory_type=MemoryType.EXPERIENTIAL,
                max_age_hours=720,  # 30 days
                max_items=100,
                archive_on_expire=True,
            ),
            MemoryType.PROCEDURAL: RetentionPolicy(
                memory_type=MemoryType.PROCEDURAL,
                max_age_hours=8760,  # 1 year
                max_items=200,
                archive_on_expire=True,
            ),
            MemoryType.USER: RetentionPolicy(
                memory_type=MemoryType.USER,
                max_age_hours=87600,  # 10 years
                max_items=50,
                archive_on_expire=False,
            ),
            MemoryType.HISTORICAL: RetentionPolicy(
                memory_type=MemoryType.HISTORICAL,
                max_age_hours=87600,  # 10 years
                max_items=1000,
                archive_on_expire=False,
            ),
        }
    )


@dataclass
class SyncEvent:
    """An event from file system synchronization."""

    event_type: str  # created, modified, deleted
    path: Path
    timestamp: datetime = field(default_factory=datetime.now)
    memory_type: Optional[MemoryType] = None


class ContextUpdater:
    """Keeps context synchronized with AFS state.

    The Updater phase of the Context Engineering Pipeline:
    1. Refresh: Update stale items from their sources
    2. Synchronize: Watch for file changes and update items
    3. Age: Apply retention policies to archive/delete old items

    Example:
        updater = ContextUpdater(store)
        await updater.start()

        # Later...
        await updater.stop()
    """

    def __init__(
        self,
        store: "ContextStore",  # type: ignore[name-defined]
        config: Optional[UpdaterConfig] = None,
        compressor: Optional[Callable[[str, int], str]] = None,
    ):
        """Initialize the updater.

        Args:
            store: Context store to update
            config: Updater configuration
            compressor: Function for compressing items during archival
        """
        from context.constructor import ContextStore

        self.store: ContextStore = store
        self.config = config or UpdaterConfig()
        self._compressor = compressor

        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        self._sync_task: Optional[asyncio.Task] = None

        # Track file modification times for sync
        self._file_mtimes: dict[Path, float] = {}

        # Event queue for sync events
        self._sync_events: list[SyncEvent] = []

    async def start(self) -> None:
        """Start the background update loop."""
        if self._running:
            return

        self._running = True

        # Start update loop
        self._update_task = asyncio.create_task(self._update_loop())

        # Start sync loop
        if self.config.enable_file_watch:
            self._sync_task = asyncio.create_task(self._sync_loop())

        logger.info("Context updater started")

    async def stop(self) -> None:
        """Stop the background update loop."""
        self._running = False

        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        logger.info("Context updater stopped")

    async def _update_loop(self) -> None:
        """Background loop for refresh and aging."""
        while self._running:
            try:
                # Refresh stale items
                await self._refresh_stale()

                # Apply retention policies
                await self._apply_retention()

                # Process queued sync events
                await self._process_sync_events()

                await asyncio.sleep(self.config.update_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(10)

    async def _sync_loop(self) -> None:
        """Background loop for file synchronization."""
        context_root = self.store.base_path

        while self._running:
            try:
                await self._scan_for_changes(context_root)
                await asyncio.sleep(self.config.sync_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                await asyncio.sleep(10)

    async def _refresh_stale(self) -> None:
        """Refresh items that are stale (not accessed recently)."""
        threshold_hours = self.config.staleness_threshold_hours

        for item in self.store.get_all():
            if item.staleness > threshold_hours:
                if item.source_path and item.source_path.exists():
                    await self._refresh_item(item)

    async def _refresh_item(self, item: ContextItem) -> None:
        """Refresh a single item from its source."""
        if not item.source_path or not item.source_path.exists():
            return

        try:
            # Read fresh content
            new_content = item.source_path.read_text()

            # Check if content changed
            if new_content != item.content:
                item.content = new_content
                item.estimated_tokens = len(new_content) // 4
                item.is_compressed = False
                item.accessed_at = datetime.now()

                # Update relevance if we have embeddings
                if item.embedding is not None:
                    # Would need to regenerate embedding
                    item.embedding = None

                self.store.save(item)
                logger.debug(f"Refreshed item {item.id} from {item.source_path}")

        except Exception as e:
            logger.warning(f"Failed to refresh item {item.id}: {e}")

    async def _apply_retention(self) -> None:
        """Apply retention policies to all items."""
        archived_count = 0
        deleted_count = 0

        for memory_type in MemoryType:
            policy = self.config.retention_policies.get(memory_type)
            if not policy:
                continue

            items = self.store.get_by_type(memory_type)

            # Sort by age (oldest first)
            items.sort(key=lambda x: x.created_at)

            # Check max items limit
            if len(items) > policy.max_items:
                excess = items[: len(items) - policy.max_items]
                for item in excess:
                    if policy.archive_on_expire:
                        await self._archive_item(item)
                        archived_count += 1
                    else:
                        self.store.delete(str(item.id))
                        deleted_count += 1
                items = items[len(items) - policy.max_items :]

            # Check age limit
            max_age = timedelta(hours=policy.max_age_hours)
            now = datetime.now()

            for item in items:
                if now - item.created_at > max_age:
                    if policy.archive_on_expire:
                        await self._archive_item(item)
                        archived_count += 1
                    else:
                        self.store.delete(str(item.id))
                        deleted_count += 1

        if archived_count or deleted_count:
            logger.info(
                f"Retention: archived {archived_count}, deleted {deleted_count}"
            )

    async def _archive_item(self, item: ContextItem) -> None:
        """Archive an item to historical storage."""
        policy = self.config.retention_policies.get(item.memory_type)

        # Compress if over threshold
        if (
            policy
            and item.estimated_tokens > policy.archive_compress_threshold
            and self._compressor
        ):
            target = policy.archive_compress_threshold
            item.content = self._compressor(item.content, target)
            item.estimated_tokens = len(item.content) // 4
            item.is_compressed = True
            item.compression_ratio = item.estimated_tokens / max(
                1, item.original_tokens
            )

        # Change type to historical
        item.memory_type = MemoryType.HISTORICAL
        item.priority = ContextPriority.BACKGROUND

        # Save updated item
        self.store.save(item)
        logger.debug(f"Archived item {item.id}")

    async def _scan_for_changes(self, root: Path) -> None:
        """Scan AFS directories for file changes."""
        # Map directories to memory types
        dir_type_map = {
            "scratchpad": MemoryType.SCRATCHPAD,
            "memory": MemoryType.EPISODIC,
            "knowledge": MemoryType.FACT,
            "history": MemoryType.EPISODIC,
        }

        for dir_name, memory_type in dir_type_map.items():
            dir_path = root / dir_name
            if not dir_path.exists():
                continue

            await self._scan_directory(dir_path, memory_type)

    async def _scan_directory(
        self,
        directory: Path,
        memory_type: MemoryType,
    ) -> None:
        """Scan a directory for file changes."""
        try:
            seen_paths: set[Path] = set()
            for path in directory.rglob("*"):
                if not path.is_file():
                    continue

                if path.name.startswith("."):
                    continue

                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue

                # Check for changes
                old_mtime = self._file_mtimes.get(path)

                if old_mtime is None:
                    # New file
                    self._sync_events.append(
                        SyncEvent(
                            event_type="created",
                            path=path,
                            memory_type=memory_type,
                        )
                    )
                elif mtime > old_mtime:
                    # Modified file
                    self._sync_events.append(
                        SyncEvent(
                            event_type="modified",
                            path=path,
                            memory_type=memory_type,
                        )
                    )

                self._file_mtimes[path] = mtime
                seen_paths.add(path)

            tracked_paths = [
                path
                for path in list(self._file_mtimes)
                if path.is_relative_to(directory)
            ]
            for path in tracked_paths:
                if path in seen_paths or path.exists():
                    continue
                self._sync_events.append(
                    SyncEvent(
                        event_type="deleted",
                        path=path,
                        memory_type=memory_type,
                    )
                )
                self._file_mtimes.pop(path, None)

        except Exception as e:
            logger.warning(f"Failed to scan directory {directory}: {e}")

    async def _process_sync_events(self) -> None:
        """Process queued synchronization events."""
        if not self._sync_events:
            return

        events = self._sync_events[:50]  # Process in batches
        self._sync_events = self._sync_events[50:]

        for event in events:
            await self._handle_sync_event(event)

    async def _handle_sync_event(self, event: SyncEvent) -> None:
        """Handle a single sync event."""
        if event.event_type == "created":
            await self._create_item_from_file(event.path, event.memory_type)
        elif event.event_type == "modified":
            await self._update_item_from_file(event.path)
        elif event.event_type == "deleted":
            await self._delete_item_for_file(event.path)

    async def _create_item_from_file(
        self,
        path: Path,
        memory_type: Optional[MemoryType],
    ) -> None:
        """Create a context item from a file."""
        try:
            content = path.read_text()

            # Determine source type
            suffix = path.suffix.lower()
            source_type_map = {
                ".md": "markdown",
                ".json": "json",
                ".py": "code",
                ".js": "code",
                ".ts": "code",
                ".txt": "text",
            }
            source_type = source_type_map.get(suffix, "text")

            item = ContextItem(
                content=content,
                memory_type=memory_type or MemoryType.FACT,
                priority=ContextPriority.MEDIUM,
                source_path=path,
                source_type=source_type,
            )

            self.store.save(item)
            logger.debug(f"Created item from {path}")

        except Exception as e:
            logger.warning(f"Failed to create item from {path}: {e}")

    async def _update_item_from_file(self, path: Path) -> None:
        """Update a context item when its source file changes."""
        # Find item by source path
        for item in self.store.get_all():
            if item.source_path == path:
                await self._refresh_item(item)
                return

        # No existing item, create new one
        memory_type = self._infer_memory_type(path)
        await self._create_item_from_file(path, memory_type)

    async def _delete_item_for_file(self, path: Path) -> None:
        """Delete context item when source file is deleted."""
        for item in self.store.get_all():
            if item.source_path == path:
                self.store.delete(str(item.id))
                logger.debug(f"Deleted item for removed file {path}")
                return

    def _infer_memory_type(self, path: Path) -> MemoryType:
        """Infer memory type from file path."""
        parts = path.parts

        if "scratchpad" in parts:
            return MemoryType.SCRATCHPAD
        elif "history" in parts:
            return MemoryType.EPISODIC
        elif "memory" in parts:
            return MemoryType.EPISODIC
        elif "knowledge" in parts:
            return MemoryType.FACT
        elif "procedures" in parts:
            return MemoryType.PROCEDURAL
        elif "user" in parts or "preferences" in parts:
            return MemoryType.USER

        return MemoryType.FACT

    def record_interaction(
        self,
        user_message: str,
        ai_response: str,
        metadata: Optional[dict] = None,
    ) -> ContextItem:
        """Record an interaction as an episodic memory.

        Args:
            user_message: User's message
            ai_response: AI's response
            metadata: Optional additional metadata

        Returns:
            Created context item
        """
        content = f"User: {user_message}\n\nAssistant: {ai_response}"

        if metadata:
            import json
            content += f"\n\n[Metadata: {json.dumps(metadata)}]"

        item = ContextItem(
            content=content,
            memory_type=MemoryType.EPISODIC,
            priority=ContextPriority.MEDIUM,
            source_type="interaction",
        )

        self.store.save(item)
        return item

    def record_learning(
        self,
        pattern: str,
        outcome: str,
        success: bool,
    ) -> ContextItem:
        """Record a learned pattern as experiential memory.

        Args:
            pattern: The pattern or strategy used
            outcome: The result of using the pattern
            success: Whether it was successful

        Returns:
            Created context item
        """
        status = "SUCCESS" if success else "FAILURE"
        content = f"[{status}] Pattern: {pattern}\n\nOutcome: {outcome}"

        item = ContextItem(
            content=content,
            memory_type=MemoryType.EXPERIENTIAL,
            priority=ContextPriority.HIGH if success else ContextPriority.MEDIUM,
            source_type="learning",
            relevance_score=0.8 if success else 0.4,
        )

        self.store.save(item)
        return item
