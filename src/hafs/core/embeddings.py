"""Shared embedding helpers for knowledge base indexing."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingCheckpoint:
    """Checkpoint for resumable embedding generation."""

    kb_name: str
    total_items: int
    processed_items: int
    last_item_id: str
    timestamp: str
    batch_size: int = 50


class BatchEmbeddingManager:
    """Manages batch embedding generation with checkpointing.

    Features:
    - Resume from last checkpoint on interruption
    - Configurable batch sizes
    - Progress tracking
    - Automatic retry on failures
    - Index bootstrap from existing embeddings
    """

    def __init__(
        self,
        kb_dir: Path,
        orchestrator: Any,
        batch_size: int = 50,
        delay_between_batches: float = 0.5,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        self.kb_dir = kb_dir
        self.orchestrator = orchestrator
        self.batch_size = batch_size
        self.delay = delay_between_batches
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model

        storage_id = self.resolve_storage_id(embedding_provider, embedding_model)
        self.embeddings_dir = self.resolve_embeddings_dir(kb_dir, storage_id)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.resolve_checkpoint_file(kb_dir, storage_id)
        self.index_file = self.resolve_index_file(kb_dir, storage_id)

        self._index: dict[str, str] = {}
        self._load_index()
        if not self._index:
            self._bootstrap_index()

    def _load_index(self) -> None:
        """Load embedding index from disk."""
        if self.index_file.exists():
            try:
                self._index = json.loads(self.index_file.read_text())
            except Exception:
                self._index = {}

    def _save_index(self) -> None:
        """Save embedding index to disk."""
        self.index_file.write_text(json.dumps(self._index, indent=2))

    def _bootstrap_index(self) -> None:
        """Seed index from existing embedding files."""
        if not self.embeddings_dir.exists():
            return

        for emb_file in self.embeddings_dir.glob("*.json"):
            try:
                data = json.loads(emb_file.read_text())
                emb_id = data.get("id")
                if emb_id:
                    self._index[emb_id] = emb_file.name
            except Exception:
                continue

        if self._index:
            self._save_index()

    @staticmethod
    def resolve_storage_id(
        embedding_provider: Optional[str],
        embedding_model: Optional[str],
    ) -> Optional[str]:
        """Build a safe storage identifier for multi-model embeddings."""
        if not embedding_provider and not embedding_model:
            return None

        def _normalize(value: Optional[str]) -> str:
            if value is None:
                return ""
            return getattr(value, "value", str(value))

        parts = []
        provider = _normalize(embedding_provider)
        model = _normalize(embedding_model)
        if provider:
            parts.append(provider)
        if model:
            parts.append(model)

        raw = "-".join(parts)
        slug = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
        return slug or None

    @staticmethod
    def resolve_embeddings_dir(kb_dir: Path, storage_id: Optional[str]) -> Path:
        if storage_id:
            return kb_dir / "embeddings" / storage_id
        return kb_dir / "embeddings"

    @staticmethod
    def resolve_index_file(kb_dir: Path, storage_id: Optional[str]) -> Path:
        if storage_id:
            return kb_dir / f"embedding_index_{storage_id}.json"
        return kb_dir / "embedding_index.json"

    @staticmethod
    def resolve_checkpoint_file(kb_dir: Path, storage_id: Optional[str]) -> Path:
        if storage_id:
            return kb_dir / f"embedding_checkpoint_{storage_id}.json"
        return kb_dir / "embedding_checkpoint.json"

    def _load_checkpoint(self) -> Optional[EmbeddingCheckpoint]:
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            try:
                data = json.loads(self.checkpoint_file.read_text())
                return EmbeddingCheckpoint(**data)
            except Exception:
                return None
        return None

    def _save_checkpoint(self, checkpoint: EmbeddingCheckpoint) -> None:
        """Save checkpoint."""
        self.checkpoint_file.write_text(json.dumps(asdict(checkpoint), indent=2))

    def _clear_checkpoint(self) -> None:
        """Clear checkpoint after completion."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()

    def has_embedding(self, item_id: str) -> bool:
        """Check if item already has embedding."""
        return item_id in self._index

    def get_embedding(self, item_id: str) -> Optional[list[float]]:
        """Get embedding for an item."""
        if item_id not in self._index:
            return None

        emb_file = self.embeddings_dir / self._index[item_id]
        if emb_file.exists():
            try:
                data = json.loads(emb_file.read_text())
                return data.get("embedding")
            except Exception:
                return None
        return None

    async def _embed_text(self, text: str) -> Optional[list[float]]:
        """Generate embedding using the configured orchestrator."""
        if hasattr(self.orchestrator, "embed"):
            return await self.orchestrator.embed(
                text,
                provider=self.embedding_provider,
                model=self.embedding_model,
            )
        if hasattr(self.orchestrator, "embed_content"):
            return await self.orchestrator.embed_content(text)
        return None

    async def generate_embeddings(
        self,
        items: list[tuple[str, str]],
        kb_name: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict[str, int]:
        """Generate embeddings with checkpointing.

        Args:
            items: List of (item_id, text) tuples.
            kb_name: Name of knowledge base.
            progress_callback: Optional callback(processed, total).

        Returns:
            Statistics dict.
        """
        checkpoint = self._load_checkpoint()
        start_index = 0

        if checkpoint and checkpoint.kb_name == kb_name:
            for i, (item_id, _) in enumerate(items):
                if item_id == checkpoint.last_item_id:
                    start_index = i + 1
                    break
            logger.info("Resuming from checkpoint: %s/%s", start_index, len(items))

        pending = [
            (item_id, text)
            for item_id, text in items[start_index:]
            if not self.has_embedding(item_id)
        ]

        if not pending:
            logger.info("All items already have embeddings")
            self._clear_checkpoint()
            return {"processed": 0, "skipped": len(items), "errors": 0}

        logger.info(
            "Generating embeddings for %s items (batch size: %s)",
            len(pending),
            self.batch_size,
        )

        processed = 0
        errors = 0

        for batch_start in range(0, len(pending), self.batch_size):
            batch = pending[batch_start:batch_start + self.batch_size]

            for item_id, text in batch:
                try:
                    embedding = await self._embed_text(text)
                    if embedding:
                        emb_hash = hashlib.md5(item_id.encode()).hexdigest()[:12]
                        emb_filename = f"{emb_hash}.json"
                        emb_file = self.embeddings_dir / emb_filename

                        emb_file.write_text(json.dumps({
                            "id": item_id,
                            "text_preview": text[:200],
                            "embedding": embedding,
                            "embedding_provider": self.embedding_provider,
                            "embedding_model": self.embedding_model,
                            "created": datetime.now().isoformat(),
                        }))

                        self._index[item_id] = emb_filename
                        processed += 1
                    else:
                        errors += 1
                except Exception as exc:
                    logger.warning("Embedding failed for %s: %s", item_id, exc)
                    errors += 1

            checkpoint = EmbeddingCheckpoint(
                kb_name=kb_name,
                total_items=len(items),
                processed_items=start_index + batch_start + len(batch),
                last_item_id=batch[-1][0] if batch else "",
                timestamp=datetime.now().isoformat(),
                batch_size=self.batch_size,
            )
            self._save_checkpoint(checkpoint)
            self._save_index()

            if progress_callback:
                progress_callback(start_index + batch_start + len(batch), len(items))

            await asyncio.sleep(self.delay)

        self._clear_checkpoint()
        self._save_index()

        return {
            "processed": processed,
            "skipped": len(items) - len(pending),
            "errors": errors,
        }
