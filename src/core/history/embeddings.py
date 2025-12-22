"""History embeddings index for semantic search over AFS history."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from core.embeddings import BatchEmbeddingManager
from core.history.models import HistoryEntry, OperationType
from core.orchestrator_v2 import UnifiedOrchestrator
from core.similarity import cosine_similarity


@dataclass(frozen=True)
class HistoryEmbedding:
    """Embedding record for a single history entry."""

    entry_id: str
    session_id: str
    timestamp: str
    operation_type: str
    name: str
    preview: str
    embedding: list[float]
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None


class HistoryEmbeddingIndex:
    """Build and query embeddings for history entries."""

    DEFAULT_TYPES = {
        OperationType.USER_INPUT,
        OperationType.AGENT_MESSAGE,
        OperationType.TOOL_CALL,
        OperationType.THOUGHT_TRACE,  # Include Gemini 3 reasoning traces
    }

    def __init__(
        self,
        context_root: Path,
        orchestrator: Optional[UnifiedOrchestrator] = None,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        self.context_root = context_root
        self.history_dir = context_root / "history"
        storage_id = BatchEmbeddingManager.resolve_storage_id(
            embedding_provider,
            embedding_model,
        )
        self.embeddings_dir = self.history_dir / "embeddings"
        if storage_id:
            self.embeddings_dir = self.embeddings_dir / storage_id
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self._orchestrator = orchestrator
        self._embedding_provider = embedding_provider
        self._embedding_model = embedding_model

    async def _get_orchestrator(self) -> UnifiedOrchestrator:
        if self._orchestrator is None:
            self._orchestrator = UnifiedOrchestrator()
        await self._orchestrator.initialize()
        return self._orchestrator

    def _embedding_path(self, entry_id: str) -> Path:
        return self.embeddings_dir / f"{entry_id}.json"

    def _iter_history_files(self) -> Iterable[Path]:
        if not self.history_dir.exists():
            return []
        files = sorted(self.history_dir.glob("*.jsonl"))
        files.reverse()
        return files

    def _parse_entry(self, line: str) -> Optional[HistoryEntry]:
        try:
            return HistoryEntry.model_validate_json(line)
        except Exception:
            return None

    def _entry_to_text(self, entry: HistoryEntry) -> Optional[str]:
        op = entry.operation
        if op.type == OperationType.USER_INPUT:
            message = op.input.get("message") if isinstance(op.input, dict) else None
            return self._normalize_text(message)
        if op.type == OperationType.AGENT_MESSAGE:
            message = op.input.get("message") if isinstance(op.input, dict) else None
            if not message and isinstance(op.output, str):
                message = op.output
            return self._normalize_text(message)
        if op.type == OperationType.TOOL_CALL:
            parts = [f"Tool: {op.name}"]
            if op.input:
                parts.append(f"Input: {self._summarize_value(op.input)}")
            if op.output:
                parts.append(f"Output: {self._summarize_value(op.output)}")
            return self._normalize_text("\n".join(parts))
        if op.type == OperationType.THOUGHT_TRACE:
            # Extract thought content for embedding
            thought_content = None
            if isinstance(op.input, dict):
                thought_content = op.input.get("thought_content")
            if not thought_content and isinstance(op.output, str):
                thought_content = op.output
            if thought_content:
                # Prefix with model info for context
                provider = op.input.get("provider", "unknown") if isinstance(op.input, dict) else "unknown"
                model = op.input.get("model", "unknown") if isinstance(op.input, dict) else "unknown"
                parts = [f"Thought ({provider}:{model}):", thought_content]
                return self._normalize_text("\n".join(parts))
        return None

    @staticmethod
    def _summarize_value(value: Any, limit: int = 800) -> str:
        try:
            text = json.dumps(value, ensure_ascii=True)
        except Exception:
            text = str(value)
        text = " ".join(text.split())
        if len(text) > limit:
            return text[: limit - 3] + "..."
        return text

    @staticmethod
    def _normalize_text(text: Any, limit: int = 1200) -> Optional[str]:
        if not text:
            return None
        if not isinstance(text, str):
            text = str(text)
        normalized = " ".join(text.split())
        if not normalized:
            return None
        if len(normalized) > limit:
            return normalized[: limit - 3] + "..."
        return normalized

    async def index_new_entries(
        self,
        *,
        limit: int = 200,
        include_types: Optional[set[OperationType]] = None,
    ) -> int:
        """Index new history entries into embeddings.

        Args:
            limit: Max number of new entries to embed in one run.
            include_types: Operation types to include.

        Returns:
            Number of embeddings created.
        """
        include_types = include_types or self.DEFAULT_TYPES
        created = 0

        orchestrator = await self._get_orchestrator()

        for history_file in self._iter_history_files():
            try:
                lines = history_file.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue

            for line in lines:
                entry = self._parse_entry(line)
                if not entry:
                    continue
                if entry.operation.type not in include_types:
                    continue
                if self._embedding_path(entry.id).exists():
                    continue

                text = self._entry_to_text(entry)
                if not text:
                    continue

                embedding = await orchestrator.embed(
                    text,
                    provider=self._embedding_provider,
                    model=self._embedding_model,
                )
                if not embedding:
                    continue

                preview = text[:200]
                payload = {
                    "entry_id": entry.id,
                    "session_id": entry.session_id,
                    "timestamp": entry.timestamp,
                    "operation_type": entry.operation.type.value,
                    "name": entry.operation.name,
                    "preview": preview,
                    "embedding": embedding,
                    "embedding_provider": self._embedding_provider,
                    "embedding_model": self._embedding_model,
                }

                self._embedding_path(entry.id).write_text(
                    json.dumps(payload, indent=2),
                    encoding="utf-8",
                )
                created += 1

                if created >= limit:
                    return created

        return created

    def _load_embeddings(self) -> list[HistoryEmbedding]:
        embeddings: list[HistoryEmbedding] = []
        for file_path in self.embeddings_dir.glob("*.json"):
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                embeddings.append(
                    HistoryEmbedding(
                        entry_id=data.get("entry_id", ""),
                        session_id=data.get("session_id", ""),
                        timestamp=data.get("timestamp", ""),
                        operation_type=str(data.get("operation_type", "")),
                        name=str(data.get("name", "")),
                        preview=str(data.get("preview", "")),
                        embedding=list(data.get("embedding", [])),
                        embedding_provider=data.get("embedding_provider"),
                        embedding_model=data.get("embedding_model"),
                    )
                )
            except Exception:
                continue
        return embeddings

    async def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Semantic search over history embeddings."""
        query = query.strip()
        if not query:
            return []

        orchestrator = await self._get_orchestrator()
        query_embedding = await orchestrator.embed(
            query,
            provider=self._embedding_provider,
            model=self._embedding_model,
        )
        if not query_embedding:
            return []

        results = []
        for record in self._load_embeddings():
            score = cosine_similarity(query_embedding, record.embedding)
            results.append(
                {
                    "score": score,
                    "entry_id": record.entry_id,
                    "session_id": record.session_id,
                    "timestamp": record.timestamp,
                    "operation_type": record.operation_type,
                    "name": record.name,
                    "preview": record.preview,
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def status(self) -> dict[str, int]:
        history_files = len(list(self.history_dir.glob("*.jsonl")))
        embeddings = len(list(self.embeddings_dir.glob("*.json")))
        return {
            "history_files": history_files,
            "embeddings": embeddings,
        }
