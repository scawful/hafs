"""Session-level summaries and embeddings for AFS history."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from core.embeddings import BatchEmbeddingManager
from core.history.logger import HistoryLogger
from core.history.models import HistoryEntry, OperationType, SessionSummary, SessionStats
from core.orchestrator_v2 import UnifiedOrchestrator, TaskTier


class HistorySessionSummaryIndex:
    """Build and query session summaries with embeddings."""

    def __init__(
        self,
        context_root: Path,
        orchestrator: Optional[UnifiedOrchestrator] = None,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        self.context_root = context_root
        self.history_dir = context_root / "history"
        self.sessions_dir = self.history_dir / "sessions"
        storage_id = BatchEmbeddingManager.resolve_storage_id(
            embedding_provider,
            embedding_model,
        )
        self.summaries_dir = self.history_dir / "summaries"
        if storage_id:
            self.summaries_dir = self.summaries_dir / storage_id
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        self._orchestrator = orchestrator
        self._logger = HistoryLogger(self.history_dir)
        self._embedding_provider = embedding_provider
        self._embedding_model = embedding_model

    async def _get_orchestrator(self) -> UnifiedOrchestrator:
        if self._orchestrator is None:
            self._orchestrator = UnifiedOrchestrator()
        await self._orchestrator.initialize()
        return self._orchestrator

    def _summary_path(self, session_id: str) -> Path:
        return self.summaries_dir / f"{session_id}.json"

    def _iter_session_ids(self) -> Iterable[str]:
        if self.sessions_dir.exists():
            for file_path in self.sessions_dir.glob("*.json"):
                yield file_path.stem
            return

        # Fallback: scan history logs
        seen: set[str] = set()
        for history_file in sorted(self.history_dir.glob("*.jsonl")):
            try:
                for line in history_file.read_text(encoding="utf-8").splitlines():
                    entry = HistoryEntry.model_validate_json(line)
                    if entry.session_id not in seen:
                        seen.add(entry.session_id)
                        yield entry.session_id
            except Exception:
                continue

    def _build_summary_prompt(self, entries: list[HistoryEntry]) -> str:
        samples: list[str] = []
        for entry in entries[:40]:
            op = entry.operation
            label = op.type.value.upper()
            if op.type == OperationType.USER_INPUT:
                msg = op.input.get("message") if isinstance(op.input, dict) else ""
                samples.append(f"[{label}] {msg}")
            elif op.type == OperationType.AGENT_MESSAGE:
                msg = op.input.get("message") if isinstance(op.input, dict) else ""
                if not msg and isinstance(op.output, str):
                    msg = op.output
                samples.append(f"[{label}] {msg}")
            elif op.type == OperationType.TOOL_CALL:
                samples.append(f"[{label}] {op.name}")

        transcript = "\n".join(s for s in samples if s)
        return (
            "Summarize this session for future retrieval.\n\n"
            "Provide JSON with keys: title, summary, topics, decisions.\n\n"
            f"SESSION CONTENT:\n{transcript}\n\n"
            "Return only JSON."
        )

    def _fallback_summary(self, entries: list[HistoryEntry]) -> dict[str, Any]:
        user_messages = [e for e in entries if e.operation.type == OperationType.USER_INPUT]
        tool_calls = [e for e in entries if e.operation.type == OperationType.TOOL_CALL]
        first_user = ""
        if user_messages:
            first_user = user_messages[0].operation.input.get("message", "")
        tool_names = sorted({e.operation.name for e in tool_calls})
        summary = "Session captured without LLM summary."
        if first_user:
            summary += f" First request: {first_user[:160]}"
        return {
            "title": "Session summary",
            "summary": summary,
            "topics": tool_names[:5],
            "decisions": [],
        }

    async def summarize_session(self, session_id: str) -> Optional[SessionSummary]:
        entries = self._logger.get_session_entries(session_id)
        if not entries:
            return None

        entries.sort(key=lambda e: e.timestamp)
        start = entries[0].timestamp
        end = entries[-1].timestamp
        duration_ms = 0
        try:
            start_dt = datetime.fromisoformat(start)
            end_dt = datetime.fromisoformat(end)
            duration_ms = int((end_dt - start_dt).total_seconds() * 1000)
        except Exception:
            duration_ms = 0

        stats = SessionStats(
            operation_count=len(entries),
            duration_ms=duration_ms,
            files_modified=sorted({f for e in entries for f in e.metadata.files_touched}),
            tools_used=sorted({e.operation.name for e in entries if e.operation.type == OperationType.TOOL_CALL}),
        )

        summary_data: dict[str, Any]
        try:
            orchestrator = await self._get_orchestrator()
            prompt = self._build_summary_prompt(entries)
            response = await orchestrator.generate(prompt=prompt, tier=TaskTier.FAST)
            if response.content:
                summary_data = json.loads(self._strip_code_block(response.content))
            else:
                summary_data = self._fallback_summary(entries)
        except Exception:
            summary_data = self._fallback_summary(entries)

        summary_text = summary_data.get("summary", "")
        embedding: Optional[list[float]] = None
        try:
            orchestrator = await self._get_orchestrator()
            embedding = await orchestrator.embed(
                summary_text,
                provider=self._embedding_provider,
                model=self._embedding_model,
            )
        except Exception:
            embedding = None

        extensions = {
            "title": summary_data.get("title"),
            "topics": summary_data.get("topics", []),
            "decisions": summary_data.get("decisions", []),
        }
        if self._embedding_provider:
            extensions["embedding_provider"] = self._embedding_provider
        if self._embedding_model:
            extensions["embedding_model"] = self._embedding_model

        summary = SessionSummary(
            session_id=session_id,
            project_id=entries[0].project_id,
            created_at=datetime.now().isoformat(),
            summary=summary_text,
            entities=[],
            stats=stats,
            embedding=embedding,
            embedding_model=self._embedding_model if embedding else None,
            extensions=extensions,
        )

        self._summary_path(session_id).write_text(
            json.dumps(summary.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )

        return summary

    async def index_missing_summaries(self, limit: int = 20) -> int:
        created = 0
        for session_id in self._iter_session_ids():
            if self._summary_path(session_id).exists():
                continue
            summary = await self.summarize_session(session_id)
            if summary:
                created += 1
            if created >= limit:
                break
        return created

    async def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
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

        results: list[dict[str, Any]] = []
        for summary_file in self.summaries_dir.glob("*.json"):
            try:
                data = json.loads(summary_file.read_text(encoding="utf-8"))
                embedding = data.get("embedding")
                if not embedding:
                    continue
                score = self._cosine_similarity(query_embedding, embedding)
                results.append(
                    {
                        "score": score,
                        "session_id": data.get("session_id"),
                        "created_at": data.get("created_at"),
                        "summary": data.get("summary", ""),
                        "title": data.get("extensions", {}).get("title"),
                        "topics": data.get("extensions", {}).get("topics", []),
                        "decisions": data.get("extensions", {}).get("decisions", []),
                    }
                )
            except Exception:
                continue

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def status(self) -> dict[str, int]:
        summaries = len(list(self.summaries_dir.glob("*.json")))
        sessions = len(list(self._iter_session_ids()))
        return {"sessions": sessions, "summaries": summaries}

    @staticmethod
    def _strip_code_block(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) > 2:
                return "\n".join(lines[1:-1])
        return text

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
