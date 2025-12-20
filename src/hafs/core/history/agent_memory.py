"""Agent-scoped memory system with temporal awareness.

Provides individual agents with their own memory indices, enabling:
- Agent-specific history retrieval ("what did I do last time?")
- Temporal bucketing (recent vs old memories)
- Recency-weighted search
- Session consolidation and summarization
- Cross-agent memory sharing when needed

Architecture:
- Each agent has a memory index stored at: history/agents/{agent_id}/
- Memory entries include: embeddings, summaries, key decisions
- Temporal buckets: working (current session), recent (last 7 days), archive
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Literal

from hafs.core.history.models import HistoryEntry, OperationType
from hafs.core.orchestrator_v2 import UnifiedOrchestrator


@dataclass
class MemoryEntry:
    """A single memory entry for an agent."""

    id: str
    agent_id: str
    timestamp: str
    memory_type: Literal["decision", "interaction", "learning", "error", "insight"]
    content: str
    context: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)
    importance: float = 0.5  # 0-1 scale, affects retention
    session_id: Optional[str] = None
    related_entries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "memory_type": self.memory_type,
            "content": self.content,
            "context": self.context,
            "embedding": self.embedding,
            "importance": self.importance,
            "session_id": self.session_id,
            "related_entries": self.related_entries,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            timestamp=data["timestamp"],
            memory_type=data["memory_type"],
            content=data["content"],
            context=data.get("context", {}),
            embedding=data.get("embedding", []),
            importance=data.get("importance", 0.5),
            session_id=data.get("session_id"),
            related_entries=data.get("related_entries", []),
        )


@dataclass
class SessionSummary:
    """Summary of an agent's session."""

    session_id: str
    agent_id: str
    start_time: str
    end_time: str
    summary: str
    key_decisions: list[str]
    tools_used: list[str]
    topics: list[str]
    outcome: Literal["success", "partial", "failed", "interrupted"]
    embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "summary": self.summary,
            "key_decisions": self.key_decisions,
            "tools_used": self.tools_used,
            "topics": self.topics,
            "outcome": self.outcome,
            "embedding": self.embedding,
        }


class AgentMemory:
    """Memory system for a single agent with temporal awareness."""

    # Temporal bucket thresholds
    WORKING_MEMORY_HOURS = 4  # Current working session
    RECENT_MEMORY_DAYS = 7   # Recent memories
    # Older = archive

    # Recency decay factor (higher = faster decay)
    RECENCY_DECAY = 0.1

    def __init__(
        self,
        agent_id: str,
        context_root: Path,
        orchestrator: Optional[UnifiedOrchestrator] = None,
    ) -> None:
        self.agent_id = agent_id
        self.context_root = context_root
        self.memory_dir = context_root / "history" / "agents" / agent_id
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.entries_dir = self.memory_dir / "entries"
        self.entries_dir.mkdir(exist_ok=True)

        self.summaries_dir = self.memory_dir / "summaries"
        self.summaries_dir.mkdir(exist_ok=True)

        self._orchestrator = orchestrator
        self._entries_cache: dict[str, MemoryEntry] = {}
        self._summaries_cache: dict[str, SessionSummary] = {}

    async def _get_orchestrator(self) -> UnifiedOrchestrator:
        if self._orchestrator is None:
            self._orchestrator = UnifiedOrchestrator()
        await self._orchestrator.initialize()
        return self._orchestrator

    def _load_entries(self) -> None:
        """Load all memory entries from disk."""
        if self._entries_cache:
            return

        for entry_file in self.entries_dir.glob("*.json"):
            try:
                data = json.loads(entry_file.read_text())
                entry = MemoryEntry.from_dict(data)
                self._entries_cache[entry.id] = entry
            except Exception:
                continue

    def _load_summaries(self) -> None:
        """Load all session summaries from disk."""
        if self._summaries_cache:
            return

        for summary_file in self.summaries_dir.glob("*.json"):
            try:
                data = json.loads(summary_file.read_text())
                summary = SessionSummary(**data)
                self._summaries_cache[summary.session_id] = summary
            except Exception:
                continue

    async def remember(
        self,
        content: str,
        memory_type: Literal["decision", "interaction", "learning", "error", "insight"],
        context: Optional[dict[str, Any]] = None,
        importance: float = 0.5,
        session_id: Optional[str] = None,
    ) -> MemoryEntry:
        """Store a new memory for this agent.

        Args:
            content: The memory content to store.
            memory_type: Type of memory (decision, interaction, learning, error, insight).
            context: Additional context about the memory.
            importance: How important this memory is (0-1).
            session_id: Current session ID.

        Returns:
            The created memory entry.
        """
        from ulid import ULID

        entry_id = str(ULID())
        timestamp = datetime.now().isoformat()

        # Generate embedding
        orchestrator = await self._get_orchestrator()
        embedding = await orchestrator.embed(content)

        entry = MemoryEntry(
            id=entry_id,
            agent_id=self.agent_id,
            timestamp=timestamp,
            memory_type=memory_type,
            content=content,
            context=context or {},
            embedding=embedding or [],
            importance=importance,
            session_id=session_id,
        )

        # Save to disk
        entry_file = self.entries_dir / f"{entry_id}.json"
        entry_file.write_text(json.dumps(entry.to_dict(), indent=2))

        # Update cache
        self._entries_cache[entry_id] = entry

        return entry

    async def recall(
        self,
        query: str,
        limit: int = 10,
        memory_types: Optional[list[str]] = None,
        temporal_bucket: Optional[Literal["working", "recent", "archive", "all"]] = "all",
        min_importance: float = 0.0,
        recency_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Search agent's memory with temporal awareness.

        Args:
            query: Search query.
            limit: Max results to return.
            memory_types: Filter by memory types.
            temporal_bucket: Filter by time bucket (working, recent, archive, all).
            min_importance: Minimum importance threshold.
            recency_weight: How much to weight recency (0-1). Higher = prefer recent.

        Returns:
            List of matching memories with scores.
        """
        self._load_entries()

        if not self._entries_cache:
            return []

        orchestrator = await self._get_orchestrator()
        query_embedding = await orchestrator.embed(query)

        if not query_embedding:
            return []

        now = datetime.now()
        results = []

        for entry in self._entries_cache.values():
            # Apply filters
            if memory_types and entry.memory_type not in memory_types:
                continue
            if entry.importance < min_importance:
                continue

            # Temporal bucket filter
            entry_time = datetime.fromisoformat(entry.timestamp)
            age_hours = (now - entry_time).total_seconds() / 3600

            if temporal_bucket == "working" and age_hours > self.WORKING_MEMORY_HOURS:
                continue
            elif temporal_bucket == "recent":
                if age_hours <= self.WORKING_MEMORY_HOURS or age_hours > self.RECENT_MEMORY_DAYS * 24:
                    continue
            elif temporal_bucket == "archive" and age_hours <= self.RECENT_MEMORY_DAYS * 24:
                continue

            # Calculate semantic similarity
            if not entry.embedding:
                continue
            semantic_score = self._cosine_similarity(query_embedding, entry.embedding)

            # Calculate recency score (exponential decay)
            recency_score = math.exp(-self.RECENCY_DECAY * (age_hours / 24))

            # Combine scores
            final_score = (1 - recency_weight) * semantic_score + recency_weight * recency_score

            # Boost by importance
            final_score *= (0.5 + 0.5 * entry.importance)

            results.append({
                "score": final_score,
                "semantic_score": semantic_score,
                "recency_score": recency_score,
                "entry": entry.to_dict(),
                "age_hours": age_hours,
                "temporal_bucket": self._get_temporal_bucket(age_hours),
            })

        # Sort by final score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def _get_temporal_bucket(self, age_hours: float) -> str:
        """Get the temporal bucket for an entry based on age."""
        if age_hours <= self.WORKING_MEMORY_HOURS:
            return "working"
        elif age_hours <= self.RECENT_MEMORY_DAYS * 24:
            return "recent"
        else:
            return "archive"

    async def recall_session(self, session_id: str) -> list[MemoryEntry]:
        """Recall all memories from a specific session."""
        self._load_entries()

        return [
            entry for entry in self._entries_cache.values()
            if entry.session_id == session_id
        ]

    async def summarize_session(
        self,
        session_id: str,
        entries: Optional[list[HistoryEntry]] = None,
    ) -> SessionSummary:
        """Generate a summary of a session for long-term memory.

        Args:
            session_id: Session to summarize.
            entries: History entries from the session (if available).

        Returns:
            Session summary with embedding.
        """
        orchestrator = await self._get_orchestrator()

        # Get session memories
        session_memories = await self.recall_session(session_id)

        # Build context for summarization
        memory_texts = [
            f"[{m.memory_type}] {m.content}"
            for m in session_memories
        ]

        # If we have history entries, extract more context
        tools_used = set()
        topics = []
        if entries:
            for entry in entries:
                if entry.operation.type == OperationType.TOOL_CALL:
                    tools_used.add(entry.operation.name)

        # Generate summary with LLM
        context_text = "\n".join(memory_texts[:40])  # Limit context

        prompt = f"""Summarize this agent session concisely (2-3 sentences).
Focus on: what was accomplished, key decisions made, and outcomes.

Agent: {self.agent_id}
Session memories:
{context_text}

Return JSON:
{{"summary": "...", "key_decisions": ["..."], "topics": ["..."], "outcome": "success|partial|failed|interrupted"}}
"""

        try:
            response = await orchestrator.generate(prompt, tier="fast")
            data = json.loads(response)
            summary_text = data.get("summary", "Session completed.")
            key_decisions = data.get("key_decisions", [])
            topics = data.get("topics", [])
            outcome = data.get("outcome", "partial")
        except Exception:
            summary_text = f"Session with {len(session_memories)} activities."
            key_decisions = []
            outcome = "partial"

        # Generate embedding for the summary
        embedding = await orchestrator.embed(summary_text) or []

        # Determine time bounds
        timestamps = [m.timestamp for m in session_memories]
        start_time = min(timestamps) if timestamps else datetime.now().isoformat()
        end_time = max(timestamps) if timestamps else datetime.now().isoformat()

        summary = SessionSummary(
            session_id=session_id,
            agent_id=self.agent_id,
            start_time=start_time,
            end_time=end_time,
            summary=summary_text,
            key_decisions=key_decisions,
            tools_used=list(tools_used),
            topics=topics,
            outcome=outcome,
            embedding=embedding,
        )

        # Save summary
        summary_file = self.summaries_dir / f"{session_id}.json"
        summary_file.write_text(json.dumps(summary.to_dict(), indent=2))
        self._summaries_cache[session_id] = summary

        return summary

    async def recall_similar_sessions(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Find sessions similar to a query.

        Useful for: "Have I dealt with something like this before?"
        """
        self._load_summaries()

        if not self._summaries_cache:
            return []

        orchestrator = await self._get_orchestrator()
        query_embedding = await orchestrator.embed(query)

        if not query_embedding:
            return []

        results = []
        for summary in self._summaries_cache.values():
            if not summary.embedding:
                continue

            score = self._cosine_similarity(query_embedding, summary.embedding)
            results.append({
                "score": score,
                "session": summary.to_dict(),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics for this agent."""
        self._load_entries()
        self._load_summaries()

        now = datetime.now()
        working_count = 0
        recent_count = 0
        archive_count = 0

        for entry in self._entries_cache.values():
            entry_time = datetime.fromisoformat(entry.timestamp)
            age_hours = (now - entry_time).total_seconds() / 3600
            bucket = self._get_temporal_bucket(age_hours)
            if bucket == "working":
                working_count += 1
            elif bucket == "recent":
                recent_count += 1
            else:
                archive_count += 1

        # Count by type
        type_counts = {}
        for entry in self._entries_cache.values():
            type_counts[entry.memory_type] = type_counts.get(entry.memory_type, 0) + 1

        return {
            "agent_id": self.agent_id,
            "total_entries": len(self._entries_cache),
            "total_summaries": len(self._summaries_cache),
            "working_memory": working_count,
            "recent_memory": recent_count,
            "archive_memory": archive_count,
            "by_type": type_counts,
        }

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


class AgentMemoryManager:
    """Manages memory for all agents."""

    def __init__(
        self,
        context_root: Path,
        orchestrator: Optional[UnifiedOrchestrator] = None,
    ) -> None:
        self.context_root = context_root
        self.agents_dir = context_root / "history" / "agents"
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self._orchestrator = orchestrator
        self._agents: dict[str, AgentMemory] = {}

    def get_agent_memory(self, agent_id: str) -> AgentMemory:
        """Get or create memory for an agent."""
        if agent_id not in self._agents:
            self._agents[agent_id] = AgentMemory(
                agent_id=agent_id,
                context_root=self.context_root,
                orchestrator=self._orchestrator,
            )
        return self._agents[agent_id]

    def list_agents(self) -> list[str]:
        """List all agents with memory."""
        return [d.name for d in self.agents_dir.iterdir() if d.is_dir()]

    async def cross_agent_search(
        self,
        query: str,
        agent_ids: Optional[list[str]] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search across multiple agents' memories.

        Useful for shared learning and coordination.
        """
        if agent_ids is None:
            agent_ids = self.list_agents()

        all_results = []
        for agent_id in agent_ids:
            memory = self.get_agent_memory(agent_id)
            results = await memory.recall(query, limit=limit // len(agent_ids) + 1)
            all_results.extend(results)

        # Sort by score and deduplicate
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:limit]

    def get_all_stats(self) -> dict[str, Any]:
        """Get memory statistics for all agents."""
        stats = {}
        for agent_id in self.list_agents():
            memory = self.get_agent_memory(agent_id)
            stats[agent_id] = memory.get_stats()
        return stats
