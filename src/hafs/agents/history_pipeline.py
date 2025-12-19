"""History Pipeline Agent (AFS Compliant).

Implements the AFS paper's immutable history logging requirements:
- Immutable transaction logging with timestamp, origin, and model version
- Session summarization using fast models
- Embedding generation for semantic search
- Verifiable reconstruction of reasoning processes

Reference: AFS Paper Section IV-A (Persistent Context Repository)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from hafs.agents.base import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class InteractionRecord:
    """Immutable record of a single interaction.

    Following AFS paper requirements for "timestamp, origin, and model version"
    metadata to enable verifiable reconstruction.
    """

    timestamp: str
    session_id: str
    origin: str  # Agent or source that created this record
    model_version: Optional[str] = None

    # Interaction content
    interaction_type: str = "message"  # message, tool_call, tool_result, decision
    role: str = "user"  # user, assistant, system, tool
    content: str = ""

    # Optional structured data
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[dict] = None

    # Cognitive state at time of interaction
    cognitive_state: Optional[dict] = None

    # Embedding vector (stored separately for efficiency)
    embedding_id: Optional[str] = None

    # Integrity hash for immutability verification
    content_hash: str = field(default="", init=False)

    def __post_init__(self):
        """Compute content hash after initialization."""
        if not self.content_hash:
            content_str = f"{self.timestamp}:{self.origin}:{self.content}"
            self.content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InteractionRecord":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k != "content_hash"})


@dataclass
class SessionSummary:
    """Summary of a completed session."""

    session_id: str
    start_time: str
    end_time: str
    duration_seconds: float

    # Content summary
    title: str
    summary: str
    key_topics: list[str]
    key_decisions: list[str]

    # Statistics
    interaction_count: int
    user_message_count: int
    assistant_message_count: int
    tool_call_count: int

    # Models used
    models_used: list[str]

    # File paths
    history_file: str
    embedding_ids: list[str] = field(default_factory=list)


class HistoryPipelineAgent(BaseAgent):
    """Agent responsible for immutable history logging per AFS requirements.

    Implements the AFS paper's Persistent Context Repository patterns:

    1. History Layer (Immutable):
       - Append-only JSONL logs in ~/.context/history/
       - Each record includes timestamp, origin, model version
       - Content hash for integrity verification

    2. Memory Transforms:
       - Summarization of completed sessions
       - Embedding generation for semantic search
       - Indexing for efficient retrieval

    3. Reconstruction:
       - Any past state can be reconstructed from history
       - Verifiable through content hashes

    Example:
        agent = HistoryPipelineAgent()
        await agent.setup()

        # Log an interaction
        await agent.log_interaction(
            session_id="session-123",
            origin="ShellAgent",
            role="user",
            content="What files are in this directory?",
            model_version="gemini-2.5-flash"
        )

        # Query history
        results = await agent.query_history("file operations", limit=10)
    """

    def __init__(self):
        super().__init__(
            "HistoryPipeline",
            "Maintain immutable history logs and enable semantic search over past interactions."
        )

        # History storage paths
        self.history_dir = self.context_root / "history"
        self.history_dir.mkdir(parents=True, exist_ok=True)

        self.summaries_dir = self.context_root / "history" / "summaries"
        self.summaries_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings_dir = self.context_root / "history" / "embeddings"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        # Index for fast lookups
        self.index_file = self.history_dir / "index.json"
        self._index: dict[str, list[str]] = {}  # session_id -> [record_hashes]

        # In-memory embedding cache (for session)
        self._embeddings_cache: dict[str, list[float]] = {}

        # Current session tracking
        self._current_session: Optional[str] = None
        self._session_start_time: Optional[datetime] = None

        # Use fast tier for summarization
        self.model_tier = "fast"

    async def setup(self):
        """Initialize the history pipeline."""
        await super().setup()
        self._load_index()
        logger.info(f"HistoryPipeline initialized with {len(self._index)} indexed sessions")

    def _load_index(self):
        """Load the history index from disk."""
        if self.index_file.exists():
            try:
                self._index = json.loads(self.index_file.read_text())
            except Exception as e:
                logger.warning(f"Failed to load history index: {e}")
                self._index = {}

    def _save_index(self):
        """Save the history index to disk."""
        try:
            self.index_file.write_text(json.dumps(self._index, indent=2))
        except Exception as e:
            logger.error(f"Failed to save history index: {e}")

    def _get_history_file(self, date: Optional[datetime] = None) -> Path:
        """Get the history file for a given date."""
        if date is None:
            date = datetime.now()
        filename = date.strftime("%Y-%m-%d.jsonl")
        return self.history_dir / filename

    def start_session(self, session_id: Optional[str] = None) -> str:
        """Start a new session for tracking.

        Args:
            session_id: Optional custom session ID.

        Returns:
            The session ID.
        """
        if session_id is None:
            session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        self._current_session = session_id
        self._session_start_time = datetime.now()

        if session_id not in self._index:
            self._index[session_id] = []

        logger.info(f"Started session: {session_id}")
        return session_id

    async def log_interaction(
        self,
        session_id: Optional[str] = None,
        origin: str = "unknown",
        interaction_type: str = "message",
        role: str = "user",
        content: str = "",
        model_version: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_input: Optional[dict] = None,
        tool_output: Optional[dict] = None,
        cognitive_state: Optional[dict] = None,
        generate_embedding: bool = False,
    ) -> InteractionRecord:
        """Log an interaction to the immutable history.

        This is the core method implementing AFS's immutable history layer.

        Args:
            session_id: Session to associate with.
            origin: Agent or source creating this record.
            interaction_type: Type of interaction.
            role: Role of the interactor.
            content: The interaction content.
            model_version: Model version if applicable.
            tool_name: Name of tool if tool interaction.
            tool_input: Tool input parameters.
            tool_output: Tool output data.
            cognitive_state: Cognitive state snapshot.
            generate_embedding: Whether to generate and store embedding.

        Returns:
            The created interaction record.
        """
        # Use current session if not specified
        if session_id is None:
            session_id = self._current_session or self.start_session()

        # Create the record
        record = InteractionRecord(
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            origin=origin,
            model_version=model_version,
            interaction_type=interaction_type,
            role=role,
            content=content,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            cognitive_state=cognitive_state,
        )

        # Append to daily JSONL file (immutable append-only)
        history_file = self._get_history_file()
        try:
            with open(history_file, "a") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to write history record: {e}")
            raise

        # Update index
        if session_id not in self._index:
            self._index[session_id] = []
        self._index[session_id].append(record.content_hash)
        self._save_index()

        # Generate embedding if requested
        if generate_embedding and content:
            await self._generate_and_store_embedding(record)

        logger.debug(f"Logged interaction: {record.content_hash} ({interaction_type})")
        return record

    async def _generate_and_store_embedding(self, record: InteractionRecord) -> Optional[str]:
        """Generate and store an embedding for a record."""
        if not self.orchestrator:
            return None

        try:
            # Generate embedding
            embedding = await self.orchestrator.embed_content(record.content)
            if embedding is None:
                return None

            # Store with record hash as ID
            embedding_id = record.content_hash
            embedding_file = self.embeddings_dir / f"{embedding_id}.json"
            embedding_file.write_text(json.dumps({
                "id": embedding_id,
                "session_id": record.session_id,
                "timestamp": record.timestamp,
                "content_preview": record.content[:100],
                "embedding": embedding,
            }))

            # Cache it
            self._embeddings_cache[embedding_id] = embedding

            return embedding_id
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return None

    async def summarize_session(
        self,
        session_id: Optional[str] = None,
    ) -> SessionSummary:
        """Create a summary of a session.

        Implements AFS paper's summarization transform for the Memory layer.

        Args:
            session_id: Session to summarize (defaults to current).

        Returns:
            Session summary.
        """
        session_id = session_id or self._current_session
        if not session_id:
            raise ValueError("No session specified or active")

        # Collect all records for this session
        records = await self.get_session_records(session_id)

        if not records:
            raise ValueError(f"No records found for session {session_id}")

        # Calculate statistics
        start_time = records[0].timestamp
        end_time = records[-1].timestamp

        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(end_time)
        duration = (end_dt - start_dt).total_seconds()

        user_msgs = [r for r in records if r.role == "user"]
        assistant_msgs = [r for r in records if r.role == "assistant"]
        tool_calls = [r for r in records if r.interaction_type == "tool_call"]

        models_used = list(set(r.model_version for r in records if r.model_version))

        # Generate summary using LLM
        summary_prompt = self._build_summary_prompt(records)
        summary_response = await self.generate_thought(summary_prompt)

        # Parse summary response (expecting structured output)
        summary_data = self._parse_summary_response(summary_response, session_id)

        # Create summary object
        summary = SessionSummary(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            title=summary_data.get("title", f"Session {session_id}"),
            summary=summary_data.get("summary", ""),
            key_topics=summary_data.get("topics", []),
            key_decisions=summary_data.get("decisions", []),
            interaction_count=len(records),
            user_message_count=len(user_msgs),
            assistant_message_count=len(assistant_msgs),
            tool_call_count=len(tool_calls),
            models_used=models_used,
            history_file=str(self._get_history_file()),
        )

        # Save summary
        summary_file = self.summaries_dir / f"{session_id}.json"
        summary_file.write_text(json.dumps(asdict(summary), indent=2))

        logger.info(f"Created summary for session {session_id}: {summary.title}")
        return summary

    def _build_summary_prompt(self, records: list[InteractionRecord]) -> str:
        """Build a prompt for summarizing a session."""
        # Collect content samples
        content_samples = []
        for r in records[:50]:  # Limit to prevent token overflow
            if r.content:
                role_prefix = f"[{r.role.upper()}]" if r.role else ""
                content_samples.append(f"{role_prefix} {r.content[:500]}")

        conversation_text = "\n".join(content_samples)

        return f"""Analyze this session and provide a structured summary.

SESSION CONTENT:
{conversation_text}

Respond with a JSON object containing:
{{
    "title": "Brief descriptive title (5-10 words)",
    "summary": "2-3 sentence summary of what was accomplished",
    "topics": ["topic1", "topic2", ...],  // Key topics discussed
    "decisions": ["decision1", ...]  // Key decisions or conclusions made
}}

Respond ONLY with the JSON object, no additional text."""

    def _parse_summary_response(
        self,
        response: str,
        session_id: str,
    ) -> dict[str, Any]:
        """Parse the LLM's summary response."""
        try:
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith("```"):
                # Strip markdown code blocks
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])

            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback if parsing fails
            return {
                "title": f"Session {session_id}",
                "summary": response[:500] if response else "No summary available",
                "topics": [],
                "decisions": [],
            }

    async def get_session_records(
        self,
        session_id: str,
    ) -> list[InteractionRecord]:
        """Get all records for a session.

        Args:
            session_id: Session to retrieve.

        Returns:
            List of interaction records.
        """
        records = []

        # Scan history files for this session
        for history_file in sorted(self.history_dir.glob("*.jsonl")):
            try:
                with open(history_file, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        if data.get("session_id") == session_id:
                            records.append(InteractionRecord.from_dict(data))
            except Exception as e:
                logger.warning(f"Error reading {history_file}: {e}")

        return records

    async def query_history(
        self,
        query: str,
        limit: int = 10,
        session_filter: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        """Query history using semantic search.

        Args:
            query: Search query.
            limit: Maximum results to return.
            session_filter: Optional session ID to filter.
            date_from: Optional start date filter.
            date_to: Optional end date filter.

        Returns:
            List of matching records with scores.
        """
        if not self.orchestrator:
            await self.setup()

        # Generate query embedding
        try:
            query_embedding = await self.orchestrator.embed_content(query)
            if query_embedding is None:
                logger.warning("Failed to generate query embedding")
                return []
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []

        # Load all embeddings
        results = []

        for embedding_file in self.embeddings_dir.glob("*.json"):
            try:
                data = json.loads(embedding_file.read_text())

                # Apply filters
                if session_filter and data.get("session_id") != session_filter:
                    continue

                record_time = datetime.fromisoformat(data.get("timestamp", ""))
                if date_from and record_time < date_from:
                    continue
                if date_to and record_time > date_to:
                    continue

                # Calculate similarity (cosine)
                stored_embedding = data.get("embedding", [])
                if stored_embedding:
                    score = self._cosine_similarity(query_embedding, stored_embedding)
                    results.append({
                        "score": score,
                        "id": data.get("id"),
                        "session_id": data.get("session_id"),
                        "timestamp": data.get("timestamp"),
                        "content_preview": data.get("content_preview"),
                    })
            except Exception as e:
                logger.debug(f"Error processing {embedding_file}: {e}")

        # Sort by score and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    async def get_recent_history(
        self,
        hours: int = 24,
        limit: int = 100,
    ) -> list[InteractionRecord]:
        """Get recent history records.

        Args:
            hours: How many hours back to look.
            limit: Maximum records to return.

        Returns:
            List of recent interaction records.
        """
        cutoff = datetime.now().timestamp() - (hours * 3600)
        records = []

        # Start from most recent files
        for history_file in sorted(self.history_dir.glob("*.jsonl"), reverse=True):
            try:
                with open(history_file, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        record_time = datetime.fromisoformat(data.get("timestamp", ""))

                        if record_time.timestamp() >= cutoff:
                            records.append(InteractionRecord.from_dict(data))

                        if len(records) >= limit:
                            break
            except Exception as e:
                logger.warning(f"Error reading {history_file}: {e}")

            if len(records) >= limit:
                break

        return records

    async def create_daily_digest(self, date: Optional[datetime] = None) -> str:
        """Create a digest of a day's activity.

        Args:
            date: Date to digest (defaults to today).

        Returns:
            Markdown formatted digest.
        """
        if date is None:
            date = datetime.now()

        history_file = self._get_history_file(date)

        if not history_file.exists():
            return f"No history found for {date.strftime('%Y-%m-%d')}"

        # Load records
        records = []
        with open(history_file, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        if not records:
            return f"No interactions logged for {date.strftime('%Y-%m-%d')}"

        # Build digest
        sessions = set(r.get("session_id") for r in records)
        origins = set(r.get("origin") for r in records if r.get("origin"))
        models = set(r.get("model_version") for r in records if r.get("model_version"))

        user_msgs = [r for r in records if r.get("role") == "user"]
        tool_calls = [r for r in records if r.get("interaction_type") == "tool_call"]

        digest = f"""# Daily Digest: {date.strftime('%Y-%m-%d')}

## Statistics
- **Total Interactions**: {len(records)}
- **Sessions**: {len(sessions)}
- **User Messages**: {len(user_msgs)}
- **Tool Calls**: {len(tool_calls)}
- **Active Agents**: {', '.join(origins) if origins else 'None recorded'}
- **Models Used**: {', '.join(models) if models else 'None recorded'}

## Sessions
"""

        for session_id in sessions:
            session_records = [r for r in records if r.get("session_id") == session_id]
            first_record = session_records[0]
            digest += f"\n### {session_id}\n"
            digest += f"- Started: {first_record.get('timestamp', 'unknown')}\n"
            digest += f"- Interactions: {len(session_records)}\n"

            # First user message as preview
            for r in session_records:
                if r.get("role") == "user" and r.get("content"):
                    digest += f"- First Query: {r['content'][:100]}...\n"
                    break

        return digest

    async def run_task(self, task: str = "status") -> dict[str, Any]:
        """Run a history pipeline task.

        Args:
            task: Task to perform:
                - "status": Get pipeline status
                - "digest": Create today's digest
                - "query:SEARCH_TERM": Search history
                - "summarize:SESSION_ID": Summarize a session

        Returns:
            Task result.
        """
        if task == "status":
            total_files = len(list(self.history_dir.glob("*.jsonl")))
            total_embeddings = len(list(self.embeddings_dir.glob("*.json")))
            total_sessions = len(self._index)

            return {
                "status": "operational",
                "history_files": total_files,
                "embeddings": total_embeddings,
                "indexed_sessions": total_sessions,
                "current_session": self._current_session,
            }

        elif task == "digest":
            digest = await self.create_daily_digest()
            return {"digest": digest}

        elif task.startswith("query:"):
            query = task[6:].strip()
            results = await self.query_history(query)
            return {"query": query, "results": results}

        elif task.startswith("summarize:"):
            session_id = task[10:].strip()
            summary = await self.summarize_session(session_id)
            return {"summary": asdict(summary)}

        else:
            return {"error": f"Unknown task: {task}"}

    def end_session(self) -> Optional[str]:
        """End the current session.

        Returns:
            The session ID that was ended.
        """
        session_id = self._current_session
        self._current_session = None
        self._session_start_time = None

        if session_id:
            logger.info(f"Ended session: {session_id}")

        return session_id


# Convenience function for wrapping agent executions
def with_history_logging(agent_class):
    """Decorator to add history logging to an agent class.

    Example:
        @with_history_logging
        class MyAgent(BaseAgent):
            ...
    """
    original_run_task = agent_class.run_task

    async def logged_run_task(self, *args, **kwargs):
        # Get or create history pipeline
        history = getattr(self, "_history_pipeline", None)
        if history is None:
            history = HistoryPipelineAgent()
            await history.setup()
            self._history_pipeline = history

        # Log task start
        await history.log_interaction(
            origin=self.name,
            interaction_type="task_start",
            role="system",
            content=f"Task: {args[0] if args else kwargs}",
        )

        # Run original task
        try:
            result = await original_run_task(self, *args, **kwargs)

            # Log task completion
            await history.log_interaction(
                origin=self.name,
                interaction_type="task_complete",
                role="system",
                content=f"Result: {str(result)[:500]}",
            )

            return result
        except Exception as e:
            # Log task failure
            await history.log_interaction(
                origin=self.name,
                interaction_type="task_error",
                role="system",
                content=f"Error: {str(e)}",
            )
            raise

    agent_class.run_task = logged_run_task
    return agent_class
