"""ALTTP Embedding and Similarity Search Specialists.

Specialized agents for managing and searching ALTTP knowledge base embeddings.
Provides high-performance vector search and semantic clustering for ASM symbols.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from agents.core import BaseAgent
from agents.knowledge.alttp import ALTTPKnowledgeBase
from hafs.core.embeddings import BatchEmbeddingManager
from hafs.core.orchestrator_v2 import UnifiedOrchestrator, TaskTier, Provider

logger = logging.getLogger(__name__)


class ALTTPEmbeddingSpecialist(BaseAgent):
    """Specialist for ALTTP semantic search and embeddings.

    Maintains the vector database for ALTTP knowledge across all versions
    and provides similarity search, clustering, and semantic grouping.

    Example:
        specialist = ALTTPEmbeddingSpecialist()
        await specialist.setup()

        # Find similar routines
        similar = await specialist.find_similar("Module07_Underworld")

        # Cluster symbols by purpose
        clusters = await specialist.cluster_symbols()
    """

    def __init__(self, kb: Optional[ALTTPKnowledgeBase] = None):
        super().__init__(
            "ALTTPEmbeddingSpecialist",
            "Expert in ALTTP vector embeddings and semantic similarity search."
        )

        self._kb = kb
        self._orchestrator = None
        self._embedding_manager = None

        self.kb_dir = self.context_root / "knowledge" / "alttp"
        self.kb_dir.mkdir(parents=True, exist_ok=True)

    async def setup(self):
        """Initialize the embedding specialist."""
        await super().setup()

        self._orchestrator = UnifiedOrchestrator()
        await self._orchestrator.initialize()

        if not self._kb:
            self._kb = ALTTPKnowledgeBase()
            await self._kb.setup()

        self._embedding_manager = BatchEmbeddingManager(
            self.kb_dir,
            self._orchestrator,
        )

        logger.info("ALTTPEmbeddingSpecialist ready")

    async def find_similar(
        self,
        item_name: str,
        limit: int = 5,
        item_type: str = "routine",
    ) -> list[dict[str, Any]]:
        """Find items similar to a given routine or symbol.

        Args:
            item_name: Name of the reference item.
            limit: Maximum matches to return.
            item_type: "routine" or "symbol".

        Returns:
            List of similar items with scores.
        """
        # Get embedding for reference item
        emb_id = f"{item_type}:{item_name}"
        ref_embedding = self._kb._embeddings.get(emb_id)

        if not ref_embedding:
            # Try to search for it first
            results = await self._kb.search(item_name, limit=1)
            if not results:
                return []

            # Use found item's embedding
            # Note: ALTTPKnowledgeBase.search returns scores but we need the raw embedding
            # For simplicity, we'll re-search by query if embedding not found
            query = f"The {item_type} {item_name} in ALTTP"
            return await self._kb.search(query, limit=limit)

        # Standard vector search (cosine similarity)
        results = []
        for other_id, other_emb in self._kb._embeddings.items():
            if other_id == emb_id:
                continue

            score = self._kb._cosine_similarity(ref_embedding, other_emb)
            results.append({
                "id": other_id,
                "score": score,
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        # Map back to KB data
        enriched = []
        for r in results[:limit]:
            if r["id"].startswith("routine:"):
                name = r["id"][8:]
                if name in self._kb._routines:
                    item = self._kb._routines[name]
                    enriched.append({
                        "type": "routine",
                        "name": name,
                        "description": item.description,
                        "score": r["score"],
                    })
            elif ":" in r["id"]:
                if r["id"] in self._kb._symbols:
                    item = self._kb._symbols[r["id"]]
                    enriched.append({
                        "type": "symbol",
                        "name": item.name,
                        "category": item.category,
                        "description": item.description,
                        "score": r["score"],
                    })

        return enriched

    async def cluster_symbols(self, category: str = "wram") -> dict[str, list[str]]:
        """Group symbols by semantic similarity into clusters.

        Args:
            category: Symbol category to cluster.

        Returns:
            Dict mapping cluster labels to lists of symbol names.
        """
        # This would typically use K-means or similar on server-side
        # Here we'll use LLM to identify themes from top symbols
        symbols = [s for s in self._kb._symbols.values() if category in s.category][:30]

        prompt = f"""Group these ALTTP {category} symbols into semantic clusters based on their names and descriptions:

{chr(10).join([f"- {s.name}: {s.description}" for s in symbols])}

Provide a JSON mapping of cluster_path -> [symbol_names]."""

        try:
            result = await self._orchestrator.generate(
                prompt=prompt,
                tier=TaskTier.RESEARCH,
                provider=Provider.GEMINI,
            )
            # Try to parse JSON from response
            match = re.search(r"\{.*\}", result.content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return {"raw_analysis": result.content}
        except Exception as e:
            return {"error": f"Clustering failed: {e}"}

    async def run_task(self, task: str) -> dict[str, Any]:
        """Run an embedding task."""
        if task.startswith("similar:"):
            name = task[8:].strip()
            return {"results": await self.find_similar(name)}
        elif task.startswith("cluster:"):
            cat = task[8:].strip() or "wram"
            return {"clusters": await self.cluster_symbols(cat)}

        return {"error": "Unknown task"}


import re
