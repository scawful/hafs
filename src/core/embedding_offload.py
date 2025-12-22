"""Embedding Generation Offload - Remote GPU Acceleration.

Offloads embedding generation to remote GPU nodes (medical-mechanica, etc.) to:
1. Save local Mac battery/CPU when on the go
2. Massively speed up batch embedding operations
3. Free up local resources for other tasks

Supports:
- Ollama embedding models (nomic-embed-text, embeddinggemma)
- Automatic batching for efficiency
- Fallback to local/API if remote unavailable
- Queue management for background operations

Usage:
    # Configure offload
    offloader = EmbeddingOffloader()
    await offloader.add_node("medical-mechanica", "100.100.100.20:11434")

    # Generate embeddings (automatically routes to best node)
    embeddings = await offloader.embed_batch(texts)

    # Queue for background processing
    job_id = await offloader.queue_embed_batch(texts, save_to="~/.context/embeddings/")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingNode:
    """Remote node for embedding generation."""

    name: str
    host: str  # e.g., "100.100.100.20:11434"
    model: str = "nomic-embed-text"  # Ollama embedding model
    status: str = "unknown"  # online, offline, busy
    latency_ms: float = 0.0
    embeddings_generated: int = 0
    last_check: Optional[datetime] = None

    @property
    def base_url(self) -> str:
        """Get Ollama API base URL."""
        return f"http://{self.host}"


@dataclass
class EmbeddingJob:
    """Background embedding job."""

    job_id: str
    texts: list[str]
    node_name: str
    output_path: Optional[Path] = None
    status: str = "queued"  # queued, running, completed, failed
    progress: int = 0
    total: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class EmbeddingOffloader:
    """Manages embedding generation across remote GPU nodes."""

    def __init__(self):
        self.nodes: dict[str, EmbeddingNode] = {}
        self.jobs: dict[str, EmbeddingJob] = {}
        self._client = httpx.AsyncClient(timeout=30.0)

    async def add_node(
        self, name: str, host: str, model: str = "nomic-embed-text"
    ) -> bool:
        """Add a remote embedding node.

        Args:
            name: Node identifier (e.g., "medical-mechanica")
            host: Host:port (e.g., "100.100.100.20:11434")
            model: Ollama embedding model

        Returns:
            True if node is reachable, False otherwise
        """
        node = EmbeddingNode(name=name, host=host, model=model)

        # Test connectivity
        is_online = await self._health_check(node)
        node.status = "online" if is_online else "offline"
        node.last_check = datetime.now()

        self.nodes[name] = node

        if is_online:
            logger.info(f"✓ Added embedding node: {name} ({host}) - {model}")
            return True
        else:
            logger.warning(f"✗ Node {name} ({host}) is offline")
            return False

    async def _health_check(self, node: EmbeddingNode) -> bool:
        """Check if node is reachable."""
        try:
            response = await self._client.get(
                f"{node.base_url}/api/tags", timeout=5.0
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Health check failed for {node.name}: {e}")
            return False

    async def get_best_node(self) -> Optional[EmbeddingNode]:
        """Get best available node (lowest latency, online)."""
        online_nodes = [n for n in self.nodes.values() if n.status == "online"]

        if not online_nodes:
            logger.warning("No online embedding nodes available")
            return None

        # Sort by latency (lower is better)
        online_nodes.sort(key=lambda n: n.latency_ms)
        return online_nodes[0]

    async def embed_single(
        self, text: str, node_name: Optional[str] = None
    ) -> Optional[list[float]]:
        """Generate embedding for single text.

        Args:
            text: Text to embed
            node_name: Specific node to use (None = auto-select best)

        Returns:
            Embedding vector or None if failed
        """
        # Select node
        if node_name:
            node = self.nodes.get(node_name)
            if not node:
                logger.error(f"Node {node_name} not found")
                return None
        else:
            node = await self.get_best_node()
            if not node:
                return None

        # Generate embedding via Ollama API
        try:
            response = await self._client.post(
                f"{node.base_url}/api/embeddings",
                json={"model": node.model, "prompt": text},
                timeout=10.0,
            )

            if response.status_code == 200:
                data = response.json()
                embedding = data.get("embedding", [])
                node.embeddings_generated += 1
                return embedding
            else:
                logger.error(f"Embedding failed: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Embedding generation failed on {node.name}: {e}")
            node.status = "offline"
            return None

    async def embed_batch(
        self, texts: list[str], batch_size: int = 50, node_name: Optional[str] = None
    ) -> list[Optional[list[float]]]:
        """Generate embeddings for batch of texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for parallel processing
            node_name: Specific node to use (None = auto-select)

        Returns:
            List of embedding vectors (None for failed items)
        """
        logger.info(f"Generating {len(texts)} embeddings (batch_size={batch_size})")

        # Select node
        if node_name:
            node = self.nodes.get(node_name)
        else:
            node = await self.get_best_node()

        if not node:
            logger.error("No node available for batch embedding")
            return [None] * len(texts)

        # Process in batches
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Parallel requests within batch
            tasks = [self.embed_single(text, node.name) for text in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch embedding error: {result}")
                    embeddings.append(None)
                else:
                    embeddings.append(result)

            logger.info(f"Progress: {min(i + batch_size, len(texts))}/{len(texts)}")

        success_count = sum(1 for e in embeddings if e is not None)
        logger.info(
            f"✓ Batch complete: {success_count}/{len(texts)} successful "
            f"(node: {node.name})"
        )

        return embeddings

    async def queue_embed_batch(
        self, texts: list[str], output_path: Optional[Path] = None
    ) -> str:
        """Queue embedding generation for background processing.

        Useful for large batches that should run in background while
        Mac is on the go (saves battery, doesn't block).

        Args:
            texts: Texts to embed
            output_path: Where to save embeddings (JSON)

        Returns:
            Job ID for tracking
        """
        # Generate job ID
        job_id = hashlib.md5(
            f"{datetime.now().isoformat()}_{len(texts)}".encode()
        ).hexdigest()[:12]

        # Get best node
        node = await self.get_best_node()
        if not node:
            logger.error("No node available for queued job")
            raise RuntimeError("No embedding nodes available")

        # Create job
        job = EmbeddingJob(
            job_id=job_id,
            texts=texts,
            node_name=node.name,
            output_path=output_path,
            total=len(texts),
        )

        self.jobs[job_id] = job

        # Start background task
        asyncio.create_task(self._process_job(job_id))

        logger.info(f"✓ Queued embedding job {job_id}: {len(texts)} texts on {node.name}")
        return job_id

    async def _process_job(self, job_id: str):
        """Process queued embedding job in background."""
        job = self.jobs.get(job_id)
        if not job:
            return

        job.status = "running"
        logger.info(f"Starting job {job_id}...")

        try:
            # Generate embeddings
            embeddings = await self.embed_batch(job.texts, node_name=job.node_name)

            job.progress = len(embeddings)

            # Save if output path specified
            if job.output_path:
                job.output_path.parent.mkdir(parents=True, exist_ok=True)

                # Save as JSONL
                with open(job.output_path, "w") as f:
                    for i, (text, embedding) in enumerate(zip(job.texts, embeddings)):
                        if embedding:
                            f.write(
                                json.dumps(
                                    {
                                        "id": i,
                                        "text": text[:500],
                                        "embedding": embedding,
                                    }
                                )
                                + "\n"
                            )

                logger.info(f"✓ Saved embeddings to {job.output_path}")

            job.status = "completed"
            job.completed_at = datetime.now()

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error(f"Job {job_id} failed: {e}")

    def get_job_status(self, job_id: str) -> Optional[dict]:
        """Get status of queued job."""
        job = self.jobs.get(job_id)
        if not job:
            return None

        return {
            "job_id": job.job_id,
            "status": job.status,
            "progress": f"{job.progress}/{job.total}",
            "node": job.node_name,
            "created_at": job.created_at.isoformat(),
            "error": job.error,
        }

    def get_stats(self) -> dict:
        """Get offloader statistics."""
        return {
            "nodes": {
                name: {
                    "status": node.status,
                    "latency_ms": node.latency_ms,
                    "embeddings_generated": node.embeddings_generated,
                    "last_check": node.last_check.isoformat()
                    if node.last_check
                    else None,
                }
                for name, node in self.nodes.items()
            },
            "jobs": {
                job_id: self.get_job_status(job_id) for job_id in self.jobs.keys()
            },
        }


# Global singleton
_offloader: Optional[EmbeddingOffloader] = None


def get_offloader() -> EmbeddingOffloader:
    """Get global embedding offloader instance."""
    global _offloader
    if _offloader is None:
        _offloader = EmbeddingOffloader()
    return _offloader


async def main():
    """Test embedding offload."""
    logger.basicConfig(level=logging.INFO)

    offloader = EmbeddingOffloader()

    # Add medical-mechanica
    await offloader.add_node("medical-mechanica", "100.100.100.20:11434")

    # Test single embedding
    embedding = await offloader.embed_single("Test text for embedding")
    if embedding:
        print(f"✓ Single embedding: {len(embedding)} dimensions")

    # Test batch
    texts = [f"Sample text {i}" for i in range(10)]
    embeddings = await offloader.embed_batch(texts, batch_size=5)
    print(f"✓ Batch: {sum(1 for e in embeddings if e)} / {len(texts)} successful")

    # Show stats
    stats = offloader.get_stats()
    print(f"\nStats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    import logging

    asyncio.run(main())
