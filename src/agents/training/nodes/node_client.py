"""Training Node Client.

Client for connecting to remote training nodes for distributed
data generation or training job submission.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from pathlib import Path

from agents.training.base import TrainingSample

logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """Information about a training node."""

    name: str
    host: str
    port: int
    gpu: str = ""
    memory_gb: int = 0
    status: str = "unknown"  # online, offline, busy, error
    last_seen: str = ""
    capabilities: list[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class Job:
    """A generation or training job."""

    job_id: str
    domain: str
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0
    samples_generated: int = 0
    error: Optional[str] = None
    created_at: str = ""
    completed_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class TrainingNodeClient:
    """Client for connecting to remote training nodes.

    Supports:
    - Submitting generation jobs
    - Monitoring job progress
    - Fetching completed results
    - Health checks
    """

    def __init__(
        self,
        host: str,
        port: int = 8765,
        timeout: float = 30.0,
    ):
        """Initialize client.

        Args:
            host: Node hostname or IP
            port: Node port
            timeout: Request timeout in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}"
        self._session = None

    async def _get_session(self):
        """Get or create HTTP session."""
        if self._session is None:
            try:
                import aiohttp
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                )
            except ImportError:
                logger.error("aiohttp not installed, HTTP client unavailable")
                raise
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def health_check(self) -> NodeInfo:
        """Check node health and get info.

        Returns:
            NodeInfo with node status and capabilities
        """
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return NodeInfo(
                        name=data.get("name", "unknown"),
                        host=self.host,
                        port=self.port,
                        gpu=data.get("gpu", ""),
                        memory_gb=data.get("memory_gb", 0),
                        status="online",
                        last_seen=datetime.now().isoformat(),
                        capabilities=data.get("capabilities", []),
                    )
                else:
                    return NodeInfo(
                        name="unknown",
                        host=self.host,
                        port=self.port,
                        status="error",
                    )
        except Exception as e:
            logger.warning(f"Health check failed for {self.host}: {e}")
            return NodeInfo(
                name="unknown",
                host=self.host,
                port=self.port,
                status="offline",
            )

    async def submit_generation_job(
        self,
        domain: str,
        source_items: list[dict[str, Any]],
        batch_size: int = 10,
        teacher_model: str = "gemini-2.0-flash",
    ) -> str:
        """Submit a generation job to the node.

        Args:
            domain: Domain for generation (asm, cpp, text)
            source_items: List of source item dictionaries
            batch_size: Batch size for generation
            teacher_model: Teacher model to use

        Returns:
            Job ID for tracking
        """
        session = await self._get_session()

        payload = {
            "domain": domain,
            "source_items": source_items,
            "batch_size": batch_size,
            "teacher_model": teacher_model,
        }

        async with session.post(
            f"{self.base_url}/jobs/generate",
            json=payload,
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data["job_id"]
            else:
                error = await resp.text()
                raise RuntimeError(f"Failed to submit job: {error}")

    async def get_job_status(self, job_id: str) -> Job:
        """Get status of a remote job.

        Args:
            job_id: Job ID to check

        Returns:
            Job object with current status
        """
        session = await self._get_session()

        async with session.get(f"{self.base_url}/jobs/{job_id}") as resp:
            if resp.status == 200:
                data = await resp.json()
                return Job(
                    job_id=job_id,
                    domain=data.get("domain", ""),
                    status=data.get("status", "unknown"),
                    progress=data.get("progress", 0.0),
                    samples_generated=data.get("samples_generated", 0),
                    error=data.get("error"),
                    created_at=data.get("created_at", ""),
                    completed_at=data.get("completed_at", ""),
                )
            elif resp.status == 404:
                raise ValueError(f"Job not found: {job_id}")
            else:
                raise RuntimeError(f"Failed to get job status: {await resp.text()}")

    async def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 5.0,
        timeout: float = 3600.0,
    ) -> Job:
        """Wait for a job to complete.

        Args:
            job_id: Job ID to wait for
            poll_interval: Seconds between status checks
            timeout: Maximum wait time in seconds

        Returns:
            Completed job
        """
        start = asyncio.get_event_loop().time()

        while True:
            job = await self.get_job_status(job_id)

            if job.status in ("completed", "failed"):
                return job

            elapsed = asyncio.get_event_loop().time() - start
            if elapsed > timeout:
                raise TimeoutError(f"Job {job_id} did not complete in {timeout}s")

            await asyncio.sleep(poll_interval)

    async def fetch_results(self, job_id: str) -> list[TrainingSample]:
        """Fetch completed samples from a job.

        Args:
            job_id: Job ID to fetch results for

        Returns:
            List of generated TrainingSamples
        """
        session = await self._get_session()

        async with session.get(f"{self.base_url}/jobs/{job_id}/results") as resp:
            if resp.status == 200:
                data = await resp.json()
                samples = []
                for item in data.get("samples", []):
                    samples.append(TrainingSample.from_dict(item))
                return samples
            else:
                raise RuntimeError(f"Failed to fetch results: {await resp.text()}")

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if cancelled successfully
        """
        session = await self._get_session()

        async with session.post(f"{self.base_url}/jobs/{job_id}/cancel") as resp:
            return resp.status == 200

    async def list_jobs(
        self,
        status: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> list[Job]:
        """List jobs on the node.

        Args:
            status: Filter by status
            domain: Filter by domain

        Returns:
            List of jobs
        """
        session = await self._get_session()

        params = {}
        if status:
            params["status"] = status
        if domain:
            params["domain"] = domain

        async with session.get(f"{self.base_url}/jobs", params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                return [
                    Job(
                        job_id=j["job_id"],
                        domain=j.get("domain", ""),
                        status=j.get("status", "unknown"),
                        progress=j.get("progress", 0.0),
                        samples_generated=j.get("samples_generated", 0),
                    )
                    for j in data.get("jobs", [])
                ]
            else:
                return []


class NodePool:
    """Pool of training nodes for load balancing."""

    def __init__(self):
        self.nodes: dict[str, TrainingNodeClient] = {}
        self._node_info: dict[str, NodeInfo] = {}

    def add_node(
        self,
        name: str,
        host: str,
        port: int = 8765,
    ) -> None:
        """Add a node to the pool."""
        self.nodes[name] = TrainingNodeClient(host, port)

    def remove_node(self, name: str) -> None:
        """Remove a node from the pool."""
        if name in self.nodes:
            del self.nodes[name]
        if name in self._node_info:
            del self._node_info[name]

    async def refresh_status(self) -> dict[str, NodeInfo]:
        """Refresh status of all nodes."""
        tasks = []
        for name, client in self.nodes.items():
            tasks.append((name, client.health_check()))

        results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

        for (name, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                self._node_info[name] = NodeInfo(
                    name=name,
                    host=self.nodes[name].host,
                    port=self.nodes[name].port,
                    status="error",
                )
            else:
                result.name = name
                self._node_info[name] = result

        return self._node_info

    def get_available_nodes(self) -> list[str]:
        """Get names of available (online) nodes."""
        return [
            name
            for name, info in self._node_info.items()
            if info.status == "online"
        ]

    def get_node(self, name: str) -> Optional[TrainingNodeClient]:
        """Get a specific node client."""
        return self.nodes.get(name)

    async def close_all(self) -> None:
        """Close all node connections."""
        for client in self.nodes.values():
            await client.close()
