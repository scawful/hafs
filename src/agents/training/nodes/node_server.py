"""Training Node Server.

Server for accepting training data generation jobs from remote clients.
Designed to run on machines with GPU acceleration (e.g., Windows with 5060TI).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agents.training.base import DataGenerator, SourceItem, TrainingSample

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = 8765
    gpu_name: str = ""
    gpu_memory_gb: int = 16
    max_concurrent_jobs: int = 2
    results_dir: Path = field(default_factory=lambda: Path.home() / ".hafs_training" / "results")
    capabilities: list[str] = field(default_factory=lambda: ["asm", "cpp", "text"])


@dataclass
class JobState:
    """Internal state for a job."""

    job_id: str
    domain: str
    source_items: list[dict[str, Any]]
    batch_size: int
    teacher_model: str
    status: str = "pending"
    progress: float = 0.0
    samples: list[TrainingSample] = field(default_factory=list)
    error: Optional[str] = None
    created_at: str = ""
    completed_at: str = ""
    task: Optional[asyncio.Task] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "domain": self.domain,
            "status": self.status,
            "progress": self.progress,
            "samples_generated": len(self.samples),
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


class TrainingNodeServer:
    """Server for accepting training generation jobs.

    Runs as a standalone service on a machine with GPU.
    Accepts jobs via HTTP API and processes them asynchronously.
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig()
        self.config.results_dir.mkdir(parents=True, exist_ok=True)

        self._jobs: dict[str, JobState] = {}
        self._generators: dict[str, DataGenerator] = {}
        self._running_jobs: int = 0
        self._app = None

    async def setup(self):
        """Initialize generators and resources."""
        # Lazy import generators
        from agents.training.generators import (
            AsmDataGenerator,
            CppDataGenerator,
            TextDataGenerator,
        )

        # Register available generators
        if "asm" in self.config.capabilities:
            gen = AsmDataGenerator()
            await gen.setup()
            self._generators["asm"] = gen

        if "cpp" in self.config.capabilities:
            gen = CppDataGenerator()
            await gen.setup()
            self._generators["cpp"] = gen

        if "text" in self.config.capabilities:
            gen = TextDataGenerator()
            await gen.setup()
            self._generators["text"] = gen

        logger.info(f"Initialized generators: {list(self._generators.keys())}")

    def _create_app(self):
        """Create the FastAPI application."""
        try:
            from fastapi import FastAPI, HTTPException, BackgroundTasks
            from pydantic import BaseModel
        except ImportError:
            logger.error("FastAPI not installed. Run: pip install fastapi uvicorn")
            raise

        app = FastAPI(title="hafs Training Node")

        class GenerationRequest(BaseModel):
            domain: str
            source_items: list[dict[str, Any]]
            batch_size: int = 10
            teacher_model: str = "gemini-2.0-flash"

        @app.get("/health")
        async def health_check():
            return {
                "name": f"hafs-node-{self.config.port}",
                "gpu": self.config.gpu_name,
                "memory_gb": self.config.gpu_memory_gb,
                "status": "online",
                "capabilities": self.config.capabilities,
                "running_jobs": self._running_jobs,
                "max_concurrent_jobs": self.config.max_concurrent_jobs,
            }

        @app.post("/jobs/generate")
        async def create_generation_job(
            request: GenerationRequest,
            background_tasks: BackgroundTasks,
        ):
            if request.domain not in self._generators:
                raise HTTPException(
                    status_code=400,
                    detail=f"Domain not supported: {request.domain}",
                )

            if self._running_jobs >= self.config.max_concurrent_jobs:
                raise HTTPException(
                    status_code=503,
                    detail="Server at capacity, try again later",
                )

            job_id = str(uuid.uuid4())
            job = JobState(
                job_id=job_id,
                domain=request.domain,
                source_items=request.source_items,
                batch_size=request.batch_size,
                teacher_model=request.teacher_model,
            )
            self._jobs[job_id] = job

            # Start processing in background
            background_tasks.add_task(self._process_job, job_id)

            return {"job_id": job_id, "status": "pending"}

        @app.get("/jobs/{job_id}")
        async def get_job_status(job_id: str):
            if job_id not in self._jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            return self._jobs[job_id].to_dict()

        @app.get("/jobs/{job_id}/results")
        async def get_job_results(job_id: str):
            if job_id not in self._jobs:
                raise HTTPException(status_code=404, detail="Job not found")

            job = self._jobs[job_id]
            if job.status != "completed":
                raise HTTPException(
                    status_code=400,
                    detail=f"Job not completed, status: {job.status}",
                )

            return {
                "job_id": job_id,
                "samples": [s.to_dict() for s in job.samples],
            }

        @app.post("/jobs/{job_id}/cancel")
        async def cancel_job(job_id: str):
            if job_id not in self._jobs:
                raise HTTPException(status_code=404, detail="Job not found")

            job = self._jobs[job_id]
            if job.task and not job.task.done():
                job.task.cancel()
                job.status = "cancelled"
                return {"status": "cancelled"}

            return {"status": job.status}

        @app.get("/jobs")
        async def list_jobs(
            status: Optional[str] = None,
            domain: Optional[str] = None,
        ):
            jobs = []
            for job in self._jobs.values():
                if status and job.status != status:
                    continue
                if domain and job.domain != domain:
                    continue
                jobs.append(job.to_dict())
            return {"jobs": jobs}

        return app

    async def _process_job(self, job_id: str):
        """Process a generation job."""
        job = self._jobs[job_id]
        job.status = "running"
        self._running_jobs += 1

        try:
            generator = self._generators[job.domain]

            # Convert source items to SourceItem objects
            from agents.training.generators import (
                AsmSourceItem,
                CppSourceItem,
                TextSourceItem,
            )

            items = []
            for item_data in job.source_items:
                if job.domain == "asm":
                    items.append(AsmSourceItem(**item_data))
                elif job.domain == "cpp":
                    items.append(CppSourceItem(**item_data))
                elif job.domain == "text":
                    items.append(TextSourceItem(**item_data))
                else:
                    items.append(SourceItem(**item_data))

            # Process items
            total = len(items)
            if total == 0:
                job.progress = 1.0
            else:
                for i, item in enumerate(items):
                    try:
                        sample = await generator.generate_sample(item)
                        if sample:
                            job.samples.append(sample)
                    except Exception as e:
                        logger.error(f"Error generating sample: {e}")

                    job.progress = (i + 1) / total

            job.status = "completed"
            job.completed_at = datetime.now().isoformat()

            # Save results to disk
            results_path = self.config.results_dir / f"{job_id}.jsonl"
            with open(results_path, "w") as f:
                for sample in job.samples:
                    f.write(json.dumps(sample.to_dict()) + "\n")

            logger.info(f"Job {job_id} completed: {len(job.samples)} samples")

        except asyncio.CancelledError:
            job.status = "cancelled"
            logger.info(f"Job {job_id} cancelled")
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error(f"Job {job_id} failed: {e}")
        finally:
            self._running_jobs -= 1

    async def start(self):
        """Start the server."""
        await self.setup()
        self._app = self._create_app()

        try:
            import uvicorn
        except ImportError:
            logger.error("uvicorn not installed. Run: pip install uvicorn")
            raise

        config = uvicorn.Config(
            self._app,
            host=self.config.host,
            port=self.config.port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()


def main():
    """CLI entry point for running the server."""
    import argparse

    parser = argparse.ArgumentParser(description="hafs Training Node Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind")
    parser.add_argument("--gpu", default="", help="GPU name")
    parser.add_argument("--memory", type=int, default=16, help="GPU memory in GB")
    args = parser.parse_args()

    config = ServerConfig(
        host=args.host,
        port=args.port,
        gpu_name=args.gpu,
        gpu_memory_gb=args.memory,
    )

    server = TrainingNodeServer(config)
    asyncio.run(server.start())


if __name__ == "__main__":
    main()
