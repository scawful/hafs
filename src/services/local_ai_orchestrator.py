"""Local AI Orchestrator for priority-based inference with tool calling.

Manages local AI inference requests (via Ollama) with:
- Priority-based queueing (Training > Interactive > Analysis > Scheduled)
- Context window management (hafs context integration)
- Tool calling and execution
- Resource monitoring to prevent training interference
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Optional

from services.ollama_client import OllamaClient
from services.tool_executor import AVAILABLE_TOOLS, ToolExecutor

logger = logging.getLogger(__name__)


class RequestPriority(IntEnum):
    """Priority levels for inference requests.

    Lower number = higher priority.
    """

    TRAINING = 1      # Training data generation (highest)
    INTERACTIVE = 2   # User queries in chat mode
    ANALYSIS = 3      # Background analysis tasks
    SCHEDULED = 4     # Scheduled reports (lowest)


@dataclass
class InferenceRequest:
    """Request for local AI inference."""

    id: str
    priority: RequestPriority
    prompt: str
    model: str = "qwen2.5:7b"
    tools: list[dict[str, Any]] = field(default_factory=list)
    context_paths: list[str] = field(default_factory=list)
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    min_p: float = 0.05
    repeat_penalty: float = 1.1
    created_at: datetime = field(default_factory=datetime.now)

    callback: Optional[Callable[[dict[str, Any]], None]] = None


@dataclass
class InferenceResult:
    """Result of inference request."""

    request_id: str
    response: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    context_tokens: int = 0
    inference_time_seconds: float = 0.0
    model: str = ""
    error: Optional[str] = None


class LocalAIOrchestrator:
    """Orchestrates local AI inference with resource management.

    Features:
    - Priority-based request queuing
    - Context window management via hafs context system
    - Tool calling (file ops, embeddings, knowledge graph)
    - Resource monitoring (don't interfere with training)
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        default_model: str = "qwen2.5:7b",
        max_concurrent_requests: int = 1,
        context_root: Optional[Path] = None,
    ):
        """Initialize orchestrator.

        Args:
            ollama_url: Ollama API endpoint
            default_model: Default model to use
            max_concurrent_requests: Max concurrent inference (usually 1 for CPU)
            context_root: Root directory for hafs context (default: ~/.context)
        """
        self.ollama = OllamaClient(base_url=ollama_url)
        self.default_model = default_model
        self.max_concurrent = max_concurrent_requests
        self.context_root = context_root or Path.home() / ".context"

        # Tool executor
        self.tool_executor = ToolExecutor(context_root=self.context_root)

        # Priority queue using asyncio.PriorityQueue
        self.request_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.active_requests: int = 0
        self.request_lock = asyncio.Lock()

        # Request tracking
        self.pending_futures: dict[str, asyncio.Future] = {}

        # Metrics
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        self.total_inference_time = 0.0

        # Background task handle
        self._processor_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        """Start orchestrator background tasks."""
        if self._running:
            logger.warning("Orchestrator already running")
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._process_queue())
        logger.info("Local AI Orchestrator started")

    async def stop(self):
        """Stop orchestrator and drain queue."""
        self._running = False

        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        logger.info("Local AI Orchestrator stopped")

    async def submit_request(self, request: InferenceRequest) -> InferenceResult:
        """Submit inference request to queue.

        Args:
            request: Inference request

        Returns:
            InferenceResult when complete
        """
        if not self._running:
            await self.start()

        self.total_requests += 1

        logger.info(
            f"Queuing request {request.id} (priority={request.priority.name}, "
            f"queue_size={self.request_queue.qsize()})"
        )

        # Create future for result
        future: asyncio.Future = asyncio.Future()
        self.pending_futures[request.id] = future

        # Add to priority queue (lower priority value = higher priority)
        await self.request_queue.put((request.priority.value, request.id, request))

        # Wait for result
        try:
            result = await future
            return result
        except Exception as e:
            logger.error(f"Request {request.id} failed: {e}")
            raise

    async def _process_queue(self):
        """Process requests from queue (background task)."""
        logger.info("Request processor started")

        while self._running:
            # Check if we can process more requests
            async with self.request_lock:
                if self.active_requests >= self.max_concurrent:
                    await asyncio.sleep(0.1)
                    continue

                # Get next request (blocks until available)
                try:
                    priority, req_id, request = await asyncio.wait_for(
                        self.request_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                self.active_requests += 1

            # Process request in background
            asyncio.create_task(self._execute_request(request))

        logger.info("Request processor stopped")

    async def _execute_request(self, request: InferenceRequest) -> None:
        """Execute single inference request.

        Args:
            request: Request to execute
        """
        start_time = datetime.now()
        future = self.pending_futures.get(request.id)

        if not future:
            logger.error(f"No future found for request {request.id}")
            async with self.request_lock:
                self.active_requests -= 1
            return

        try:
            # 1. Build context
            context_data = await self._build_context(request)

            # 2. Prepare prompt with context
            full_prompt = self._prepare_prompt(request.prompt, context_data)

            # 3. Call Ollama with tools
            response = await self._call_ollama(
                prompt=full_prompt,
                model=request.model or self.default_model,
                tools=request.tools,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                min_p=request.min_p,
                repeat_penalty=request.repeat_penalty,
            )

            if not response:
                raise RuntimeError("Ollama returned empty response")

            # 4. Execute tool calls if any
            tool_results = []
            if response.get("tool_calls"):
                for tool_call in response["tool_calls"]:
                    result = await self.tool_executor.execute(tool_call)
                    tool_results.append(result)

            # 5. Build result
            inference_time = (datetime.now() - start_time).total_seconds()
            self.total_inference_time += inference_time

            result = InferenceResult(
                request_id=request.id,
                response=response.get("response", ""),
                tool_calls=tool_results,
                context_tokens=context_data.get("token_count", 0),
                inference_time_seconds=inference_time,
                model=request.model or self.default_model,
            )

            # Set result on future
            future.set_result(result)
            self.completed_requests += 1

            logger.info(
                f"Request {request.id} completed in {inference_time:.2f}s "
                f"({len(tool_results)} tool calls)"
            )

        except Exception as e:
            logger.error(f"Request {request.id} failed: {e}")

            result = InferenceResult(
                request_id=request.id,
                response="",
                error=str(e),
                model=request.model or self.default_model,
            )

            future.set_result(result)
            self.failed_requests += 1

        finally:
            # Cleanup
            self.pending_futures.pop(request.id, None)
            async with self.request_lock:
                self.active_requests -= 1

    async def _build_context(self, request: InferenceRequest) -> dict[str, Any]:
        """Build context from hafs context system.

        Args:
            request: Inference request with context_paths

        Returns:
            Context data dictionary
        """
        context = {
            "scratchpad": {},
            "files": [],
            "token_count": 0,
        }

        # Load scratchpad if requested
        if "scratchpad" in request.context_paths:
            for category in ["state", "metacognition", "epistemic"]:
                scratchpad_file = self.context_root / "scratchpad" / f"{category}.json"
                if scratchpad_file.exists():
                    try:
                        with open(scratchpad_file) as f:
                            context["scratchpad"][category] = json.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to load scratchpad/{category}: {e}")

        # Load specific files if requested
        for path in request.context_paths:
            if path == "scratchpad":
                continue  # Already handled

            file_path = Path(path).expanduser()
            if file_path.exists() and file_path.is_file():
                try:
                    with open(file_path) as f:
                        context["files"].append({
                            "path": str(file_path),
                            "content": f.read(4096),  # First 4KB
                        })
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")

        # Estimate token count (rough: 1 token ≈ 4 characters)
        context_text = json.dumps(context)
        context["token_count"] = len(context_text) // 4

        return context

    def _prepare_prompt(self, prompt: str, context_data: dict[str, Any]) -> str:
        """Prepare full prompt with context injection.

        Args:
            prompt: Original user prompt
            context_data: Retrieved context

        Returns:
            Full prompt with context
        """
        sections = []

        # Add scratchpad context if available
        if context_data.get("scratchpad"):
            sections.append("# hafs Context\n")

            for category, data in context_data["scratchpad"].items():
                if data:
                    sections.append(f"## {category.title()}\n")
                    sections.append(f"```json\n{json.dumps(data, indent=2)}\n```\n")

        # Add file context if available
        if context_data.get("files"):
            sections.append("# Referenced Files\n")
            for file_data in context_data["files"]:
                sections.append(f"## {file_data['path']}\n")
                sections.append(f"```\n{file_data['content']}\n```\n")

        # Add user prompt
        sections.append("# Task\n")
        sections.append(prompt)

        return "\n".join(sections)

    async def _call_ollama(
        self,
        prompt: str,
        model: str,
        tools: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> Optional[dict[str, Any]]:
        """Call Ollama API.

        Args:
            prompt: Full prompt
            model: Model name
            tools: Tool definitions
            max_tokens: Max tokens
            temperature: Temperature

        Returns:
            Response dict or None on error
        """
        # Use chat API for better tool calling support
        messages = [{"role": "user", "content": prompt}]

        response = await self.ollama.chat_async(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools if tools else None,
            **kwargs,
        )

        return response

    def get_stats(self) -> dict[str, Any]:
        """Get orchestrator statistics.

        Returns:
            Statistics dictionary
        """
        avg_time = (
            self.total_inference_time / self.completed_requests
            if self.completed_requests > 0
            else 0.0
        )

        return {
            "total_requests": self.total_requests,
            "completed": self.completed_requests,
            "failed": self.failed_requests,
            "active": self.active_requests,
            "queued": self.request_queue.qsize(),
            "avg_inference_time_seconds": round(avg_time, 2),
            "total_inference_time_seconds": round(self.total_inference_time, 2),
            "success_rate": (
                self.completed_requests / self.total_requests
                if self.total_requests > 0
                else 0.0
            ),
        }


# Convenience function for quick queries
async def query_local_ai(
    prompt: str,
    tools: bool = True,
    priority: RequestPriority = RequestPriority.INTERACTIVE,
    model: str = "qwen2.5:7b",
) -> str:
    """Quick helper for local AI queries.

    Args:
        prompt: Question or prompt
        tools: Whether to enable tool calling
        priority: Request priority
        model: Model to use

    Returns:
        Response text
    """
    orchestrator = LocalAIOrchestrator(default_model=model)
    await orchestrator.start()

    request = InferenceRequest(
        id=f"query_{datetime.now().timestamp()}",
        priority=priority,
        prompt=prompt,
        tools=AVAILABLE_TOOLS if tools else [],
    )

    result = await orchestrator.submit_request(request)

    await orchestrator.stop()

    if result.error:
        raise RuntimeError(f"Query failed: {result.error}")

    return result.response


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        # Create orchestrator
        orch = LocalAIOrchestrator()
        await orch.start()

        # Submit requests with different priorities

        # High priority: training
        training_req = InferenceRequest(
            id="training_001",
            priority=RequestPriority.TRAINING,
            prompt="Generate ASM training sample...",
            model="qwen2.5:3b",
        )

        # Low priority: scheduled analysis
        analysis_req = InferenceRequest(
            id="analysis_001",
            priority=RequestPriority.SCHEDULED,
            prompt="Analyze filesystem and suggest organization...",
            tools=AVAILABLE_TOOLS,
            context_paths=["scratchpad"],
        )

        # Submit both (training will execute first)
        results = await asyncio.gather(
            orch.submit_request(training_req),
            orch.submit_request(analysis_req),
        )

        print(f"\nTraining result: {results[0].response[:100]}...")
        print(f"Analysis result: {results[1].response[:100]}...")
        print(f"\nStats: {orch.get_stats()}")

        await orch.stop()

    asyncio.run(main())
