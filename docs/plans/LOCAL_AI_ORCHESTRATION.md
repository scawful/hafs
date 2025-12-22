# Local AI Orchestration System

**Created:** 2025-12-21
**Status:** Design Phase
**Goal:** Full local AI orchestration with tool calling, context management, and resource scheduling

## Problem Statement

We need Ollama models to:
1. **Call tools** (file operations, knowledge graph queries, code execution)
2. **Access hafs context** (embeddings, knowledge graph, scratchpad)
3. **Manage resources** (queue inference requests, respect training priority)
4. **Handle context windows** (summarization, chunking, retrieval)

This goes beyond simple text generation - we're building a local Claude-like system.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Local AI Orchestrator                        │
│                  (New Component - Python)                       │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              Priority Queue Manager                       │ │
│  │  • Training jobs: Priority 1 (highest)                    │ │
│  │  • Interactive queries: Priority 2                        │ │
│  │  • Background analysis: Priority 3                        │ │
│  │  • Scheduled reports: Priority 4 (lowest)                 │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              │                                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │           Context Window Manager                          │ │
│  │  • Track token usage per request                          │ │
│  │  • Auto-summarize long contexts                           │ │
│  │  • Retrieve relevant embeddings                           │ │
│  │  • Integrate with context_agent_daemon                    │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              │                                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              Tool Execution Engine                        │ │
│  │  • File operations (read, write, search)                  │ │
│  │  • Knowledge graph queries                                │ │
│  │  • Embedding search                                       │ │
│  │  • Code execution (sandboxed)                             │ │
│  │  • hafs command dispatch                                  │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
┌───────────────▼──────┐       ┌────────────▼──────────┐
│  Ollama API          │       │  hafs Context System  │
│  (llama.cpp/Ollama)  │       │                       │
│                      │       │  • Knowledge Graph    │
│  • Function Calling  │       │  • Embeddings         │
│  • CPU Inference     │       │  • Scratchpad         │
│  • qwen2.5:7b        │       │  • Agent History      │
└──────────────────────┘       └───────────────────────┘
```

## Core Components

### 1. Local AI Orchestrator (`local_ai_orchestrator.py`)

Main orchestration service that manages all local AI requests.

```python
# src/hafs/services/local_ai_orchestrator.py

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Optional

from hafs.core.context_manager import ContextManager
from hafs.services.ollama_client import OllamaClient
from hafs.services.tool_executor import ToolExecutor

logger = logging.getLogger(__name__)


class RequestPriority(IntEnum):
    """Priority levels for inference requests."""
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
    tools: list[dict[str, Any]] = field(default_factory=list)
    context_ids: list[str] = field(default_factory=list)
    max_tokens: int = 4096
    temperature: float = 0.7
    created_at: datetime = field(default_factory=datetime.now)
    callback: Optional[Callable[[str], None]] = None


@dataclass
class InferenceResult:
    """Result of inference request."""

    request_id: str
    response: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    context_used_tokens: int = 0
    inference_time_seconds: float = 0.0
    error: Optional[str] = None


class LocalAIOrchestrator:
    """Orchestrates local AI inference with resource management and tool calling.

    Features:
    - Priority-based request queuing
    - Context window management via hafs context system
    - Tool calling (file ops, KG queries, embeddings)
    - Resource monitoring (don't interfere with training)
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        max_concurrent_requests: int = 1,
        context_manager: Optional[ContextManager] = None,
    ):
        """Initialize orchestrator.

        Args:
            ollama_url: Ollama API endpoint
            model: Model to use (must support function calling)
            max_concurrent_requests: Max concurrent inference (usually 1 for CPU)
            context_manager: hafs context manager for embeddings/KG
        """
        self.ollama = OllamaClient(base_url=ollama_url)
        self.model = model
        self.max_concurrent = max_concurrent_requests
        self.context_manager = context_manager or ContextManager()
        self.tool_executor = ToolExecutor(context_manager=self.context_manager)

        # Priority queue (heapq)
        self.request_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.active_requests: int = 0
        self.request_lock = asyncio.Lock()

        # Metrics
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0

    async def submit_request(self, request: InferenceRequest) -> InferenceResult:
        """Submit inference request to queue.

        Args:
            request: Inference request

        Returns:
            InferenceResult when complete
        """
        self.total_requests += 1
        logger.info(
            f"Queuing request {request.id} (priority={request.priority.name}, "
            f"queue_size={self.request_queue.qsize()})"
        )

        # Create future for result
        future = asyncio.Future()

        # Add to priority queue (lower number = higher priority)
        await self.request_queue.put((request.priority.value, request.id, request, future))

        # Wait for result
        result = await future
        return result

    async def _process_queue(self):
        """Process requests from queue (background task)."""
        while True:
            # Check if we can process more requests
            async with self.request_lock:
                if self.active_requests >= self.max_concurrent:
                    await asyncio.sleep(0.1)
                    continue

                # Get next request
                try:
                    priority, req_id, request, future = await asyncio.wait_for(
                        self.request_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.1)
                    continue

                self.active_requests += 1

            # Process request
            try:
                result = await self._execute_request(request)
                future.set_result(result)
                self.completed_requests += 1
            except Exception as e:
                logger.error(f"Request {req_id} failed: {e}")
                future.set_exception(e)
                self.failed_requests += 1
            finally:
                async with self.request_lock:
                    self.active_requests -= 1

    async def _execute_request(self, request: InferenceRequest) -> InferenceResult:
        """Execute single inference request.

        Args:
            request: Request to execute

        Returns:
            InferenceResult
        """
        start_time = datetime.now()

        # 1. Build context from hafs system
        context_data = await self._build_context(request)

        # 2. Prepare prompt with context
        full_prompt = self._prepare_prompt(request, context_data)

        # 3. Call Ollama with tools
        response = await self._call_ollama_with_tools(
            prompt=full_prompt,
            tools=request.tools,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        # 4. Execute tool calls if any
        tool_results = []
        if response.get("tool_calls"):
            for tool_call in response["tool_calls"]:
                result = await self.tool_executor.execute(tool_call)
                tool_results.append(result)

        # 5. Build result
        inference_time = (datetime.now() - start_time).total_seconds()

        return InferenceResult(
            request_id=request.id,
            response=response.get("response", ""),
            tool_calls=tool_results,
            context_used_tokens=context_data.get("token_count", 0),
            inference_time_seconds=inference_time,
        )

    async def _build_context(self, request: InferenceRequest) -> dict[str, Any]:
        """Build context from hafs context system.

        Args:
            request: Inference request with context_ids

        Returns:
            Context data with embeddings, KG entries, scratchpad
        """
        context = {
            "embeddings": [],
            "knowledge_graph": [],
            "scratchpad": [],
            "token_count": 0,
        }

        # Retrieve relevant embeddings
        if request.context_ids:
            for context_id in request.context_ids:
                # Query embedding service
                results = await self.context_manager.search_embeddings(
                    query=request.prompt,
                    context_id=context_id,
                    top_k=5,
                )
                context["embeddings"].extend(results)

        # Query knowledge graph
        kg_results = await self.context_manager.query_knowledge_graph(
            query=request.prompt,
            max_results=10,
        )
        context["knowledge_graph"] = kg_results

        # Load scratchpad state
        scratchpad = await self.context_manager.load_scratchpad()
        context["scratchpad"] = scratchpad

        # Estimate token count
        context["token_count"] = self._estimate_tokens(context)

        return context

    def _prepare_prompt(
        self,
        request: InferenceRequest,
        context_data: dict[str, Any]
    ) -> str:
        """Prepare full prompt with context injection.

        Args:
            request: Original request
            context_data: Retrieved context

        Returns:
            Full prompt with context
        """
        sections = []

        # System context
        sections.append("# hafs Context System")
        sections.append("")

        # Embeddings
        if context_data["embeddings"]:
            sections.append("## Relevant Documents")
            for emb in context_data["embeddings"][:5]:
                sections.append(f"- {emb['title']}: {emb['snippet']}")
            sections.append("")

        # Knowledge graph
        if context_data["knowledge_graph"]:
            sections.append("## Knowledge Graph")
            for entry in context_data["knowledge_graph"][:10]:
                sections.append(f"- {entry['entity']}: {entry['relation']} {entry['target']}")
            sections.append("")

        # Scratchpad
        if context_data["scratchpad"]:
            sections.append("## Current State")
            sections.append(f"```json\n{context_data['scratchpad']}\n```")
            sections.append("")

        # User request
        sections.append("# User Request")
        sections.append(request.prompt)

        return "\n".join(sections)

    async def _call_ollama_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        """Call Ollama with tool/function calling support.

        Args:
            prompt: Full prompt
            tools: Tool definitions
            max_tokens: Max generation tokens
            temperature: Sampling temperature

        Returns:
            Response with optional tool calls
        """
        # Prepare request
        request_data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        # Add tools if supported
        if tools:
            request_data["tools"] = tools

        # Call Ollama
        response = await self.ollama.generate_async(request_data)

        return response

    def _estimate_tokens(self, context: dict[str, Any]) -> int:
        """Estimate token count for context.

        Args:
            context: Context data

        Returns:
            Estimated token count
        """
        # Rough estimate: 1 token ≈ 4 characters
        text = str(context)
        return len(text) // 4

    async def start(self):
        """Start orchestrator background tasks."""
        logger.info("Starting Local AI Orchestrator")
        asyncio.create_task(self._process_queue())

    async def stop(self):
        """Stop orchestrator and drain queue."""
        logger.info("Stopping Local AI Orchestrator")
        # Wait for queue to drain
        await self.request_queue.join()
```

### 2. Tool Executor (`tool_executor.py`)

Executes tools called by the LLM.

```python
# src/hafs/services/tool_executor.py

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from hafs.core.context_manager import ContextManager

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Executes tools/functions for local AI."""

    def __init__(self, context_manager: ContextManager):
        """Initialize tool executor.

        Args:
            context_manager: hafs context manager
        """
        self.context_manager = context_manager

        # Register available tools
        self.tools = {
            "read_file": self._read_file,
            "search_embeddings": self._search_embeddings,
            "query_knowledge_graph": self._query_knowledge_graph,
            "run_command": self._run_command,
            "search_code": self._search_code,
        }

    async def execute(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool call.

        Args:
            tool_call: Tool call from LLM

        Returns:
            Tool result
        """
        tool_name = tool_call.get("name")
        parameters = tool_call.get("parameters", {})

        if tool_name not in self.tools:
            return {
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(self.tools.keys()),
            }

        try:
            result = await self.tools[tool_name](**parameters)
            return {"result": result}
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return {"error": str(e)}

    async def _read_file(self, path: str) -> str:
        """Read file contents.

        Args:
            path: File path

        Returns:
            File contents
        """
        file_path = Path(path).expanduser()

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(file_path) as f:
            return f.read()

    async def _search_embeddings(
        self,
        query: str,
        context_id: str = "default",
        top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Search embeddings.

        Args:
            query: Search query
            context_id: Context/domain ID
            top_k: Number of results

        Returns:
            Search results
        """
        return await self.context_manager.search_embeddings(
            query=query,
            context_id=context_id,
            top_k=top_k,
        )

    async def _query_knowledge_graph(
        self,
        query: str,
        max_results: int = 10
    ) -> list[dict[str, Any]]:
        """Query knowledge graph.

        Args:
            query: Natural language query
            max_results: Max results

        Returns:
            KG results
        """
        return await self.context_manager.query_knowledge_graph(
            query=query,
            max_results=max_results,
        )

    async def _run_command(
        self,
        command: str,
        cwd: Optional[str] = None
    ) -> str:
        """Run shell command (sandboxed).

        Args:
            command: Command to run
            cwd: Working directory

        Returns:
            Command output
        """
        # Whitelist of safe commands
        safe_commands = ["git", "ls", "find", "grep", "cat", "hafs"]

        cmd_name = command.split()[0]
        if cmd_name not in safe_commands:
            raise PermissionError(f"Command not allowed: {cmd_name}")

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"Command failed: {stderr.decode()}")

        return stdout.decode()

    async def _search_code(
        self,
        pattern: str,
        directory: str = "."
    ) -> list[dict[str, Any]]:
        """Search code using ripgrep.

        Args:
            pattern: Regex pattern
            directory: Directory to search

        Returns:
            Search results
        """
        command = f"rg --json '{pattern}' {directory}"
        output = await self._run_command(command)

        # Parse ripgrep JSON output
        import json
        results = []
        for line in output.strip().split("\n"):
            if line:
                try:
                    data = json.loads(line)
                    if data.get("type") == "match":
                        results.append({
                            "file": data["data"]["path"]["text"],
                            "line": data["data"]["line_number"],
                            "text": data["data"]["lines"]["text"],
                        })
                except json.JSONDecodeError:
                    pass

        return results


# Tool definitions for Ollama
AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to file to read",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_embeddings",
            "description": "Search hafs embeddings for relevant documents",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "context_id": {
                        "type": "string",
                        "description": "Context/domain ID",
                        "default": "default",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_graph",
            "description": "Query hafs knowledge graph",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search code using ripgrep",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Directory to search",
                        "default": ".",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
]
```

### 3. Integration with Consolidation Analyzer

```python
# src/agents/background/consolidation_analyzer.py (updated)

from hafs.services.local_ai_orchestrator import (
    LocalAIOrchestrator,
    InferenceRequest,
    RequestPriority,
)
from hafs.services.tool_executor import AVAILABLE_TOOLS

class ConsolidationAnalyzerAgent(BackgroundAgent):
    def __init__(self, config_path: str | Path | None = None, verbose: bool = False):
        super().__init__(config_path, verbose)

        # Initialize local AI orchestrator
        self.use_ai = self.config.tasks.get("use_ai_recommendations", True)
        if self.use_ai:
            self.ai_orchestrator = LocalAIOrchestrator(
                model=self.config.tasks.get("ai_model", "qwen2.5:7b"),
                max_concurrent_requests=1,  # CPU only
            )

    async def run(self) -> dict[str, Any]:
        """Execute consolidation analysis with AI."""
        results = {...}

        # ... existing rule-based analysis ...

        # AI recommendations with tool calling
        if self.use_ai:
            ai_recommendations = await self._generate_ai_recommendations(
                inventory, results
            )
            results["ai_recommendations"] = ai_recommendations

        return results

    async def _generate_ai_recommendations(
        self,
        inventory: dict[str, Any],
        analysis: dict[str, Any]
    ) -> str:
        """Generate AI recommendations with tool access."""

        # Create summary
        summary = self._create_inventory_summary(inventory, analysis)

        # Build prompt
        prompt = f"""You are a filesystem organization expert with access to tools for file operations and knowledge retrieval.

## Filesystem Summary

{summary}

## Available Tools

You can use these tools to gather more information:
- read_file(path): Read file contents
- search_embeddings(query): Search hafs knowledge for similar projects
- query_knowledge_graph(query): Query knowledge about file organization best practices
- search_code(pattern): Search code for specific patterns

## Your Task

Analyze this filesystem and provide 5-10 specific, actionable recommendations. Use tools to:
1. Check sample files to understand project structure
2. Query knowledge graph for organization best practices
3. Search for similar projects in hafs context

Provide concrete, file-path-specific recommendations for:
- File organization strategies
- Duplicate cleanup priorities
- Archival candidates
- Waste reduction opportunities

Format as markdown with clear headings and bullet points.
"""

        # Submit inference request with tools
        request = InferenceRequest(
            id=f"consolidation_{datetime.now().isoformat()}",
            priority=RequestPriority.SCHEDULED,
            prompt=prompt,
            tools=AVAILABLE_TOOLS,
            context_ids=["filesystem", "organization"],
            max_tokens=4096,
            temperature=0.7,
        )

        # Wait for result
        result = await self.ai_orchestrator.submit_request(request)

        if result.error:
            return f"Error generating AI recommendations: {result.error}"

        # Include tool call results in response
        response_parts = [result.response]

        if result.tool_calls:
            response_parts.append("\n\n## Tool Calls Made\n")
            for tool_call in result.tool_calls:
                response_parts.append(f"- {tool_call}")

        return "\n".join(response_parts)
```

## Configuration

```toml
# config/windows_filesystem_agents.toml

[agents.consolidationanalyzer.tasks]
use_ai_recommendations = true
ai_model = "qwen2.5:7b"  # Supports function calling
ollama_url = "http://localhost:11434"

# Context integration
use_hafs_context = true
context_ids = ["filesystem", "organization", "best_practices"]

# Resource limits
max_concurrent_ai_requests = 1  # CPU only
ai_request_timeout_seconds = 300  # 5 minutes

[services.local_ai_orchestrator]
enabled = true
model = "qwen2.5:7b"
max_concurrent_requests = 1

# Priority settings
training_priority = 1
interactive_priority = 2
analysis_priority = 3
scheduled_priority = 4

[services.local_ai_orchestrator.tools]
# Whitelist of allowed tools
enabled_tools = [
    "read_file",
    "search_embeddings",
    "query_knowledge_graph",
    "search_code",
]
```

## Usage Example

```python
# Training campaign (highest priority)
training_request = InferenceRequest(
    id="training_sample_123",
    priority=RequestPriority.TRAINING,  # Priority 1
    prompt="Generate ASM training sample...",
    tools=[],
    max_tokens=2048,
)

# Background analysis (lower priority)
analysis_request = InferenceRequest(
    id="filesystem_analysis",
    priority=RequestPriority.SCHEDULED,  # Priority 4
    prompt="Analyze filesystem...",
    tools=AVAILABLE_TOOLS,
    max_tokens=4096,
)

# Submit both - training will execute first
orchestrator = LocalAIOrchestrator()
await orchestrator.start()

training_result = await orchestrator.submit_request(training_request)
analysis_result = await orchestrator.submit_request(analysis_request)
```

## Benefits

### 1. Tool Calling
- **File Access**: Read actual files to make informed decisions
- **Knowledge Integration**: Query hafs knowledge graph for best practices
- **Embedding Search**: Find similar projects and solutions
- **Code Search**: Understand project structure

### 2. Context Management
- **Auto-Retrieval**: Automatically fetch relevant embeddings
- **Context Window**: Manage token limits with summarization
- **State Tracking**: Access scratchpad for current system state

### 3. Resource Orchestration
- **Priority Queue**: Training > Interactive > Analysis > Scheduled
- **Concurrency Control**: Limit concurrent requests (CPU-bound)
- **Monitoring**: Track active requests and queue depth

### 4. hafs Integration
- **Seamless**: Uses existing context_agent_daemon
- **Unified**: Same knowledge graph and embeddings as other agents
- **Consistent**: Same scratchpad and state management

## Model Requirements

**Recommended Model:** `qwen2.5:7b` or `qwen2.5:14b`

**Why Qwen 2.5?**
- Native function calling support
- Good at following instructions
- Fast inference on CPU
- Multilingual (helpful for ASM/Japanese)

**Alternatives:**
- `llama3.1:8b` (also supports tools)
- `mistral:7b-instruct` (decent tool calling)

**Install:**
```bash
ollama pull qwen2.5:7b
```

## Next Steps

1. **Implement LocalAIOrchestrator** (~2-3 hours)
2. **Create ToolExecutor** (~1-2 hours)
3. **Integrate with consolidation_analyzer** (~1 hour)
4. **Test with filesystem data** (~1 hour)
5. **Deploy to Windows** (~30 min)
6. **Monitor during training campaign** (~ongoing)

**Total:** ~5-7 hours implementation

## Success Criteria

- ✅ AI requests queued by priority
- ✅ Training requests never blocked by analysis
- ✅ Tool calls execute successfully
- ✅ Context retrieved from hafs system
- ✅ Recommendations reference actual files
- ✅ No GPU interference with training
- ✅ Graceful fallback if Ollama unavailable
