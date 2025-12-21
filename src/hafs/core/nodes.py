"""Distributed compute node management for HAFS.

Manages local and remote Ollama nodes via Tailscale for distributed AI inference.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import aiohttp

# Lazy import aiohttp
_aiohttp = None


def _ensure_aiohttp():
    """Lazy load aiohttp."""
    global _aiohttp
    if _aiohttp is None:
        try:
            import aiohttp as _aio
            _aiohttp = _aio
        except ImportError:
            raise ImportError(
                "aiohttp package not installed. "
                "Install with: pip install aiohttp"
            )
    return _aiohttp


# Use tomllib from stdlib (Python 3.11+) or tomli as fallback
try:
    import tomllib
except ImportError:
    import tomli as tomllib

from hafs.backends.ollama import OllamaBackend

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Status of a compute node."""
    UNKNOWN = "unknown"
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"


@dataclass
class ComputeNode:
    """Represents a compute node for distributed inference."""

    name: str
    host: str
    port: int = 11434
    node_type: str = "compute"
    platform: str = "unknown"
    capabilities: list[str] = field(default_factory=lambda: ["ollama"])
    models: list[str] = field(default_factory=list)
    prefer_for: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    services: list[dict[str, Any]] = field(default_factory=list)
    health_url: Optional[str] = None
    afs_root: Optional[str] = None
    sync_profiles: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: NodeStatus = NodeStatus.UNKNOWN
    latency_ms: int = 0
    gpu_memory_mb: Optional[int] = None
    last_check: float = 0.0
    error_message: Optional[str] = None

    @property
    def base_url(self) -> str:
        """Get the base URL for this node."""
        return f"http://{self.host}:{self.port}"

    @property
    def is_local(self) -> bool:
        """Check if this is a local node."""
        return self.host in ("localhost", "127.0.0.1", "::1")

    @property
    def has_gpu(self) -> bool:
        """Check if this node has GPU capability."""
        return "gpu" in self.capabilities

    def has_capability(self, capability: str) -> bool:
        """Check if a node advertises a capability."""
        return capability in self.capabilities

    def matches_preference(self, task_type: str) -> bool:
        """Check if this node prefers handling this task type."""
        return task_type in self.prefer_for

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "node_type": self.node_type,
            "platform": self.platform,
            "capabilities": self.capabilities,
            "models": self.models,
            "prefer_for": self.prefer_for,
            "tags": self.tags,
            "services": self.services,
            "health_url": self.health_url,
            "afs_root": self.afs_root,
            "sync_profiles": self.sync_profiles,
            "metadata": self.metadata,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "gpu_memory_mb": self.gpu_memory_mb,
        }


class NodeManager:
    """Manages distributed compute nodes for HAFS.

    Handles node discovery, health checking, and intelligent routing
    to optimal nodes based on task requirements.

    Example:
        manager = NodeManager()
        await manager.load_config()
        await manager.health_check_all()

        # Get best node for a coding task
        node = await manager.get_best_node(task_type="coding")
        if node:
            backend = manager.create_backend(node)
            await backend.start()
    """

    CONFIG_PATHS = [
        Path.home() / ".config" / "hafs" / "nodes.toml",
        Path.home() / ".hafs" / "nodes.toml",
    ]

    def __init__(self):
        """Initialize the node manager."""
        self._nodes: dict[str, ComputeNode] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._health_check_interval = 60  # seconds
        self._last_health_check = 0.0

    @property
    def nodes(self) -> list[ComputeNode]:
        """Get all registered nodes."""
        return list(self._nodes.values())

    @property
    def online_nodes(self) -> list[ComputeNode]:
        """Get all online nodes."""
        return [n for n in self._nodes.values() if n.status == NodeStatus.ONLINE]

    async def load_config(self, config_path: Optional[Path] = None) -> int:
        """Load node configuration from TOML file.

        Args:
            config_path: Optional path to config file.

        Returns:
            Number of nodes loaded.
        """
        # Find config file
        if config_path and config_path.exists():
            path = config_path
        else:
            path = None
            for p in self.CONFIG_PATHS:
                if p.exists():
                    path = p
                    break

        if not path:
            # Create default config with local node
            self._add_default_local_node()
            return 1

        try:
            with open(path, "rb") as f:
                config = tomllib.load(f)

            nodes = config.get("nodes", [])
            for node_config in nodes:
                node = ComputeNode(
                    name=node_config.get("name", "unnamed"),
                    host=node_config.get("host", "localhost"),
                    port=node_config.get("port", 11434),
                    node_type=node_config.get("node_type", node_config.get("type", "compute")),
                    platform=node_config.get("platform", "unknown"),
                    capabilities=node_config.get("capabilities", ["ollama"]),
                    models=node_config.get("models", []),
                    prefer_for=node_config.get("prefer_for", []),
                    tags=node_config.get("tags", []),
                    services=node_config.get("services", []),
                    health_url=node_config.get("health_url"),
                    afs_root=node_config.get("afs_root"),
                    sync_profiles=node_config.get("sync_profiles", []),
                    metadata=node_config.get("metadata", {}),
                    gpu_memory_mb=node_config.get("gpu_memory_mb"),
                )
                self._nodes[node.name] = node

            logger.info(f"Loaded {len(self._nodes)} nodes from {path}")
            return len(self._nodes)

        except Exception as e:
            logger.error(f"Failed to load node config from {path}: {e}")
            self._add_default_local_node()
            return 1

    def _add_default_local_node(self):
        """Add a default local Ollama node."""
        local = ComputeNode(
            name="local",
            host="localhost",
            port=11434,
            capabilities=["ollama"],
        )
        self._nodes["local"] = local
        logger.info("Added default local node")

    def add_node(self, node: ComputeNode) -> None:
        """Add a node to the manager.

        Args:
            node: Node to add.
        """
        self._nodes[node.name] = node
        logger.info(f"Added node: {node.name} at {node.base_url}")

    def remove_node(self, name: str) -> bool:
        """Remove a node by name.

        Args:
            name: Node name to remove.

        Returns:
            True if node was removed.
        """
        if name in self._nodes:
            del self._nodes[name]
            return True
        return False

    def get_node(self, name: str) -> Optional[ComputeNode]:
        """Get a node by name.

        Args:
            name: Node name.

        Returns:
            Node if found, None otherwise.
        """
        return self._nodes.get(name)

    async def _ensure_session(self):
        """Ensure HTTP session is available."""
        if self._session is None or self._session.closed:
            aiohttp = _ensure_aiohttp()
            timeout = aiohttp.ClientTimeout(total=10)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def health_check(self, node: ComputeNode) -> NodeStatus:
        """Check health of a single node.

        Args:
            node: Node to check.

        Returns:
            Node status.
        """
        await self._ensure_session()

        start_time = time.time()

        if not node.has_capability("ollama"):
            if not node.health_url:
                node.status = NodeStatus.UNKNOWN
                node.error_message = "No health_url configured"
                node.last_check = time.time()
                return node.status

            try:
                async with self._session.get(node.health_url) as resp:
                    latency = int((time.time() - start_time) * 1000)
                    node.latency_ms = latency
                    node.last_check = time.time()

                    if resp.status < 400:
                        node.status = NodeStatus.ONLINE
                        node.error_message = None
                    else:
                        node.status = NodeStatus.ERROR
                        node.error_message = f"HTTP {resp.status}"
                return node.status
            except asyncio.TimeoutError:
                node.status = NodeStatus.OFFLINE
                node.error_message = "Timeout"
                node.latency_ms = 10000
                return node.status
            except Exception as e:
                if "ClientConnectorError" in type(e).__name__:
                    node.status = NodeStatus.OFFLINE
                    node.error_message = "Connection refused"
                else:
                    node.status = NodeStatus.ERROR
                    node.error_message = str(e)
                return node.status

        try:
            async with self._session.get(f"{node.base_url}/api/tags") as resp:
                latency = int((time.time() - start_time) * 1000)
                node.latency_ms = latency
                node.last_check = time.time()

                if resp.status == 200:
                    data = await resp.json()
                    models = data.get("models", [])
                    node.models = [m.get("name", "") for m in models]
                    node.status = NodeStatus.ONLINE
                    node.error_message = None
                    logger.debug(f"Node {node.name} online, latency: {latency}ms, models: {len(node.models)}")
                else:
                    node.status = NodeStatus.ERROR
                    node.error_message = f"HTTP {resp.status}"

        except asyncio.TimeoutError:
            node.status = NodeStatus.OFFLINE
            node.error_message = "Timeout"
            node.latency_ms = 10000
        except Exception as e:
            if "ClientConnectorError" in type(e).__name__:
                node.status = NodeStatus.OFFLINE
                node.error_message = "Connection refused"
            else:
                node.status = NodeStatus.ERROR
                node.error_message = str(e)

        return node.status

    async def health_check_all(self) -> dict[str, NodeStatus]:
        """Check health of all nodes concurrently.

        Returns:
            Dict mapping node names to their status.
        """
        if not self._nodes:
            await self.load_config()

        tasks = [self.health_check(node) for node in self._nodes.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        status_map = {}
        for node, result in zip(self._nodes.values(), results):
            if isinstance(result, Exception):
                node.status = NodeStatus.ERROR
                node.error_message = str(result)
            status_map[node.name] = node.status

        self._last_health_check = time.time()
        return status_map

    async def get_best_node(
        self,
        task_type: Optional[str] = None,
        required_model: Optional[str] = None,
        prefer_gpu: bool = False,
        prefer_local: bool = False,
        prefer_remote: bool = False,
    ) -> Optional[ComputeNode]:
        """Get the best available node for a task.

        Args:
            task_type: Type of task (e.g., "coding", "reasoning", "fast").
            required_model: Specific model that must be available.
            prefer_gpu: Prefer nodes with GPU.
            prefer_local: Prefer local nodes.
            prefer_remote: Prefer remote (non-local) nodes.

        Returns:
            Best node for the task, or None if no suitable node found.
        """
        # Refresh health if stale
        if time.time() - self._last_health_check > self._health_check_interval:
            await self.health_check_all()

        # Filter to online nodes
        candidates = [n for n in self.online_nodes if n.has_capability("ollama")]

        if not candidates:
            logger.warning("No online nodes available")
            return None

        # Filter by required model
        if required_model:
            candidates = [
                n for n in candidates
                if self.resolve_model_for_node(n, required_model) is not None
            ]
            if not candidates:
                logger.warning(f"No nodes have model {required_model}")
                return None

        # Score candidates
        def score_node(node: ComputeNode) -> float:
            score = 0.0

            # Prefer nodes that match task type
            if task_type and node.matches_preference(task_type):
                score += 100

            # Prefer GPU nodes if requested
            if prefer_gpu and node.has_gpu:
                score += 50

            # Prefer local nodes if requested
            if prefer_local and node.is_local:
                score += 30

            # Prefer remote nodes if requested
            if prefer_remote and not node.is_local:
                score += 30

            # Prefer lower latency
            score -= node.latency_ms / 100

            # Prefer nodes with more models (more capable)
            score += len(node.models) * 2

            return score

        # Sort by score (highest first)
        candidates.sort(key=score_node, reverse=True)

        return candidates[0] if candidates else None

    def create_backend(
        self,
        node: ComputeNode,
        model: Optional[str] = None,
    ) -> OllamaBackend:
        """Create an OllamaBackend for a node.

        Args:
            node: Node to create backend for.
            model: Model to use (defaults to first available).

        Returns:
            Configured OllamaBackend.
        """
        # Pick model
        if model:
            resolved = self.resolve_model_for_node(node, model)
            if resolved:
                use_model = resolved
            elif node.models:
                use_model = node.models[0]
            else:
                use_model = model
        elif node.models:
            # Prefer llama3 or codellama if available
            for preferred in ["llama3", "codellama", "mistral"]:
                for m in node.models:
                    if preferred in m:
                        use_model = m
                        break
                else:
                    continue
                break
            else:
                use_model = node.models[0]
        else:
            use_model = "llama3:latest"  # Default fallback (available model)

        return OllamaBackend(
            host=node.host,
            port=node.port,
            model=use_model,
        )

    def resolve_model_for_node(self, node: ComputeNode, required_model: str) -> Optional[str]:
        """Resolve a compatible model name available on a node."""
        if not required_model:
            return None

        required_norm = required_model.strip().lower()
        if not required_norm:
            return None

        for model in node.models:
            if model.lower() == required_norm:
                return model

        required_base = required_norm.split(":", 1)[0]
        for model in node.models:
            if model.lower().split(":", 1)[0] == required_base:
                return model

        for model in node.models:
            if required_base in model.lower():
                return model

        return None

    async def discover_tailscale_nodes(self) -> list[ComputeNode]:
        """Discover Ollama nodes on Tailscale network.

        Scans common Tailscale IPs for Ollama instances.

        Returns:
            List of discovered nodes.
        """
        discovered = []

        # Get Tailscale IPs (typically 100.x.x.x)
        # This is a simple scan - could be enhanced with Tailscale API
        await self._ensure_session()

        # Try to connect to status endpoint to get peers
        try:
            # Check if Tailscale is running locally
            async with self._session.get("http://localhost:41112/localapi/v0/status") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    peers = data.get("Peer", {})

                    for peer_id, peer_info in peers.items():
                        addresses = peer_info.get("TailscaleIPs", [])
                        hostname = peer_info.get("HostName", "unknown")

                        for addr in addresses:
                            # Try Ollama on default port
                            node = ComputeNode(
                                name=f"tailscale-{hostname}",
                                host=addr,
                                port=11434,
                                node_type="compute",
                                platform="tailscale",
                                capabilities=["ollama", "tailscale"],
                            )

                            # Quick health check
                            status = await self.health_check(node)
                            if status == NodeStatus.ONLINE:
                                discovered.append(node)
                                self._nodes[node.name] = node
                                logger.info(f"Discovered Tailscale node: {node.name} at {addr}")

        except Exception as e:
            logger.debug(f"Tailscale discovery failed: {e}")

        return discovered

    async def close(self):
        """Close the node manager and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()

    def summary(self) -> str:
        """Get a summary of node status.

        Returns:
            Human-readable summary string.
        """
        lines = [f"Node Manager: {len(self._nodes)} nodes"]

        for node in self._nodes.values():
            status_emoji = {
                NodeStatus.ONLINE: "🟢",
                NodeStatus.OFFLINE: "🔴",
                NodeStatus.BUSY: "🟡",
                NodeStatus.ERROR: "⚠️",
                NodeStatus.UNKNOWN: "⚪",
            }.get(node.status, "❓")

            gpu = " [GPU]" if node.has_gpu else ""
            local = " [local]" if node.is_local else ""
            node_type = f" [{node.node_type}]" if node.node_type else ""
            platform = f" [{node.platform}]" if node.platform else ""
            models = f" ({len(node.models)} models)" if node.models else ""
            latency = f" {node.latency_ms}ms" if node.latency_ms > 0 else ""

            lines.append(
                f"  {status_emoji} {node.name}: {node.host}:{node.port}{gpu}{local}"
                f"{node_type}{platform}{models}{latency}"
            )

        return "\n".join(lines)


# Global node manager instance
node_manager = NodeManager()
