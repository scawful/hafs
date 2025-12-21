"""Unified Multi-Provider Orchestrator v2.

Intelligent orchestrator that routes requests to optimal backends based on:
- Task requirements (context size, model capabilities)
- Provider availability (quota, health, latency)
- Cost optimization (prefer local/cheap providers)
- Fallback chains for reliability

Supports:
- Gemini (google-genai SDK)
- Anthropic (Claude via API)
- OpenAI (GPT via API)
- Llama.cpp (OpenAI-compatible API)
- Ollama (local and distributed nodes via Tailscale)
- halext-org (backend AI gateway)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Optional

from hafs.backends import (
    AnthropicBackend,
    BackendRegistry,
    BaseChatBackend,
    LlamaCppBackend,
    OllamaBackend,
    OpenAIBackend,
)
from hafs.core.config import hafs_config
from hafs.core.nodes import NodeManager, NodeStatus
from hafs.core.quota import quota_manager

logger = logging.getLogger(__name__)

# Lazy import for history logger to avoid circular imports
_history_logger = None


def _get_history_logger():
    """Get the global history logger instance."""
    global _history_logger
    if _history_logger is None:
        try:
            from pathlib import Path
            from hafs.core.history.logger import HistoryLogger
            context_root = Path.home() / ".context"
            history_dir = context_root / "history"
            _history_logger = HistoryLogger(history_dir)
        except Exception as e:
            logger.warning(f"Could not initialize history logger: {e}")
    return _history_logger

# Lazy imports
genai = None


def _ensure_genai():
    """Lazy load google-genai SDK."""
    global genai
    if genai is None:
        try:
            import google.genai as _genai
            genai = _genai
        except ImportError:
            pass
    return genai


class Provider(Enum):
    """Available AI providers."""

    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    LLAMACPP = "llamacpp"
    OLLAMA = "ollama"
    HALEXT = "halext"


class TaskTier(Enum):
    """Task tiers for routing decisions."""

    REASONING = "reasoning"  # Complex multi-step reasoning
    FAST = "fast"            # Quick responses, low latency
    CODING = "coding"        # Code generation and analysis
    CREATIVE = "creative"    # Creative writing, brainstorming
    RESEARCH = "research"    # Deep research, large context
    LOCAL = "local"          # Force local execution
    CHEAP = "cheap"          # Minimize cost


@dataclass
class ProviderConfig:
    """Configuration for a provider."""

    provider: Provider
    enabled: bool = True
    api_key_env: Optional[str] = None
    default_model: Optional[str] = None
    priority: int = 50  # Lower = higher priority
    cost_per_1k_tokens: float = 0.0
    max_context_tokens: int = 100000
    supports_streaming: bool = True


@dataclass
class RouteResult:
    """Result of routing a request."""

    provider: Provider
    model: str
    backend: Optional[BaseChatBackend] = None
    node_name: Optional[str] = None
    latency_estimate_ms: int = 0
    cost_estimate: float = 0.0


@dataclass
class GenerationResult:
    """Result of a generation request."""

    content: str
    provider: Provider
    model: str
    tokens_used: int = 0
    latency_ms: int = 0
    fallback_used: bool = False
    error: Optional[str] = None
    thought_content: Optional[str] = None  # Gemini 3 thought/reasoning traces
    raw_parts: Optional[list] = None  # All response parts for debugging


class UnifiedOrchestrator:
    """Multi-provider orchestrator with intelligent routing.

    Example:
        orchestrator = UnifiedOrchestrator()
        await orchestrator.initialize()

        # Generate with automatic routing
        result = await orchestrator.generate(
            prompt="Explain quantum computing",
            tier=TaskTier.REASONING
        )

        # Force specific provider
        result = await orchestrator.generate(
            prompt="Quick question",
            provider=Provider.OLLAMA
        )

        # Stream response
        async for chunk in orchestrator.stream_generate(
            prompt="Write a story",
            tier=TaskTier.CREATIVE
        ):
            print(chunk, end="")
    """

    # Available Gemini models (December 2025)
    GEMINI_MODELS = {
        # Gemini 3 series (latest - December 2025)
        "gemini-3-flash-preview": "Gemini 3 Flash - Pro-grade reasoning at Flash speed, 1M context",
        "gemini-3-pro-preview": "Gemini 3 Pro - Best reasoning, deep thinking, 1M context",
        # Gemini 2.5 series
        "gemini-2.5-flash": "Fast responses, 1M context",
        "gemini-2.5-pro": "Strong reasoning, 1M context",
        # Legacy
        "gemini-2.0-flash": "Previous gen fast model",
    }

    # Anthropic models (December 2025)
    ANTHROPIC_MODELS = {
        "claude-opus-4-5-20251101": "Claude Opus 4.5 - Best reasoning, 200k context",
        "claude-sonnet-4-20250514": "Claude Sonnet 4 - Balanced performance",
        "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet - Fast and capable",
        "claude-3-haiku-20240307": "Claude 3 Haiku - Fastest",
    }

    # OpenAI models (December 2025)
    OPENAI_MODELS = {
        "gpt-5.2": "GPT-5.2 - Latest flagship model",
        "gpt-5.2-mini": "GPT-5.2 Mini - Fast and efficient",
        "gpt-4-turbo": "GPT-4 Turbo - Previous flagship",
        "gpt-4o-mini": "GPT-4o Mini - Fast responses",
    }

    # Ollama models available on nodes (medical-mechanica, local)
    OLLAMA_MODELS = {
        # Coding models
        "qwen2.5-coder:7b": "Qwen 2.5 Coder 7B - Excellent for code",
        "qwen2.5-coder:14b": "Qwen 2.5 Coder 14B - Larger code model",
        "deepseek-coder-v2-lite": "DeepSeek Coder V2 Lite - MoE coding",
        "codellama:7b": "CodeLlama 7B - Meta's code model",
        "codellama:13b": "CodeLlama 13B - Larger CodeLlama",
        # General models
        "qwen3:8b": "Qwen3 8B - Best general local",
        "deepseek-r1:8b": "DeepSeek R1 8B - Good reasoning",
        "llama3:latest": "Llama 3 - Reliable general purpose",
        "gemma3:4b": "Gemma 3 4B - Fastest local",
        "mistral:7b": "Mistral 7B - Efficient general model",
        # Embedding models
        "embeddinggemma": "Embedding Gemma - 768-dim embeddings",
        "nomic-embed-text": "Nomic Embed - Text embeddings",
    }

    # Provider configurations with defaults
    DEFAULT_CONFIGS = {
        Provider.GEMINI: ProviderConfig(
            provider=Provider.GEMINI,
            api_key_env="GEMINI_API_KEY",
            default_model="gemini-3-flash-preview",
            priority=10,  # Prioritize Gemini
            cost_per_1k_tokens=0.0005,  # $0.50/1M input
            max_context_tokens=1000000,
        ),
        Provider.ANTHROPIC: ProviderConfig(
            provider=Provider.ANTHROPIC,
            api_key_env="ANTHROPIC_API_KEY",
            default_model="claude-opus-4-5-20251101",  # Latest Opus 4.5
            priority=30,
            cost_per_1k_tokens=0.015,
            max_context_tokens=200000,
        ),
        Provider.OPENAI: ProviderConfig(
            provider=Provider.OPENAI,
            api_key_env="OPENAI_API_KEY",
            default_model="gpt-5.2",  # Latest GPT-5.2
            priority=35,
            cost_per_1k_tokens=0.02,
            max_context_tokens=256000,
        ),
        Provider.LLAMACPP: ProviderConfig(
            provider=Provider.LLAMACPP,
            enabled=True,
            api_key_env="LLAMACPP_API_KEY",
            default_model="qwen3-14b",
            priority=15,  # Prefer local llama.cpp
            cost_per_1k_tokens=0.0,
            max_context_tokens=8192,
        ),
        Provider.OLLAMA: ProviderConfig(
            provider=Provider.OLLAMA,
            enabled=False,  # Disable by default to avoid discovery overhead
            api_key_env=None,  # No API key needed
            default_model="llama3:latest",  # Use available model
            priority=20,  # Prefer local
            cost_per_1k_tokens=0.0,  # Free
            max_context_tokens=8192,
        ),
        Provider.HALEXT: ProviderConfig(
            provider=Provider.HALEXT,
            api_key_env="HALEXT_API_KEY",
            default_model="default",
            priority=45,
            cost_per_1k_tokens=0.0,
            max_context_tokens=100000,
        ),
    }

    # Tier to provider/model mappings
    # NOTE: Ollama models (local) should only be used for small, fast tasks
    # Available Ollama models: llama3:latest, qwen3:8b, deepseek-r1:8b, gemma3:4b, qwen2.5-coder:7b
    TIER_ROUTES = {
        TaskTier.REASONING: [
            (Provider.GEMINI, "gemini-3-flash-preview"),  # Preferred fast reasoning
            (Provider.GEMINI, "gemini-3-pro-preview"),  # Best reasoning Dec 2025
            (Provider.ANTHROPIC, "claude-opus-4-5-20251101"),  # Opus 4.5
            (Provider.OPENAI, "gpt-5.2"),  # GPT-5.2
            (Provider.OLLAMA, "deepseek-r1:8b"),  # Local reasoning fallback
        ],
        TaskTier.FAST: [
            (Provider.GEMINI, "gemini-3-flash-preview"),  # Gemini 3 Flash
            (Provider.LLAMACPP, "qwen3-14b"),  # Local llama.cpp GPU
            (Provider.OLLAMA, "gemma3:4b"),  # Fastest local model for quick tasks
            (Provider.OPENAI, "gpt-5.2-mini"),  # GPT-5.2 Mini
            (Provider.ANTHROPIC, "claude-3-haiku-20240307"),  # Fastest Claude
        ],
        TaskTier.CODING: [
            (Provider.GEMINI, "gemini-3-flash-preview"),  # Gemini 3 Flash - great for code
            (Provider.LLAMACPP, "qwen3-14b"),  # Local llama.cpp GPU
            (Provider.ANTHROPIC, "claude-opus-4-5-20251101"),  # Opus 4.5 for complex code
            (Provider.OLLAMA, "qwen2.5-coder:14b"),  # Best local coding (medical-mechanica)
            (Provider.OLLAMA, "qwen2.5-coder:7b"),  # Smaller local coding
            (Provider.OLLAMA, "deepseek-coder-v2-lite"),  # DeepSeek Coder
        ],
        TaskTier.CREATIVE: [
            (Provider.GEMINI, "gemini-3-flash-preview"),  # Preferred creative
            (Provider.GEMINI, "gemini-3-pro-preview"),  # Best creative
            (Provider.ANTHROPIC, "claude-opus-4-5-20251101"),  # Opus 4.5
            (Provider.OPENAI, "gpt-5.2"),  # GPT-5.2
        ],
        TaskTier.RESEARCH: [
            (Provider.GEMINI, "gemini-3-flash-preview"),
            (Provider.GEMINI, "gemini-3-pro-preview"),  # Deep thinking, 1M context
            (Provider.ANTHROPIC, "claude-opus-4-5-20251101"),  # Opus 4.5, 200k context
            (Provider.OPENAI, "gpt-5.2"),  # GPT-5.2, 256k context
        ],
        TaskTier.LOCAL: [
            # All available local Ollama models (medical-mechanica + local)
            (Provider.LLAMACPP, "qwen3-14b"),  # llama.cpp local GPU node
            (Provider.OLLAMA, "qwen2.5-coder:14b"),  # Best for 16GB GPU
            (Provider.OLLAMA, "qwen3:8b"),  # Best general local
            (Provider.OLLAMA, "deepseek-r1:8b"),  # Good reasoning
            (Provider.OLLAMA, "deepseek-coder-v2-lite"),  # Code-focused MoE
            (Provider.OLLAMA, "qwen2.5-coder:7b"),  # Smaller coder
            (Provider.OLLAMA, "llama3:latest"),  # Reliable fallback
            (Provider.OLLAMA, "mistral:7b"),  # Efficient general
            (Provider.OLLAMA, "gemma3:4b"),  # Fastest
        ],
        TaskTier.CHEAP: [
            (Provider.GEMINI, "gemini-3-flash-preview"),  # Very cheap
            (Provider.LLAMACPP, "qwen3-14b"),  # Free local llama.cpp
            (Provider.OLLAMA, "gemma3:4b"),  # Free and fast
            (Provider.OLLAMA, "llama3:latest"),  # Free fallback
            (Provider.OPENAI, "gpt-5.2-mini"),  # Cheap GPT
        ],
    }

    # Ollama model selection by task characteristics
    # Maps task traits to best local model
    OLLAMA_MODEL_SELECTION = {
        "coding": "qwen2.5-coder:7b",  # Best for code generation/analysis
        "fast": "gemma3:4b",           # Fastest response time
        "reasoning": "deepseek-r1:8b", # Best local reasoning
        "general": "qwen3:8b",         # Best general purpose
        "fallback": "llama3:latest",   # Most reliable
    }

    # Max tokens for Ollama use (small tasks only)
    OLLAMA_MAX_PROMPT_TOKENS = 2000  # Only use Ollama for prompts < 2k tokens

    @staticmethod
    def _parse_bool(value: Optional[str]) -> bool:
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _parse_provider(value: Optional[str]) -> Optional[Provider]:
        if not value:
            return None
        try:
            return Provider(value.strip().lower())
        except ValueError:
            return None

    @staticmethod
    def _parse_rotation(value: Optional[str]) -> list[tuple[Provider, str]]:
        if not value:
            return []
        rotation: list[tuple[Provider, str]] = []
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                continue
            provider_raw, model = item.split(":", 1)
            provider = UnifiedOrchestrator._parse_provider(provider_raw.strip())
            if provider and model.strip():
                rotation.append((provider, model.strip()))
        return rotation

    def __init__(
        self,
        configs: Optional[dict[Provider, ProviderConfig]] = None,
        prefer_local: bool = False,
        max_cost_per_request: Optional[float] = None,
        log_thoughts: bool = True,
    ):
        """Initialize the orchestrator.

        Args:
            configs: Optional custom provider configs.
            prefer_local: If True, prefer local providers when possible.
            max_cost_per_request: Optional cost limit per request.
            log_thoughts: If True, log thought traces to history.
        """
        self.configs = configs or self.DEFAULT_CONFIGS.copy()
        self.prefer_local = prefer_local
        self.max_cost_per_request = max_cost_per_request
        self.log_thoughts = log_thoughts
        self.prefer_gpu_nodes = self._parse_bool(os.environ.get("HAFS_PREFER_GPU_NODES"))
        self.prefer_remote_nodes = self._parse_bool(os.environ.get("HAFS_PREFER_REMOTE_NODES"))

        self._override_provider = self._parse_provider(os.environ.get("HAFS_MODEL_PROVIDER"))
        self._override_model = os.environ.get("HAFS_MODEL_MODEL") or None
        self._rotation = self._parse_rotation(os.environ.get("HAFS_MODEL_ROTATION"))
        self._llamacpp_config = self._load_llamacpp_config()
        self._apply_llamacpp_config()
        self._apply_provider_env_overrides()

        # Backend instances (lazy initialized)
        self._backends: dict[Provider, BaseChatBackend] = {}
        self._gemini_client = None
        self._halext_client = None
        self._node_manager: Optional[NodeManager] = None

        # Provider availability cache
        self._provider_health: dict[Provider, bool] = {}
        self._last_health_check: dict[Provider, float] = {}

        self._initialized = False

    async def initialize(self):
        """Initialize the orchestrator and check provider availability."""
        if self._initialized:
            return

        logger.info("Initializing UnifiedOrchestrator v2...")
        print("Initializing UnifiedOrchestrator v2...")

        # Initialize node manager for Ollama routing
        print("Initializing NodeManager...")
        self._node_manager = NodeManager()
        await self._node_manager.load_config()
        try:
            await self._node_manager.health_check_all()
        except Exception as exc:
            logger.debug("Node health check failed: %s", exc)

        # Check which providers are available
        print("Checking provider availability...")
        await self._check_provider_availability()

        self._initialized = True
        logger.info(f"UnifiedOrchestrator ready. Available: {list(self._provider_health.keys())}")
        print(f"UnifiedOrchestrator ready. Available: {list(self._provider_health.keys())}")

    async def _check_provider_availability(self):
        """Check which providers are available."""
        for provider, config in self.configs.items():
            if not config.enabled:
                continue

            # Check if explicitly disabled in global config
            backend_cfg = hafs_config.get_backend_config(provider.value)
            if backend_cfg and not backend_cfg.enabled:
                self._provider_health[provider] = False
                continue

            available = False

            if provider == Provider.GEMINI:
                api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("AISTUDIO_API_KEY")
                # Fallback to config file
                if not api_key:
                    api_key = hafs_config.aistudio_api_key
                
                if api_key and _ensure_genai():
                    try:
                        self._gemini_client = genai.Client(api_key=api_key)
                        available = True
                    except Exception as e:
                        logger.warning(f"Gemini init failed: {e}")

            elif provider == Provider.ANTHROPIC:
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                available = bool(api_key)

            elif provider == Provider.OPENAI:
                api_key = os.environ.get("OPENAI_API_KEY")
                base_url = os.environ.get("OPENAI_BASE_URL")
                available = bool(api_key or base_url)

            elif provider == Provider.LLAMACPP:
                if not self._parse_bool(os.environ.get("HAFS_ENABLE_LLAMACPP")) and not self._llamacpp_configured():
                    self._provider_health[provider] = False
                    continue
                try:
                    cfg = self._llamacpp_config
                    max_tokens = getattr(cfg, "max_tokens", 128) if cfg else 128
                    temperature = getattr(cfg, "temperature", 0.0) if cfg else 0.0
                    backend = self._build_llamacpp_backend(
                        config.default_model or "qwen3-14b",
                        max_tokens,
                        temperature,
                        None,
                    )
                    health = await backend.check_health()
                    available = health.get("status") in {"online", "ok"}
                    await backend.stop()
                except Exception:
                    available = False

            elif provider == Provider.OLLAMA:
                # Prefer any reachable Ollama node (local or remote)
                if self._node_manager:
                    try:
                        statuses = await self._node_manager.health_check_all()
                        available = any(status == NodeStatus.ONLINE for status in statuses.values())
                    except Exception:
                        available = False

                if not available:
                    try:
                        backend = OllamaBackend()
                        health = await backend.check_health()
                        available = health.get("status") in {"online", "ok"}
                        await backend.stop()
                    except Exception:
                        available = False

            elif provider == Provider.HALEXT:
                # Check halext-org availability
                api_key = os.environ.get("HALEXT_USERNAME")
                available = bool(api_key)

            if available:
                self._provider_health[provider] = True
                logger.info(f"Provider available: {provider.value}")
            else:
                self._provider_health[provider] = False

    def select_ollama_model(self, task_type: str = "general") -> str:
        """Select the best Ollama model for a given task type.

        Args:
            task_type: Type of task (coding, fast, reasoning, general).

        Returns:
            Model name suitable for the task.
        """
        return self.OLLAMA_MODEL_SELECTION.get(
            task_type,
            self.OLLAMA_MODEL_SELECTION["fallback"]
        )

    def _apply_provider_env_overrides(self) -> None:
        """Enable/disable providers based on environment overrides."""
        enable_local = self._parse_bool(os.environ.get("HAFS_ENABLE_LOCAL_MODELS"))
        disable_local = self._parse_bool(os.environ.get("HAFS_DISABLE_LOCAL_MODELS"))
        enable_ollama = self._parse_bool(os.environ.get("HAFS_ENABLE_OLLAMA"))
        disable_ollama = self._parse_bool(os.environ.get("HAFS_DISABLE_OLLAMA"))
        enable_llamacpp = self._parse_bool(os.environ.get("HAFS_ENABLE_LLAMACPP"))
        disable_llamacpp = self._parse_bool(os.environ.get("HAFS_DISABLE_LLAMACPP"))

        if enable_local:
            for provider in (Provider.OLLAMA, Provider.LLAMACPP):
                if provider in self.configs:
                    self.configs[provider].enabled = True
        if disable_local:
            for provider in (Provider.OLLAMA, Provider.LLAMACPP):
                if provider in self.configs:
                    self.configs[provider].enabled = False

        if Provider.OLLAMA in self.configs:
            if enable_ollama:
                self.configs[Provider.OLLAMA].enabled = True
            if disable_ollama:
                self.configs[Provider.OLLAMA].enabled = False

        if Provider.LLAMACPP in self.configs:
            if enable_llamacpp:
                self.configs[Provider.LLAMACPP].enabled = True
            if disable_llamacpp:
                self.configs[Provider.LLAMACPP].enabled = False

    def _llamacpp_configured(self) -> bool:
        """Check whether llama.cpp connection settings are configured."""
        keys = (
            "LLAMACPP_BASE_URL",
            "LLAMA_CPP_BASE_URL",
            "LLAMACPP_HOST",
            "LLAMA_CPP_HOST",
            "LLAMACPP_PORT",
            "LLAMA_CPP_PORT",
        )
        return any(os.environ.get(k) for k in keys)

    def _load_llamacpp_config(self) -> Optional[Any]:
        try:
            return hafs_config.llamacpp
        except Exception:
            return None

    def _resolve_llamacpp_api_key(self) -> Optional[str]:
        cfg = self._llamacpp_config
        if not cfg:
            return None
        env_name = getattr(cfg, "api_key_env", None)
        if env_name:
            return os.environ.get(env_name) or None
        return None

    def _apply_llamacpp_config(self) -> None:
        cfg = self._llamacpp_config
        if not cfg:
            return
        provider_cfg = self.configs.get(Provider.LLAMACPP)
        if not provider_cfg:
            return
        provider_cfg.enabled = bool(getattr(cfg, "enabled", True))
        model = getattr(cfg, "model", None)
        if model:
            provider_cfg.default_model = model
        context_size = getattr(cfg, "context_size", None)
        if context_size:
            provider_cfg.max_context_tokens = context_size

    def _build_llamacpp_backend(
        self,
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
    ) -> LlamaCppBackend:
        cfg = self._llamacpp_config
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system_prompt": system_prompt,
        }
        if cfg:
            base_url = getattr(cfg, "base_url", None)
            host = getattr(cfg, "host", None)
            port = getattr(cfg, "port", None)
            timeout = getattr(cfg, "timeout_seconds", None)
            context_size = getattr(cfg, "context_size", None)
            if base_url:
                kwargs["base_url"] = base_url
            if host:
                kwargs["host"] = host
            if port:
                kwargs["port"] = port
            if timeout:
                kwargs["timeout"] = timeout
            if context_size:
                kwargs["context_size"] = context_size

            api_key = self._resolve_llamacpp_api_key()
            if api_key:
                kwargs["api_key"] = api_key

        return LlamaCppBackend(**kwargs)

    def should_use_ollama(self, prompt: str, tier: TaskTier) -> bool:
        """Check if Ollama should be used for this request.

        Ollama is only suitable for small, fast tasks. Returns False
        for complex reasoning, research, or long prompts.

        Args:
            prompt: The prompt to evaluate.
            tier: The task tier.

        Returns:
            True if Ollama is appropriate for this request.
        """
        # Never use Ollama for complex tasks
        if tier in (TaskTier.REASONING, TaskTier.RESEARCH, TaskTier.CREATIVE):
            return False

        # Check prompt length (rough token estimate)
        estimated_tokens = len(prompt) // 4
        if estimated_tokens > self.OLLAMA_MAX_PROMPT_TOKENS:
            logger.debug(
                f"Prompt too long for Ollama ({estimated_tokens} tokens > "
                f"{self.OLLAMA_MAX_PROMPT_TOKENS})"
            )
            return False

        return True

    def _get_backend(self, provider: Provider, model: str) -> BaseChatBackend:
        """Get or create a backend instance for a provider.

        Args:
            provider: The provider to get backend for.
            model: The model to configure.

        Returns:
            Configured backend instance.
        """
        key = (provider, model)

        if provider == Provider.ANTHROPIC:
            return AnthropicBackend(model=model)
        elif provider == Provider.OPENAI:
            return OpenAIBackend(model=model)
        elif provider == Provider.LLAMACPP:
            cfg = self._llamacpp_config
            max_tokens = getattr(cfg, "max_tokens", 4096) if cfg else 4096
            temperature = getattr(cfg, "temperature", 0.7) if cfg else 0.7
            return self._build_llamacpp_backend(model, max_tokens, temperature, None)
        elif provider == Provider.OLLAMA:
            return OllamaBackend(model=model)

        raise ValueError(f"No backend for provider: {provider}")

    async def route(
        self,
        prompt: str,
        tier: TaskTier = TaskTier.FAST,
        provider: Optional[Provider] = None,
        model: Optional[str] = None,
        require_streaming: bool = False,
        min_context_tokens: Optional[int] = None,
    ) -> RouteResult:
        """Route a request to the optimal provider.

        Args:
            prompt: The prompt (for token estimation).
            tier: Task tier for routing.
            provider: Force specific provider.
            require_streaming: If True, only use streaming-capable providers.
            min_context_tokens: Minimum context size required.

        Returns:
            Routing result with selected provider/model.
        """
        if not self._initialized:
            await self.initialize()

        print(f"DEBUG: Routing request for tier {tier}", flush=True)

        estimated_tokens = len(prompt) // 4

        # If specific provider requested, validate and return
        if provider:
            if provider not in self._provider_health:
                raise ValueError(f"Provider not available: {provider}")

            config = self.configs[provider]
            chosen_model = model or config.default_model
            return RouteResult(
                provider=provider,
                model=chosen_model,
                latency_estimate_ms=100,
            )

        # Get candidates from tier routes
        candidates = self.TIER_ROUTES.get(tier, self.TIER_ROUTES[TaskTier.FAST])

        # Filter by availability and requirements
        valid_candidates = []
        for prov, model in candidates:
            if prov not in self._provider_health:
                continue

            config = self.configs[prov]

            # Check streaming requirement
            if require_streaming and not config.supports_streaming:
                continue

            # Check context size
            if min_context_tokens and config.max_context_tokens < min_context_tokens:
                continue

            # Check cost limit
            if self.max_cost_per_request:
                est_cost = (estimated_tokens / 1000) * config.cost_per_1k_tokens
                if est_cost > self.max_cost_per_request:
                    continue

            # Skip Ollama for large prompts or complex tasks
            if prov == Provider.OLLAMA and not self.should_use_ollama(prompt, tier):
                logger.debug(f"Skipping Ollama for {tier.value} task (prompt too long or complex)")
                continue

            valid_candidates.append((prov, model, config))

        if not valid_candidates:
            raise RuntimeError(f"No providers available for tier {tier}")

        # Sort by priority (and prefer local if configured)
        def score(item):
            prov, model, config = item
            score = config.priority

            if self.prefer_local and prov in {Provider.OLLAMA, Provider.LLAMACPP}:
                score -= 20

            return score

        valid_candidates.sort(key=score)

        # Select best candidate
        best_provider, best_model, best_config = valid_candidates[0]

        # For Ollama, try to find optimal node
        node_name = None
        if best_provider == Provider.OLLAMA and self._node_manager:
            task_type = tier.value if tier != TaskTier.FAST else None
            node = await self._node_manager.get_best_node(
                task_type=task_type,
                required_model=best_model,
                prefer_gpu=self.prefer_gpu_nodes,
                prefer_local=self.prefer_local,
                prefer_remote=self.prefer_remote_nodes,
            )
            if not node:
                node = await self._node_manager.get_best_node(
                    task_type=task_type,
                    prefer_gpu=self.prefer_gpu_nodes,
                    prefer_local=self.prefer_local,
                    prefer_remote=self.prefer_remote_nodes,
                )

            if node:
                resolved = self._node_manager.resolve_model_for_node(node, best_model)
                if resolved:
                    best_model = resolved
                elif node.models:
                    best_model = node.models[0]
                node_name = node.name

        return RouteResult(
            provider=best_provider,
            model=best_model,
            node_name=node_name,
            latency_estimate_ms=100,  # Could be enhanced with actual measurements
            cost_estimate=(estimated_tokens / 1000) * best_config.cost_per_1k_tokens,
        )

    async def generate(
        self,
        prompt: str,
        tier: TaskTier = TaskTier.FAST,
        provider: Optional[Provider] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> GenerationResult:
        """Generate content with automatic routing and fallback.

        Args:
            prompt: The prompt to send.
            tier: Task tier for routing.
            provider: Force specific provider.
            system_prompt: Optional system prompt.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Returns:
            Generation result with content and metadata.
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        rotation = self._rotation if not provider and not model else []

        # Attempt explicit rotation first
        if rotation:
            for rot_provider, rot_model in rotation:
                if not self._provider_health.get(rot_provider, False):
                    continue
                if rot_provider == Provider.OLLAMA and not self.should_use_ollama(prompt, tier):
                    continue
                try:
                    if rot_provider == Provider.OLLAMA:
                        rot_route = await self.route(
                            prompt,
                            tier=tier,
                            provider=rot_provider,
                            model=rot_model,
                        )
                        result = await self._generate_with_provider(
                            rot_provider,
                            rot_route.model,
                            prompt,
                            system_prompt,
                            max_tokens,
                            temperature,
                            rot_route.node_name,
                        )
                    else:
                        result = await self._generate_with_provider(
                            rot_provider,
                            rot_model,
                            prompt,
                            system_prompt,
                            max_tokens,
                            temperature,
                        )

                    gen_result = GenerationResult(
                        content=result["content"],
                        provider=rot_provider,
                        model=rot_route.model if rot_provider == Provider.OLLAMA else rot_model,
                        tokens_used=result.get("tokens_used", 0),
                        latency_ms=int((time.time() - start_time) * 1000),
                        thought_content=result.get("thought_content"),
                        raw_parts=result.get("raw_parts"),
                    )
                    self._log_thought_if_present(gen_result, prompt)
                    return gen_result
                except Exception:
                    continue

        override_provider = provider or self._override_provider
        override_model = model or (self._override_model if override_provider else None)

        if override_provider and not self._provider_health.get(override_provider, False):
            logger.warning("Override provider unavailable; falling back to routing.")
            override_provider = None
            override_model = None

        route = await self.route(
            prompt,
            tier=tier,
            provider=override_provider,
            model=override_model,
        )
        print(f"DEBUG: Selected route provider: {route.provider}, model: {route.model}", flush=True)

        errors = []
        fallback_used = False

        # Try primary route
        try:
            result = await self._generate_with_provider(
                route.provider,
                route.model,
                prompt,
                system_prompt,
                max_tokens,
                temperature,
                route.node_name,
            )

            gen_result = GenerationResult(
                content=result["content"],
                provider=route.provider,
                model=route.model,
                tokens_used=result.get("tokens_used", 0),
                latency_ms=int((time.time() - start_time) * 1000),
                thought_content=result.get("thought_content"),
                raw_parts=result.get("raw_parts"),
            )

            # Log thought trace to history if present
            self._log_thought_if_present(gen_result, prompt)
            return gen_result

        except Exception as e:
            errors.append(f"{route.provider.value}: {e}")
            logger.warning(f"Primary provider failed: {e}")
            print(f"DEBUG: Primary provider failed: {e}")

        # Try fallbacks
        candidates = self.TIER_ROUTES.get(tier, self.TIER_ROUTES[TaskTier.FAST])

        for prov, model in candidates:
            if prov == route.provider:
                continue  # Already tried

            if not self._provider_health.get(prov, False):
                continue

            try:
                result = await self._generate_with_provider(
                    prov, model, prompt, system_prompt, max_tokens, temperature
                )

                gen_result = GenerationResult(
                    content=result["content"],
                    provider=prov,
                    model=model,
                    tokens_used=result.get("tokens_used", 0),
                    latency_ms=int((time.time() - start_time) * 1000),
                    fallback_used=True,
                    thought_content=result.get("thought_content"),
                    raw_parts=result.get("raw_parts"),
                )

                # Log thought trace to history if present
                self._log_thought_if_present(gen_result, prompt)
                return gen_result

            except Exception as e:
                errors.append(f"{prov.value}: {e}")
                continue

        # All failed
        return GenerationResult(
            content="",
            provider=route.provider,
            model=route.model,
            latency_ms=int((time.time() - start_time) * 1000),
            error=f"All providers failed: {'; '.join(errors)}",
        )

    def _log_thought_if_present(
        self,
        result: GenerationResult,
        prompt: str,
    ) -> None:
        """Log thought traces to history if present and enabled.

        Args:
            result: The generation result.
            prompt: The original prompt (for preview).
        """
        if not self.log_thoughts:
            return

        if not result.thought_content:
            return

        try:
            history_logger = _get_history_logger()
            if history_logger:
                history_logger.log_thought_trace(
                    thought_content=result.thought_content,
                    provider=result.provider.value,
                    model=result.model,
                    prompt_preview=prompt,
                    response_preview=result.content,
                )
                logger.debug(
                    f"Logged thought trace from {result.provider.value}:{result.model} "
                    f"({len(result.thought_content)} chars)"
                )
        except Exception as e:
            logger.warning(f"Failed to log thought trace: {e}")

    async def _generate_with_provider(
        self,
        provider: Provider,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        node_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Generate content with a specific provider.

        Args:
            provider: The provider to use.
            model: The model to use.
            prompt: The prompt.
            system_prompt: Optional system prompt.
            max_tokens: Max tokens.
            temperature: Temperature.
            node_name: Optional Ollama node name.

        Returns:
            Dict with 'content', optionally 'thought_content', 'tokens_used', 'raw_parts'.
        """
        if provider == Provider.GEMINI:
            return await self._generate_gemini(prompt, model, system_prompt)

        elif provider == Provider.ANTHROPIC:
            backend = AnthropicBackend(
                model=model,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )
            await backend.start()
            try:
                result = await backend.generate_one_shot(prompt, system_prompt, max_tokens)
                return {"content": result, "thought_content": None, "tokens_used": 0, "raw_parts": None}
            finally:
                await backend.stop()

        elif provider == Provider.OPENAI:
            backend = OpenAIBackend(
                model=model,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                temperature=temperature,
            )
            await backend.start()
            try:
                result = await backend.generate_one_shot(prompt, system_prompt, max_tokens, temperature)
                return {"content": result, "thought_content": None, "tokens_used": 0, "raw_parts": None}
            finally:
                await backend.stop()

        elif provider == Provider.LLAMACPP:
            backend = self._build_llamacpp_backend(
                model,
                max_tokens,
                temperature,
                system_prompt,
            )
            await backend.start()
            try:
                result = await backend.generate_one_shot(prompt, system_prompt, max_tokens, temperature)
                return {"content": result, "thought_content": None, "tokens_used": 0, "raw_parts": None}
            finally:
                await backend.stop()

        elif provider == Provider.OLLAMA:
            # Get node if specified
            if node_name and self._node_manager:
                node = self._node_manager.get_node(node_name)
                if node:
                    backend = self._node_manager.create_backend(node, model)
                else:
                    backend = OllamaBackend(model=model)
            else:
                backend = OllamaBackend(model=model)

            await backend.start()
            try:
                result = await backend.generate_one_shot(prompt, system_prompt)
                return {"content": result, "thought_content": None, "tokens_used": 0, "raw_parts": None}
            finally:
                await backend.stop()

        elif provider == Provider.HALEXT:
            result = await self._generate_halext(prompt, model)
            return {"content": result, "thought_content": None, "tokens_used": 0, "raw_parts": None}

        raise ValueError(f"Unknown provider: {provider}")

    async def _generate_gemini(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Generate with Gemini.

        Returns:
            Dict with 'content', 'thought_content', 'tokens_used', and 'raw_parts'.
        """
        if not self._gemini_client:
            raise RuntimeError("Gemini client not initialized")

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        print(f"Calling Gemini API with model {model}...")
        response = await self._gemini_client.aio.models.generate_content(
            model=model,
            contents=full_prompt,
        )
        print("Gemini API call successful.")

        # Log usage
        tokens_used = 0
        if response.usage_metadata:
            tokens_used = response.usage_metadata.total_token_count
            quota_manager.log_usage(model, tokens_used)

        # Extract all parts from response, including thought signatures
        content_parts = []
        thought_parts = []
        raw_parts = []

        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        part_type = type(part).__name__
                        raw_parts.append({"type": part_type, "data": str(part)})

                        # Check for text content
                        if hasattr(part, 'text') and part.text:
                            content_parts.append(part.text)

                        # Check for thought/reasoning traces (Gemini 3)
                        if hasattr(part, 'thought') and part.thought:
                            thought_parts.append(part.thought)
                        elif hasattr(part, 'thought_signature') and part.thought_signature:
                            thought_parts.append(str(part.thought_signature))

                        # Also check for thought in the part data
                        if hasattr(part, '_pb'):
                            pb = part._pb
                            if hasattr(pb, 'thought') and pb.thought:
                                thought_parts.append(pb.thought)

        return {
            "content": "".join(content_parts) if content_parts else response.text or "",
            "thought_content": "\n".join(thought_parts) if thought_parts else None,
            "tokens_used": tokens_used,
            "raw_parts": raw_parts,
        }

    async def _generate_halext(
        self,
        prompt: str,
        model: str,
    ) -> str:
        """Generate via halext-org AI gateway."""
        if not self._halext_client:
            from hafs.integrations.halext_client import HalextOrgClient
            self._halext_client = HalextOrgClient()
            await self._halext_client.login()

        return await self._halext_client.ai_chat(
            messages=[{"role": "user", "content": prompt}],
            model=model if model != "default" else None,
        )

    async def stream_generate(
        self,
        prompt: str,
        tier: TaskTier = TaskTier.FAST,
        provider: Optional[Provider] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        """Stream generated content.

        Args:
            prompt: The prompt.
            tier: Task tier.
            provider: Force specific provider.
            system_prompt: Optional system prompt.
            max_tokens: Max tokens.

        Yields:
            Content chunks.
        """
        if not self._initialized:
            await self.initialize()

        override_provider = provider or self._override_provider
        override_model = model or (self._override_model if override_provider else None)

        if override_provider and not self._provider_health.get(override_provider, False):
            override_provider = None
            override_model = None

        route = await self.route(
            prompt,
            tier=tier,
            provider=override_provider,
            model=override_model,
            require_streaming=True,
        )

        try:
            async for chunk in self._stream_with_provider(
                route.provider,
                route.model,
                prompt,
                system_prompt,
                max_tokens,
                route.node_name,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"[Error: {e}]"

    async def _stream_with_provider(
        self,
        provider: Provider,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        node_name: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream with a specific provider."""
        if provider == Provider.ANTHROPIC:
            backend = AnthropicBackend(
                model=model,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )
            await backend.start()
            try:
                await backend.send_message(prompt)
                async for chunk in backend.stream_response():
                    yield chunk
            finally:
                await backend.stop()

        elif provider == Provider.OPENAI:
            backend = OpenAIBackend(
                model=model,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )
            await backend.start()
            try:
                await backend.send_message(prompt)
                async for chunk in backend.stream_response():
                    yield chunk
            finally:
                await backend.stop()

        elif provider == Provider.LLAMACPP:
            cfg = self._llamacpp_config
            temperature = getattr(cfg, "temperature", 0.7) if cfg else 0.7
            backend = self._build_llamacpp_backend(
                model,
                max_tokens,
                temperature,
                system_prompt,
            )
            await backend.start()
            try:
                await backend.send_message(prompt)
                async for chunk in backend.stream_response():
                    yield chunk
            finally:
                await backend.stop()

        elif provider == Provider.OLLAMA:
            if node_name and self._node_manager:
                node = self._node_manager.get_node(node_name)
                if node:
                    backend = self._node_manager.create_backend(node, model)
                else:
                    backend = OllamaBackend(model=model)
            else:
                backend = OllamaBackend(model=model)

            await backend.start()
            try:
                await backend.send_message(prompt)
                async for chunk in backend.stream_response():
                    yield chunk
            finally:
                await backend.stop()

        elif provider == Provider.GEMINI:
            # Gemini streaming would need different handling
            result = await self._generate_gemini(prompt, model, system_prompt)
            yield result

        else:
            raise ValueError(f"Streaming not supported for: {provider}")

    async def embed(
        self,
        text: str,
        provider: Optional[Provider | str] = None,
        model: Optional[str] = None,
    ) -> list[float]:
        """Generate embeddings.

        Args:
            text: Text to embed.
            provider: Force specific provider.
            model: Provider-specific embedding model override.

        Returns:
            Embedding vector.
        """
        if not self._initialized:
            await self.initialize()

        # Prefer Gemini for embeddings (free, high quality)
        if isinstance(provider, str):
            try:
                provider = Provider(provider)
            except ValueError:
                provider = None

        if provider is None:
            if Provider.GEMINI in self._provider_health:
                provider = Provider.GEMINI
            elif Provider.OPENAI in self._provider_health:
                provider = Provider.OPENAI

        if provider == Provider.GEMINI and self._gemini_client:
            embed_model = model or "text-embedding-004"
            response = await self._gemini_client.aio.models.embed_content(
                model=embed_model,
                contents=text,
            )
            return response.embeddings[0].values

        elif provider == Provider.OPENAI:
            backend = OpenAIBackend()
            await backend.start()
            try:
                embeddings = await backend.generate_embeddings(
                    [text],
                    model=model or "text-embedding-3-small",
                )
                return embeddings[0] if embeddings else []
            finally:
                await backend.stop()

        return []

    def get_provider_status(self) -> dict[str, Any]:
        """Get status of all providers.

        Returns:
            Provider status information.
        """
        return {
            provider.value: {
                "available": self._provider_health.get(provider, False),
                "enabled": self.configs[provider].enabled,
                "default_model": self.configs[provider].default_model,
                "cost_per_1k": self.configs[provider].cost_per_1k_tokens,
            }
            for provider in Provider
        }

    def set_provider_enabled(self, provider: Provider, enabled: bool):
        """Enable or disable a provider.

        Args:
            provider: Provider to configure.
            enabled: Whether to enable.
        """
        if provider in self.configs:
            self.configs[provider].enabled = enabled

    async def close(self):
        """Cleanup resources."""
        for backend in self._backends.values():
            try:
                await backend.stop()
            except:
                pass

        if self._node_manager:
            await self._node_manager.close()

        if self._halext_client:
            await self._halext_client.close()


# Global orchestrator instance
orchestrator_v2: Optional[UnifiedOrchestrator] = None


async def get_orchestrator() -> UnifiedOrchestrator:
    """Get or create the global orchestrator instance."""
    global orchestrator_v2

    if orchestrator_v2 is None:
        orchestrator_v2 = UnifiedOrchestrator()
        await orchestrator_v2.initialize()

    return orchestrator_v2
