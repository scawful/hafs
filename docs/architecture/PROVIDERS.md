# Provider System

hAFS uses a pluggable provider system that abstracts LLM interactions, allowing seamless switching between providers and easy integration of custom backends.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    UnifiedOrchestrator                       │
├─────────────────────────────────────────────────────────────┤
│  generate() / embed() / stream()                            │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Provider Selection                       │   │
│  │  • Task tier routing (REASONING, FAST, CODING, etc.) │   │
│  │  • Provider preference                                │   │
│  │  • Fallback chain                                     │   │
│  └──────────────────────────────────────────────────────┘   │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐   │
│  │ Gemini   │Anthropic │ OpenAI   │ Ollama   │ Custom   │   │
│  │ Adapter  │ Adapter  │ Adapter  │ Adapter  │ Adapter  │   │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Provider Enum

```python
from enum import Enum

class Provider(str, Enum):
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    LLAMACPP = "llamacpp"
    # Add custom providers here
```

## Task Tiers

Route requests based on task complexity:

```python
class TaskTier(str, Enum):
    REASONING = "reasoning"   # Complex analysis, planning
    FAST = "fast"             # Quick responses, simple tasks
    CODING = "coding"         # Code generation, review
    RESEARCH = "research"     # Information gathering
    CREATIVE = "creative"     # Writing, ideation
```

## Basic Usage

```python
from core.orchestrator_v2 import UnifiedOrchestrator, Provider, TaskTier

async def main():
    orch = UnifiedOrchestrator()
    await orch.initialize()

    # Auto-select provider based on tier
    result = await orch.generate(
        prompt="Write a function to sort a list",
        tier=TaskTier.CODING,
    )

    # Explicitly specify provider
    result = await orch.generate(
        prompt="Analyze this architecture",
        tier=TaskTier.REASONING,
        provider=Provider.GEMINI,
        model="gemini-3-pro-preview",
    )

    print(result.content)
```

## Provider Configuration

### Environment Variables

```bash
# API Keys
export GEMINI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."

# Override default provider
export HAFS_MODEL_PROVIDER="gemini"
export HAFS_MODEL_MODEL="gemini-3-flash-preview"

# Rotation (fallback chain)
export HAFS_MODEL_ROTATION="gemini:gemini-3-flash-preview,anthropic:claude-sonnet-4"
```

### Config File

`~/.config/hafs/config.toml`:

```toml
[providers]
default = "gemini"

[providers.gemini]
api_key_env = "GEMINI_API_KEY"
default_model = "gemini-3-flash-preview"

[providers.anthropic]
api_key_env = "ANTHROPIC_API_KEY"
default_model = "claude-sonnet-4-20250514"

[providers.ollama]
base_url = "http://localhost:11434"
default_model = "qwen2.5-coder:14b"
```

## Custom Provider Adapters

Create adapters for internal or custom LLM backends:

### 1. Define the Provider

```python
# In your plugin's __init__.py or a dedicated providers.py

from core.orchestrator_v2 import Provider

# Extend the Provider enum (requires modifying core or using string literals)
CUSTOM_PROVIDER = "custom"
```

### 2. Create the Adapter

```python
# my_plugin/adapters/custom_adapter.py

from typing import Optional
from dataclasses import dataclass

@dataclass
class CustomResponse:
    content: str
    tokens_used: int = 0

class CustomAdapter:
    """Adapter for custom LLM backend."""

    def __init__(self, api_key: Optional[str] = None, base_url: str = ""):
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    async def initialize(self):
        """Initialize the client."""
        # Set up your custom client here
        pass

    async def generate(
        self,
        prompt: str,
        model: str = "default-model",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> CustomResponse:
        """Generate a response."""
        # Implement your API call here
        response = await self._call_api(prompt, model, max_tokens, temperature)
        return CustomResponse(content=response)

    async def embed(self, text: str, model: str = "embedding-model") -> list[float]:
        """Generate embeddings."""
        # Implement embedding API call
        return await self._call_embedding_api(text, model)
```

### 3. Register with Orchestrator

```python
# my_plugin/__init__.py

from core.orchestrator_v2 import UnifiedOrchestrator
from .adapters.custom_adapter import CustomAdapter

def register_provider(orch: UnifiedOrchestrator):
    """Register custom provider with the orchestrator."""
    adapter = CustomAdapter(
        api_key=os.getenv("CUSTOM_API_KEY"),
        base_url="https://api.custom.example.com",
    )
    orch.register_adapter("custom", adapter)
```

## Tier Routing Configuration

Customize which providers handle which task tiers:

```python
# Default routing (in core/orchestrator_v2.py)
TIER_ROUTES = {
    TaskTier.REASONING: [
        (Provider.GEMINI, "gemini-3-pro-preview"),
        (Provider.ANTHROPIC, "claude-opus-4-5-20251101"),
    ],
    TaskTier.FAST: [
        (Provider.GEMINI, "gemini-3-flash-preview"),
        (Provider.OPENAI, "gpt-5.2-mini"),
    ],
    TaskTier.CODING: [
        (Provider.GEMINI, "gemini-3-flash-preview"),
        (Provider.OLLAMA, "qwen2.5-coder:14b"),
    ],
}
```

Override in your plugin:

```python
def configure_tier_routes(orch: UnifiedOrchestrator):
    """Configure custom tier routing."""
    orch.set_tier_route(
        TaskTier.CODING,
        [
            ("custom", "custom-code-model"),
            (Provider.GEMINI, "gemini-3-flash-preview"),
        ]
    )
```

## Fallback Behavior

The orchestrator automatically handles provider failures:

1. Try primary provider/model
2. On failure, try next in tier route
3. Log errors and continue until success or all exhausted

```python
# Explicit fallback chain
result = await orch.generate(
    prompt="...",
    tier=TaskTier.CODING,
    fallback_providers=[
        (Provider.GEMINI, "gemini-3-flash-preview"),
        (Provider.OLLAMA, "qwen2.5-coder:14b"),
        ("custom", "custom-model"),
    ],
)
```

## Embeddings

```python
# Generate embeddings
embeddings = await orch.embed(
    text="Sample text for embedding",
    provider=Provider.GEMINI,
    model="text-embedding-004",
)

# Returns list[float] - vector representation
```

## Best Practices

1. **Use task tiers** - Let the orchestrator choose the right provider
2. **Set up fallbacks** - Configure multiple providers for reliability
3. **Use environment variables** - Keep API keys out of code
4. **Create adapters** - Wrap internal/custom APIs with the adapter pattern
5. **Test locally first** - Use Ollama for development before using paid APIs
