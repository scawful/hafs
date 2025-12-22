# hAFS Documentation

**Hybrid Agent File System** - A modular framework for building AI agents with knowledge bases, embeddings, and training pipelines.

## Quick Start

```bash
pip install hafs
hafs --help
```

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Agents** | Autonomous units that process tasks using LLMs |
| **Knowledge Bases** | Structured repositories of domain knowledge |
| **Embeddings** | Vector representations for semantic search |
| **Training** | Fine-tuning pipelines for custom models |
| **Providers** | Pluggable LLM backends (Gemini, Anthropic, OpenAI, local) |

## Documentation

### Getting Started
- [Usage Guide](guides/USAGE.md) - Basic commands and workflows
- [Configuration](guides/CONFIGURATION.md) - System configuration
- [Agents Quickstart](guides/AGENTS_QUICKSTART.md) - Creating agents

### Plugin Development
- [Plugin Development](plugins/PLUGIN_DEVELOPMENT.md) - Creating plugins
- [Plugin Adapters](plugins/PLUGIN_ADAPTER_GUIDE.md) - Adapter pattern
- [Generator Plugins](plugins/GENERATOR_PLUGINS.md) - Training data generators

### Knowledge & Embeddings
- [Knowledge Bases](guides/KNOWLEDGE_BASES.md) - Creating and using KBs
- [Embedding Service](guides/EMBEDDINGS.md) - Vector embeddings

### Training
- [Training Overview](training/TRAINING_PIPELINE_OVERVIEW.md) - Pipeline architecture
- [LoRA Training](training/LORA_TRAINING.md) - Fine-tuning with LoRA
- [Quality Pipeline](training/QUALITY_IMPROVEMENT_GUIDE.md) - Sample quality

### Architecture
- [Architecture](architecture/ARCHITECTURE.md) - System design
- [Provider System](architecture/PROVIDERS.md) - LLM provider abstraction

## Provider Abstraction

hAFS supports multiple LLM providers through a unified interface:

```python
from core.orchestrator_v2 import UnifiedOrchestrator, Provider, TaskTier

orch = UnifiedOrchestrator()
await orch.initialize()

# Use any provider
result = await orch.generate(
    prompt="Explain this code",
    tier=TaskTier.CODING,
    provider=Provider.GEMINI,  # or ANTHROPIC, OPENAI, OLLAMA, etc.
)
```

### Available Providers

| Provider | Models | Use Case |
|----------|--------|----------|
| `GEMINI` | gemini-3-flash, gemini-3-pro | Fast, capable |
| `ANTHROPIC` | claude-opus-4.5, claude-sonnet-4 | Deep reasoning |
| `OPENAI` | gpt-5.2, gpt-5.2-mini | General purpose |
| `OLLAMA` | Local models | Offline/private |
| `LLAMACPP` | GGUF models | Local inference |

### Custom Provider Adapters

Create adapters for internal or custom providers:

```python
from core.orchestrator_v2 import Provider, ProviderConfig

# Register custom provider in your plugin
PROVIDER_CONFIGS[Provider.CUSTOM] = ProviderConfig(
    provider=Provider.CUSTOM,
    api_key_env="CUSTOM_API_KEY",
    models=["custom-model-v1"],
)
```

## Plugin System

Extend hAFS with plugins for domain-specific functionality:

```
my-plugin/
├── config/
│   └── training_paths.toml    # Path configuration
├── generators/
│   ├── __init__.py            # register_generators()
│   └── my_generator.py        # Custom data generator
└── config.toml                # Plugin metadata
```

### Generator Plugin Example

```python
# generators/__init__.py
def register_generators(curator):
    """Called by hAFS to register domain-specific generators."""
    from .my_generator import MyGenerator

    gen = MyGenerator()
    await gen.setup()
    curator.register_generator("my_domain", gen)
```

## Knowledge Base Creation

```python
from agents.knowledge import KnowledgeBase

kb = KnowledgeBase(name="my-project")
await kb.setup()

# Add documents
await kb.add_document("path/to/file.py", category="code")

# Query
results = await kb.search("how does X work?", top_k=5)
```

## Training Pipeline

```python
from agents.training.curator import DataCurator
from agents.training.generators import load_plugin_generators

curator = DataCurator()
await curator.setup()

# Load generators from plugins
load_plugin_generators(curator)

# Generate dataset
result = await curator.curate_dataset(
    domains=["my_domain"],
    target_count=1000,
    quality_threshold=0.5,
)
```

## Configuration

Main config: `~/.config/hafs/config.toml`

```toml
[providers]
default = "gemini"

[providers.gemini]
api_key_env = "GEMINI_API_KEY"

[plugins]
plugin_dirs = ["~/my-plugins"]
```

## License

MIT
