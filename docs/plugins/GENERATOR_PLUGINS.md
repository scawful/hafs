# Generator Plugins

Training data generators are loaded from plugins, keeping domain-specific code separate from the core framework.

## Plugin Structure

```
my-domain-plugin/
├── config/
│   └── training_paths.toml    # Configurable source paths
├── generators/
│   ├── __init__.py            # Plugin registration
│   ├── my_generator.py        # Generator implementation
│   └── my_other_generator.py
└── config.toml                # Plugin metadata
```

## Registration

The plugin must export a `register_generators()` function:

```python
# generators/__init__.py

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Plugin paths
PLUGIN_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PLUGIN_ROOT / "config"

def get_training_paths() -> dict:
    """Load paths from plugin config."""
    import tomllib

    config_file = CONFIG_DIR / "training_paths.toml"
    if not config_file.exists():
        return {}

    with open(config_file, "rb") as f:
        return tomllib.load(f)

def register_generators(curator):
    """Register generators with the DataCurator."""
    from .my_generator import MyGenerator

    async def _register():
        paths = get_training_paths()

        # Initialize and register generator
        gen = MyGenerator(source_path=paths.get("my_source"))
        await gen.setup()
        curator.register_generator("my_domain", gen)
        logger.info("Registered: my_domain generator")

    asyncio.run(_register())
```

## Generator Implementation

Generators must extend `DataGenerator`:

```python
# generators/my_generator.py

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agents.training.base import DataGenerator, SourceItem, TrainingSample

@dataclass
class MySourceItem(SourceItem):
    """Source item for my domain."""

    file_path: str = ""
    content: str = ""
    metadata: dict = None

    @property
    def item_id(self) -> str:
        return f"my_domain:{self.file_path}"


class MyGenerator(DataGenerator):
    """Generate training data from my domain sources."""

    def __init__(self, source_path: Optional[Path] = None):
        super().__init__(
            name="MyGenerator",
            domain="my_domain",
            teacher_tier="coding",  # TaskTier for LLM calls
        )
        self.source_path = source_path
        self._orchestrator = None

    async def setup(self):
        """Initialize resources."""
        await super().setup()

        from core.orchestrator_v2 import UnifiedOrchestrator
        self._orchestrator = UnifiedOrchestrator()
        await self._orchestrator.initialize()

    async def extract_source_items(self) -> list[MySourceItem]:
        """Extract source items from the domain."""
        if not self.source_path or not self.source_path.exists():
            return []

        items = []
        for file in self.source_path.rglob("*.py"):
            content = file.read_text()
            items.append(MySourceItem(
                name=file.stem,
                content=content,
                source="my_domain",
                file_path=str(file),
            ))

        return items

    def get_teacher_prompt(self, item: SourceItem) -> str:
        """Generate prompt for teacher LLM."""
        return f"""Analyze this code and generate a training sample.

CODE:
```
{item.content}
```

Generate a JSON object:
{{
  "instruction": "A question about this code",
  "input": "Context or additional info",
  "output": "Detailed answer/explanation"
}}
"""

    async def generate_sample(self, item: SourceItem) -> Optional[TrainingSample]:
        """Generate a training sample from a source item."""
        from agents.training.json_utils import extract_json_from_response
        from core.orchestrator_v2 import TaskTier

        prompt = self.get_teacher_prompt(item)

        try:
            result = await self._orchestrator.generate(
                prompt=prompt,
                tier=TaskTier.CODING,
            )

            data = extract_json_from_response(result.content)
            if not data:
                return None

            return TrainingSample(
                instruction=data.get("instruction", ""),
                input=data.get("input", ""),
                output=data.get("output", ""),
                domain="my_domain",
                source=item.source,
                teacher_model=result.model,
                teacher_prompt=prompt,
            )
        except Exception as e:
            logger.error(f"Failed to generate sample: {e}")
            return None
```

## Path Configuration

Configure source paths in your plugin's config:

```toml
# config/training_paths.toml

[paths]
my_source = "~/Code/my-project/src"
other_source = "~/Data/other-data"

[knowledge_bases]
my_kb = "~/.context/knowledge/my-domain"
```

## Discovery

The framework discovers plugins in these locations:

1. `~/Code/{plugin_name}/generators/` - Development plugins
2. `~/.config/hafs/plugins/{plugin_name}/generators/` - Installed plugins

Each must have `generators/__init__.py` with `register_generators()`.

## Loading Generators

```python
from agents.training.curator import DataCurator
from agents.training.generators import load_plugin_generators

async def main():
    curator = DataCurator()
    await curator.setup()

    # Auto-discover and load plugin generators
    count = load_plugin_generators(curator)
    print(f"Loaded {count} generators from plugins")

    # List available domains
    domains = curator.list_domains()
    print(f"Available domains: {domains}")

    # Generate dataset
    result = await curator.curate_dataset(
        domains=domains,
        target_count=1000,
    )
```

## Quality Integration

Generators integrate with the quality pipeline:

```python
# In quality.py, add domain-specific thresholds
DOMAIN_THRESHOLDS = {
    "my_domain": 0.5,  # Adjust based on domain complexity
    # ...
}

# Code domains get more lenient instruction/output ratio checks
CODE_DOMAINS = ["my_domain", "asm", "cpp", ...]
```

## Best Practices

1. **Keep paths configurable** - Use `training_paths.toml`, not hardcoded paths
2. **Handle missing sources gracefully** - Return empty lists if paths don't exist
3. **Use appropriate task tiers** - Match teacher LLM to task complexity
4. **Implement good prompts** - Clear, structured prompts improve sample quality
5. **Log progress** - Use logging for debugging generation issues
6. **Test locally** - Run small pilots before full generation
