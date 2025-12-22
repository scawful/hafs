# Knowledge Bases

Knowledge bases (KBs) store structured domain knowledge for semantic search and training data generation.

## Structure

```
~/.context/knowledge/{name}/
├── symbols.json      # Named entities (functions, variables, labels)
├── routines.json     # Code routines with snippets
├── modules.json      # Module/file metadata
└── embeddings/       # Vector embeddings (optional)
    └── *.jsonl
```

## Creating a Knowledge Base

### From Code

```python
from agents.knowledge import KnowledgeBase

async def build_kb():
    kb = KnowledgeBase(name="my-project")
    await kb.setup()

    # Index a directory
    await kb.index_directory(
        path="~/Code/my-project/src",
        file_patterns=["*.py", "*.ts"],
        category="code",
    )

    # Add individual documents
    await kb.add_document(
        path="docs/architecture.md",
        category="documentation",
    )

    # Save to disk
    await kb.save()
```

### From JSON

```python
import json
from pathlib import Path

kb_path = Path.home() / ".context" / "knowledge" / "my-kb"
kb_path.mkdir(parents=True, exist_ok=True)

# Create symbols.json
symbols = [
    {
        "id": "my-kb:process_data",
        "name": "process_data",
        "address": "",
        "symbol_type": "function",
        "file_path": "src/processor.py",
        "line_number": 42,
        "description": "Process input data and return results",
        "category": "core",
    },
    # ... more symbols
]

with open(kb_path / "symbols.json", "w") as f:
    json.dump(symbols, f, indent=2)
```

## Schema

### symbols.json

```json
[
  {
    "id": "unique-id",
    "name": "symbol_name",
    "address": "optional-address",
    "symbol_type": "function|class|variable|constant|label",
    "file_path": "relative/path/to/file",
    "line_number": 42,
    "description": "What this symbol does",
    "category": "core|utility|api|...",
    "vanilla_reference": null
  }
]
```

### routines.json

```json
[
  {
    "id": "unique-id",
    "name": "routine_name",
    "address": "optional-address",
    "bank": "",
    "file_path": "relative/path/to/file",
    "line_number": 42,
    "description": "What this routine does",
    "category": "core|utility|api|...",
    "code_snippet": "def routine_name():\n    ...",
    "calls": ["other_routine", "helper_func"],
    "called_by": ["main", "caller_routine"],
    "is_hook": false,
    "hooks_vanilla": null,
    "memory_access": []
  }
]
```

## Querying

```python
from agents.knowledge import KnowledgeBase

async def search_kb():
    kb = KnowledgeBase(name="my-project")
    await kb.setup()
    await kb.load()

    # Semantic search
    results = await kb.search(
        query="how to process input data",
        top_k=5,
    )

    for result in results:
        print(f"{result.name}: {result.description}")

    # Filter by category
    results = await kb.search(
        query="error handling",
        top_k=10,
        category="utility",
    )

    # Get specific symbol
    symbol = kb.get_symbol("process_data")
```

## Embeddings

Generate embeddings for semantic search:

```python
from agents.knowledge import KnowledgeBase

async def generate_embeddings():
    kb = KnowledgeBase(name="my-project")
    await kb.setup()
    await kb.load()

    # Generate embeddings for all symbols
    await kb.generate_embeddings(
        model="text-embedding-004",
        batch_size=100,
    )

    # Embeddings saved to ~/.context/knowledge/my-project/embeddings/
```

## Using with Training

KBs provide source material for training generators:

```python
from agents.training.generators import AsmDataGenerator

class MyKBGenerator(DataGenerator):
    def __init__(self, kb_path: str):
        super().__init__(name="MyKBGen", domain="my_domain")
        self.kb_path = kb_path
        self._kb = None

    async def setup(self):
        await super().setup()
        self._kb = KnowledgeBase(name=self.kb_path)
        await self._kb.setup()
        await self._kb.load()

    async def extract_source_items(self):
        """Extract routines from KB as source items."""
        items = []
        for routine in self._kb.routines:
            items.append(SourceItem(
                name=routine["name"],
                content=routine["code_snippet"],
                source=self.kb_path,
            ))
        return items
```

## CLI Commands

```bash
# List knowledge bases
hafs kb list

# Index a directory
hafs kb index my-project ~/Code/my-project/src

# Generate embeddings
hafs kb embed my-project

# Search
hafs kb search my-project "how to handle errors"
```

## Best Practices

1. **Use descriptive categories** - Makes filtering easier
2. **Include code snippets** - Essential for code KBs
3. **Generate embeddings** - Enables semantic search
4. **Keep KBs focused** - One KB per project/domain
5. **Update regularly** - Re-index when source changes
