# Embeddings

hAFS uses vector embeddings for semantic search across knowledge bases and code.

## Overview

Embeddings convert text into dense vectors that capture semantic meaning, enabling:
- Semantic search (find similar content)
- Clustering related documents
- Training data deduplication

## Generating Embeddings

### Via Orchestrator

```python
from core.orchestrator_v2 import UnifiedOrchestrator, Provider

async def embed_text():
    orch = UnifiedOrchestrator()
    await orch.initialize()

    # Generate embedding
    vector = await orch.embed(
        text="How to implement a binary search tree",
        provider=Provider.GEMINI,
        model="text-embedding-004",
    )

    print(f"Embedding dimension: {len(vector)}")  # 768 for text-embedding-004
```

### Via Embedding Service

```python
from context.embedding_service import EmbeddingService

async def batch_embed():
    service = EmbeddingService()
    await service.setup()

    texts = [
        "First document content",
        "Second document content",
        "Third document content",
    ]

    # Batch embedding
    vectors = await service.embed_batch(texts, batch_size=100)

    for i, vec in enumerate(vectors):
        print(f"Doc {i}: {len(vec)} dimensions")
```

## Storage

Embeddings are stored in JSONL format:

```
~/.context/knowledge/{kb_name}/embeddings/
├── symbols.jsonl
├── routines.jsonl
└── documents.jsonl
```

### JSONL Format

```json
{"id": "symbol:process_data", "vector": [0.123, -0.456, ...], "text": "process_data function"}
{"id": "symbol:handle_error", "vector": [0.789, 0.012, ...], "text": "handle_error function"}
```

## Semantic Search

```python
from context.embedding_service import EmbeddingService
import numpy as np

async def semantic_search():
    service = EmbeddingService()
    await service.setup()

    # Load corpus embeddings
    corpus = service.load_embeddings("~/.context/knowledge/my-kb/embeddings/")

    # Embed query
    query_vec = await service.embed("how to handle errors gracefully")

    # Compute similarities
    similarities = np.dot(corpus.vectors, query_vec)

    # Get top results
    top_indices = np.argsort(similarities)[-10:][::-1]

    for idx in top_indices:
        print(f"{corpus.ids[idx]}: {similarities[idx]:.3f}")
```

## Embedding Models

| Provider | Model | Dimensions | Use Case |
|----------|-------|------------|----------|
| Gemini | text-embedding-004 | 768 | General purpose |
| OpenAI | text-embedding-3-small | 1536 | Cost-effective |
| OpenAI | text-embedding-3-large | 3072 | High accuracy |
| Local | sentence-transformers | varies | Offline |

## Configuration

```toml
# ~/.config/hafs/config.toml

[embeddings]
default_provider = "gemini"
default_model = "text-embedding-004"
batch_size = 100
cache_dir = "~/.context/embedding_cache"

[embeddings.providers.gemini]
model = "text-embedding-004"

[embeddings.providers.openai]
model = "text-embedding-3-small"
```

## Deduplication

Use embeddings to detect near-duplicate training samples:

```python
from agents.training.dedup import EmbeddingDeduplicator

async def deduplicate_samples():
    dedup = EmbeddingDeduplicator(threshold=0.95)
    await dedup.setup()

    samples = [...]  # Your training samples

    unique_samples = await dedup.deduplicate(samples)

    print(f"Removed {len(samples) - len(unique_samples)} duplicates")
```

## Best Practices

1. **Batch embedding** - More efficient than one-by-one
2. **Cache embeddings** - Avoid recomputing for unchanged content
3. **Choose appropriate model** - Balance cost vs accuracy
4. **Normalize vectors** - For cosine similarity
5. **Use approximate search** - For large corpora (HNSW, etc.)
