import warnings
from agents.knowledge.alttp_embeddings import ALTTPEmbeddingSpecialist

warnings.warn(
    "hafs.agents.alttp_embeddings is deprecated. Import from 'agents.knowledge.alttp_embeddings' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export classes
ALTTPEmbeddingSpecialist = ALTTPEmbeddingSpecialist
