import warnings

from agents.knowledge.alttp_unified import (
    OracleOfSecretsKB,
    UnifiedALTTPKnowledge,
    UnifiedSearchResult,
)
from hafs.core.embeddings import BatchEmbeddingManager

warnings.warn(
    "hafs.agents.alttp_unified_kb is deprecated. Import from 'agents.knowledge.alttp_unified' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export classes
UnifiedALTTPKnowledge = UnifiedALTTPKnowledge
OracleOfSecretsKB = OracleOfSecretsKB
UnifiedSearchResult = UnifiedSearchResult
BatchEmbeddingManager = BatchEmbeddingManager

__all__ = [
    "UnifiedALTTPKnowledge",
    "OracleOfSecretsKB",
    "UnifiedSearchResult",
    "BatchEmbeddingManager",
]
