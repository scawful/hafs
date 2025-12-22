import warnings

from agents.knowledge.alttp_unified import (
    OracleOfSecretsKB,
    UnifiedALTTPKnowledge,
    UnifiedSearchResult,
)
from core.embeddings import BatchEmbeddingManager

warnings.warn(
    "agents.knowledge.alttp_unified is deprecated. Import from 'agents.knowledge.alttp_unified' instead.",
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
