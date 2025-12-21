import warnings
from agents.knowledge.alttp_unified import UnifiedALTTPKnowledge

warnings.warn(
    "hafs.agents.alttp_unified_kb is deprecated. Import from 'agents.knowledge.alttp_unified' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export classes
UnifiedALTTPKnowledge = UnifiedALTTPKnowledge
