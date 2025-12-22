import warnings
from agents.knowledge.alttp_multi import ALTTPMultiKBManager

warnings.warn(
    "agents.knowledge.alttp_multi is deprecated. Import from 'agents.knowledge.alttp_multi' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export classes
ALTTPMultiKBManager = ALTTPMultiKBManager
