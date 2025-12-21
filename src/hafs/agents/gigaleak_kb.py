import warnings
from agents.knowledge.gigaleak import GigaleakKB

warnings.warn(
    "hafs.agents.gigaleak_kb is deprecated. Import from 'agents.knowledge.gigaleak' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export classes
GigaleakKB = GigaleakKB
