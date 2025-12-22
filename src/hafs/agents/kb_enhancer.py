import warnings
from agents.knowledge.enhancer import KBEnhancer

warnings.warn(
    "agents.knowledge.enhancer is deprecated. Import from 'agents.knowledge.enhancer' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export classes
KBEnhancer = KBEnhancer
