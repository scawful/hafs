import warnings
from agents.knowledge.enhancer import KBEnhancer

warnings.warn(
    "hafs.agents.kb_enhancer is deprecated. Import from 'agents.knowledge.enhancer' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export classes
KBEnhancer = KBEnhancer
