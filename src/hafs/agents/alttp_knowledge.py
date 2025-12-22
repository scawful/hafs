import warnings
from agents.knowledge.alttp import ALTTPKnowledgeBase

warnings.warn(
    "agents.knowledge.alttp is deprecated. Import from 'agents.knowledge.alttp' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export classes
ALTTPKnowledgeBase = ALTTPKnowledgeBase
