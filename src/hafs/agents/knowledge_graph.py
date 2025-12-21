import warnings
from agents.knowledge.graph import KnowledgeGraphAgent

warnings.warn(
    "hafs.agents.knowledge_graph is deprecated. Import from 'agents.knowledge.graph' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export classes
KnowledgeGraphAgent = KnowledgeGraphAgent
