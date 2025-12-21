import warnings
from agents.knowledge.oracle import OracleKnowledgeBase, OracleKBBuilder

warnings.warn(
    "hafs.agents.oracle_kb_builder is deprecated. Import from 'agents.knowledge.oracle' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export classes
OracleKnowledgeBase = OracleKnowledgeBase
OracleKBBuilder = OracleKBBuilder
