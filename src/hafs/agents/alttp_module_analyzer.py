import warnings
from agents.knowledge.alttp_analyzer import ALTTPModuleAnalyzer

warnings.warn(
    "agents.knowledge.alttp_analyzer is deprecated. Import from 'agents.knowledge.alttp_analyzer' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export classes
ALTTPModuleAnalyzer = ALTTPModuleAnalyzer
