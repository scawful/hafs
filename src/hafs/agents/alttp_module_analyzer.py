import warnings
from agents.knowledge.alttp_analyzer import ALTTPModuleAnalyzer

warnings.warn(
    "hafs.agents.alttp_module_analyzer is deprecated. Import from 'agents.knowledge.alttp_analyzer' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export classes
ALTTPModuleAnalyzer = ALTTPModuleAnalyzer
