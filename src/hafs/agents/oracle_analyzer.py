import warnings
from agents.knowledge.oracle_analyzer import OracleOfSecretsAnalyzer

warnings.warn(
    "hafs.agents.oracle_analyzer is deprecated. Import from 'agents.knowledge.oracle_analyzer' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export classes
OracleOfSecretsAnalyzer = OracleOfSecretsAnalyzer
