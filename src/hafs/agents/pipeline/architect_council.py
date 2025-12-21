import warnings
from agents.pipeline.architect_council import ArchitectCouncil

warnings.warn(
    "hafs.agents.pipeline.architect_council is deprecated. Import from 'agents.pipeline.architect_council' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export
ArchitectCouncil = ArchitectCouncil
