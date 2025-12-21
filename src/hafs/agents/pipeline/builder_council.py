import warnings
from agents.pipeline.builder_council import BuilderCouncil

warnings.warn(
    "hafs.agents.pipeline.builder_council is deprecated. Import from 'agents.pipeline.builder_council' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export
BuilderCouncil = BuilderCouncil
