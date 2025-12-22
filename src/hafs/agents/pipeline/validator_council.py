import warnings
from agents.pipeline.validator_council import ValidatorCouncil

warnings.warn(
    "agents.pipeline.validator_council is deprecated. Import from 'agents.pipeline.validator_council' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export
ValidatorCouncil = ValidatorCouncil