import warnings
from agents.pipeline.code_writer import CodeWriter

warnings.warn(
    "agents.pipeline.code_writer is deprecated. Import from 'agents.pipeline.code_writer' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export
CodeWriter = CodeWriter
