import warnings
from agents.pipeline.test_writer import TestWriter

warnings.warn(
    "hafs.agents.pipeline.test_writer is deprecated. Import from 'agents.pipeline.test_writer' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export
TestWriter = TestWriter
