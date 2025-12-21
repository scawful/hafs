import warnings
from agents.pipeline.doc_writer import DocWriter

warnings.warn(
    "hafs.agents.pipeline.doc_writer is deprecated. Import from 'agents.pipeline.doc_writer' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export
DocWriter = DocWriter
