"""Legacy path for MentionRouter. Use agents.core.router instead."""
import warnings
from agents.core.router import MentionRouter

warnings.warn(
    "hafs.agents.router is deprecated. Import from 'agents.core.router' instead.",
    DeprecationWarning,
    stacklevel=2
)
