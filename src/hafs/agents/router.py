"""Legacy path for MentionRouter. Use agents.core.router instead."""
import warnings
from agents.core.router import MentionRouter

warnings.warn(
    "agents.core.router is deprecated. Import from 'agents.core.router' instead.",
    DeprecationWarning,
    stacklevel=2
)
