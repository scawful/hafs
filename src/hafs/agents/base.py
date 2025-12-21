"""Legacy path for BaseAgent. Use agents.core.base instead."""
import warnings
from agents.core.base import BaseAgent, AgentMetrics

warnings.warn(
    "hafs.agents.base is deprecated. Import from 'agents.core.base' instead.",
    DeprecationWarning,
    stacklevel=2
)