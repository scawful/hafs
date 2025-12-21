"""Legacy path for AgentCoordinator. Use agents.core.coordinator instead."""
import warnings
from agents.core.coordinator import AgentCoordinator, CoordinatorMode

warnings.warn(
    "hafs.agents.coordinator is deprecated. Import from 'agents.core.coordinator' instead.",
    DeprecationWarning,
    stacklevel=2
)
