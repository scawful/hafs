"""Legacy path for AgentLane. Use agents.core.lane instead."""
import warnings
from agents.core.lane import AgentLane, AgentLaneManager

warnings.warn(
    "agents.core.lane is deprecated. Import from 'agents.core.lane' instead.",
    DeprecationWarning,
    stacklevel=2
)
