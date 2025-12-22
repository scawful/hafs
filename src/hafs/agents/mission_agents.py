import warnings
from agents.mission.mission_agents import *

warnings.warn(
    "agents is deprecated. Import from 'agents' instead.",
    DeprecationWarning,
    stacklevel=2,
)

warnings.warn(
    "agents.mission.mission_agents is deprecated, use agents.mission.mission_agents instead",
    DeprecationWarning,
    stacklevel=2
)
