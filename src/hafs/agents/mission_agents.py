import warnings
from agents.mission.mission_agents import *

warnings.warn(
    "hafs.agents is deprecated. Import from 'agents' instead.",
    DeprecationWarning,
    stacklevel=2,
)

warnings.warn(
    "hafs.agents.mission_agents is deprecated, use agents.mission.mission_agents instead",
    DeprecationWarning,
    stacklevel=2
)
