"""Multi-agent orchestration system."""

from hafs.agents.coordinator import AgentCoordinator, CoordinatorMode
from hafs.agents.lane import AgentLane, AgentLaneManager
from hafs.agents.roles import (
    ROLE_DESCRIPTIONS,
    ROLE_KEYWORDS,
    get_role_keywords,
    get_role_system_prompt,
    match_role_by_keywords,
)
from hafs.agents.router import MentionRouter
from hafs.models.agent import Agent, AgentMessage, AgentRole, SharedContext

# Specialized agents
from hafs.agents.history_pipeline import HistoryPipelineAgent, with_history_logging
from hafs.agents.rom_specialist import RomHackingSpecialist
from hafs.agents.observability import DistributedObservabilityAgent

__all__ = [
    # Coordinator
    "AgentCoordinator",
    "CoordinatorMode",
    # Lane management
    "AgentLane",
    "AgentLaneManager",
    # Routing
    "MentionRouter",
    # Roles
    "get_role_system_prompt",
    "get_role_keywords",
    "match_role_by_keywords",
    "ROLE_DESCRIPTIONS",
    "ROLE_KEYWORDS",
    # Models (re-exported for convenience)
    "Agent",
    "AgentMessage",
    "AgentRole",
    "SharedContext",
    # Specialized agents
    "HistoryPipelineAgent",
    "with_history_logging",
    "RomHackingSpecialist",
    "DistributedObservabilityAgent",
]
