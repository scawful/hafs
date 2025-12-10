"""Multi-agent orchestration system."""

from hafs.agents.coordinator import AgentCoordinator
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

__all__ = [
    # Coordinator
    "AgentCoordinator",
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
]
