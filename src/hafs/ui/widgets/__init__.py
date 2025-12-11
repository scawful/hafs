"""TUI widgets for HAFS."""

# Orchestration widgets
from hafs.ui.widgets.agent_lane import AgentLaneWidget
from hafs.ui.widgets.chat_input import ChatInput
from hafs.ui.widgets.context_panel import ContextPanel
from hafs.ui.widgets.context_viewer import ContextViewer
from hafs.ui.widgets.lane_container import LaneContainer
from hafs.ui.widgets.mode_toggle import ModeToggle
from hafs.ui.widgets.plan_viewer import PlanViewer
from hafs.ui.widgets.project_tree import ProjectTree
from hafs.ui.widgets.session_list import SessionList
from hafs.ui.widgets.stats_panel import StatsPanel
from hafs.ui.widgets.synergy_panel import SynergyPanel

__all__ = [
    # Existing widgets
    "ProjectTree",
    "ContextViewer",
    "SessionList",
    "PlanViewer",
    "StatsPanel",
    # Orchestration widgets
    "AgentLaneWidget",
    "ChatInput",
    "ContextPanel",
    "LaneContainer",
    "ModeToggle",
    "SynergyPanel",
]
