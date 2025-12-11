"""Container widget for multiple agent lanes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widget import Widget

from hafs.ui.widgets.agent_lane import AgentLaneWidget

if TYPE_CHECKING:
    from hafs.agents.lane import AgentLane


class LaneContainer(Widget):
    """Container for multiple agent lane widgets.

    Manages horizontal layout of agent lanes with dynamic add/remove.

    Example:
        container = LaneContainer(id="lanes")
        await container.add_lane(planner_lane)
        await container.add_lane(coder_lane)
    """

    DEFAULT_CSS = """
    LaneContainer {
        width: 100%;
        height: 1fr;
    }

    LaneContainer > Horizontal {
        width: 100%;
        height: 100%;
    }

    /* Focus mode - hide non-focused lanes */
    LaneContainer .lane-hidden {
        display: none;
    }

    /* Focus mode - focused lane takes full width */
    LaneContainer .lane-focused {
        width: 100%;
    }

    /* Multi mode - truncated lanes */
    LaneContainer .lane-truncated {
        height: 12;
        overflow: hidden;
    }
    """

    class LaneAdded(Message):
        """Message sent when a lane is added."""

        def __init__(self, lane_id: str):
            self.lane_id = lane_id
            super().__init__()

    class LaneRemoved(Message):
        """Message sent when a lane is removed."""

        def __init__(self, lane_id: str):
            self.lane_id = lane_id
            super().__init__()

    def __init__(
        self,
        id: str | None = None,
        classes: str | None = None,
    ):
        """Initialize lane container.

        Args:
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._lanes: dict[str, AgentLaneWidget] = {}
        self._view_mode: str = "focus"  # "focus" or "multi"
        self._focused_lane_id: str | None = None

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        yield Horizontal(id="lane-row")

    async def add_lane(self, lane: "AgentLane", lane_id: str | None = None) -> AgentLaneWidget:
        """Add an agent lane to the container.

        Args:
            lane: The AgentLane to add.
            lane_id: Optional ID for the lane widget.

        Returns:
            The created AgentLaneWidget.
        """
        widget_id = lane_id or f"lane-{lane.agent.name.lower()}"

        if widget_id in self._lanes:
            return self._lanes[widget_id]

        lane_widget = AgentLaneWidget(lane, id=widget_id)
        self._lanes[widget_id] = lane_widget

        lane_row = self.query_one("#lane-row", Horizontal)
        await lane_row.mount(lane_widget)

        # Start streaming immediately so we capture all PTY output
        await lane_widget.start_streaming()

        self.post_message(self.LaneAdded(widget_id))
        return lane_widget

    async def remove_lane(self, lane_id: str) -> bool:
        """Remove an agent lane from the container.

        Args:
            lane_id: ID of the lane to remove.

        Returns:
            True if lane was removed.
        """
        if lane_id not in self._lanes:
            return False

        lane_widget = self._lanes[lane_id]
        await lane_widget.stop_streaming()
        await lane_widget.remove()

        del self._lanes[lane_id]
        self.post_message(self.LaneRemoved(lane_id))
        return True

    def get_lane(self, lane_id: str) -> AgentLaneWidget | None:
        """Get a lane widget by ID.

        Args:
            lane_id: Lane ID.

        Returns:
            The lane widget, or None if not found.
        """
        return self._lanes.get(lane_id)

    def get_lane_by_agent_name(self, name: str) -> AgentLaneWidget | None:
        """Get a lane widget by agent name.

        Args:
            name: Agent name.

        Returns:
            The lane widget, or None if not found.
        """
        for lane in self._lanes.values():
            if lane.lane.agent.name.lower() == name.lower():
                return lane
        return None

    @property
    def lane_ids(self) -> list[str]:
        """Get all lane IDs."""
        return list(self._lanes.keys())

    @property
    def lane_count(self) -> int:
        """Get number of lanes."""
        return len(self._lanes)

    async def clear_all(self) -> None:
        """Remove all lanes."""
        for lane_id in list(self._lanes.keys()):
            await self.remove_lane(lane_id)

    def set_view_mode(self, mode: str) -> None:
        """Set the view mode for lanes.

        Args:
            mode: "focus" for single lane view, "multi" for all lanes visible.
        """
        self._view_mode = mode

        if mode == "focus":
            # In focus mode, show only the focused lane
            if self._focused_lane_id:
                self.set_focused_lane(self._focused_lane_id)
        else:
            # In multi mode, show all lanes
            self._show_all_lanes()

    def set_focused_lane(self, lane_id: str) -> None:
        """Focus on a single lane, hiding others in focus mode.

        Args:
            lane_id: ID of the lane to focus.
        """
        self._focused_lane_id = lane_id

        if self._view_mode == "focus":
            # Hide all lanes except the focused one
            for lid, lane in self._lanes.items():
                if lid == lane_id:
                    lane.remove_class("lane-hidden")
                    lane.add_class("lane-focused")
                    lane.remove_class("lane-truncated")
                else:
                    lane.add_class("lane-hidden")
                    lane.remove_class("lane-focused")
                    lane.remove_class("lane-truncated")
        else:
            # In multi mode, just highlight the focused lane
            for lid, lane in self._lanes.items():
                lane.remove_class("lane-hidden")
                lane.remove_class("lane-focused")
                if lid == lane_id:
                    lane.remove_class("lane-truncated")
                elif len(self._lanes) >= 3:
                    # Truncate non-focused lanes when 3+ agents
                    lane.add_class("lane-truncated")

    def _show_all_lanes(self) -> None:
        """Show all lanes (for multi mode)."""
        for lid, lane in self._lanes.items():
            lane.remove_class("lane-hidden")
            lane.remove_class("lane-focused")
            # Truncate lanes when 3+ agents in multi mode
            if len(self._lanes) >= 3 and lid != self._focused_lane_id:
                lane.add_class("lane-truncated")
            else:
                lane.remove_class("lane-truncated")

    @property
    def view_mode(self) -> str:
        """Get current view mode."""
        return self._view_mode

    @property
    def focused_lane_id(self) -> str | None:
        """Get the currently focused lane ID."""
        return self._focused_lane_id
