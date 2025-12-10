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
