"""Agent lane widget for displaying a single AI agent."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import RichLog, Static

if TYPE_CHECKING:
    from hafs.agents.lane import AgentLane


class AgentLaneWidget(Widget):
    """Widget displaying a single agent lane.

    Shows agent name, role, status indicator, and streaming terminal output.

    Example:
        lane_widget = AgentLaneWidget(agent_lane, id="lane-planner")
        await lane_widget.start_streaming()
    """

    DEFAULT_CSS = """
    AgentLaneWidget {
        width: 1fr;
        height: 100%;
        border: solid $primary;
        margin: 0 1;
    }

    AgentLaneWidget .lane-header {
        height: 3;
        background: $primary;
        padding: 0 1;
    }

    AgentLaneWidget .lane-header-row {
        height: auto;
        width: 100%;
    }

    AgentLaneWidget .lane-terminal {
        height: 1fr;
        background: $surface;
        padding: 0 1;
    }

    AgentLaneWidget .status-active {
        color: $success;
    }

    AgentLaneWidget .status-inactive {
        color: $error;
    }

    AgentLaneWidget .status-busy {
        color: $warning;
    }

    AgentLaneWidget .role-label {
        color: $text-muted;
    }
    """

    is_active: reactive[bool] = reactive(False)
    is_busy: reactive[bool] = reactive(False)

    class OutputReceived(Message):
        """Message sent when output is received from the agent."""

        def __init__(self, lane_id: str, content: str):
            self.lane_id = lane_id
            self.content = content
            super().__init__()

    class AgentStopped(Message):
        """Message sent when the agent stops."""

        def __init__(self, lane_id: str):
            self.lane_id = lane_id
            super().__init__()

    def __init__(
        self,
        lane: "AgentLane",
        id: str | None = None,
        classes: str | None = None,
    ):
        """Initialize agent lane widget.

        Args:
            lane: The AgentLane to display.
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self.lane = lane
        self._stream_task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        with Vertical():
            # Header with agent info
            with Vertical(classes="lane-header"):
                yield Static(self._get_header_text(), id="header-text")

            # Terminal output area
            yield RichLog(id="terminal", classes="lane-terminal", highlight=True, markup=True)

    def _get_header_text(self) -> str:
        """Generate header text with status indicator."""
        agent = self.lane.agent

        if self.is_busy:
            status = "[yellow]◉[/]"
        elif self.is_active:
            status = "[green]●[/]"
        else:
            status = "[red]○[/]"

        return f"{status} [bold]{agent.name}[/] [dim]({agent.role.value})[/]"

    def watch_is_active(self, active: bool) -> None:
        """React to active state changes."""
        self._update_header()

    def watch_is_busy(self, busy: bool) -> None:
        """React to busy state changes."""
        self._update_header()

    def _update_header(self) -> None:
        """Update the header text."""
        try:
            header = self.query_one("#header-text", Static)
            header.update(self._get_header_text())
        except Exception:
            pass

    async def start_streaming(self) -> None:
        """Start streaming output from the agent."""
        if self._stream_task and not self._stream_task.done():
            return

        self._stream_task = asyncio.create_task(self._stream_loop())

    async def _stream_loop(self) -> None:
        """Background task to stream agent output."""
        terminal = self.query_one("#terminal", RichLog)

        try:
            async for chunk in self.lane.stream_output():
                terminal.write(chunk)
                self.post_message(self.OutputReceived(self.id or "", chunk))
        except asyncio.CancelledError:
            pass
        finally:
            self.post_message(self.AgentStopped(self.id or ""))

    async def stop_streaming(self) -> None:
        """Stop streaming output."""
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None

    def append_user_message(self, message: str) -> None:
        """Append a user message to the terminal.

        Args:
            message: User's message.
        """
        terminal = self.query_one("#terminal", RichLog)
        terminal.write("")
        terminal.write(f"[bold cyan]You:[/] {message}")
        terminal.write("")

    def append_separator(self) -> None:
        """Append a visual separator to the terminal."""
        terminal = self.query_one("#terminal", RichLog)
        terminal.write("[dim]" + "─" * 40 + "[/]")

    def clear_terminal(self) -> None:
        """Clear the terminal output."""
        terminal = self.query_one("#terminal", RichLog)
        terminal.clear()

    async def send_message(self, message: str) -> None:
        """Send a message to this agent.

        Args:
            message: Message to send.
        """
        self.is_busy = True
        self.append_user_message(message)

        try:
            await self.lane.send_message(message)
            await self.start_streaming()
        finally:
            self.is_busy = False
