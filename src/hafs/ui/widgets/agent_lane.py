"""Agent lane widget for displaying a single AI agent."""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import RichLog, Static

if TYPE_CHECKING:
    from hafs.agents.lane import AgentLane


class AgentState(str, Enum):
    """Agent state indicators."""

    THINKING = "thinking"
    ACTIVE = "active"
    WAITING = "waiting"
    ERROR = "error"


class AgentStateIndicator(Widget):
    """Visual indicator for agent state.

    Shows animated indicator based on current agent state:
    - thinking: Yellow dots (...)
    - active: Green asterisks (***)
    - waiting: Dim dashes (---)
    - error: Red exclamation marks (!!!)

    Example:
        indicator = AgentStateIndicator()
        indicator.state = AgentState.THINKING
    """

    DEFAULT_CSS = """
    AgentStateIndicator {
        width: 100%;
        height: 1;
        background: transparent;
    }
    """

    state: reactive[AgentState] = reactive(AgentState.WAITING)

    def render(self) -> str:
        """Render the state indicator."""
        indicators = {
            AgentState.THINKING: "[yellow]...[/]",
            AgentState.ACTIVE: "[green]***[/]",
            AgentState.WAITING: "[dim]---[/]",
            AgentState.ERROR: "[red]!!![/]",
        }
        return indicators.get(self.state, "[dim]---[/]")

    def set_state(self, state: AgentState | str) -> None:
        """Set the indicator state.

        Args:
            state: New state (AgentState enum or string).
        """
        if isinstance(state, str):
            try:
                state = AgentState(state)
            except ValueError:
                state = AgentState.WAITING
        self.state = state


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
                yield AgentStateIndicator(id="state-indicator")

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
        self._update_state_indicator()

    def watch_is_busy(self, busy: bool) -> None:
        """React to busy state changes."""
        self._update_header()
        self._update_state_indicator()

    def _update_header(self) -> None:
        """Update the header text."""
        try:
            header = self.query_one("#header-text", Static)
            header.update(self._get_header_text())
        except Exception:
            pass

    def _update_state_indicator(self) -> None:
        """Update the state indicator based on current state."""
        try:
            indicator = self.query_one("#state-indicator", AgentStateIndicator)
            if self.is_busy:
                indicator.set_state(AgentState.THINKING)
            elif self.is_active:
                indicator.set_state(AgentState.ACTIVE)
            else:
                indicator.set_state(AgentState.WAITING)
        except Exception:
            pass

    def set_error_state(self) -> None:
        """Set the indicator to error state."""
        try:
            indicator = self.query_one("#state-indicator", AgentStateIndicator)
            indicator.set_state(AgentState.ERROR)
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
