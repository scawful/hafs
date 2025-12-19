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
from textual.widgets import Static

from hafs.ui.widgets.terminal_emulator import TerminalDisplay

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

            # Terminal output area using pyte-based terminal emulation
            yield TerminalDisplay(id="terminal", classes="lane-terminal")

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

    def _on_raw_output(self, data: str) -> None:
        """Handle raw PTY output by feeding it to the terminal display.

        This callback is called by the backend's PTY reader task with
        raw terminal data (including ANSI escape sequences).

        Args:
            data: Raw terminal output string.
        """
        try:
            terminal = self.query_one("#terminal", TerminalDisplay)
            terminal.feed(data)
            self.post_message(self.OutputReceived(self.id or "", data))
        except Exception:
            pass

    def _write_to_pty(self, data: str) -> None:
        """Write data to the PTY stdin.

        This callback is used by TerminalDisplay to send keyboard input
        to the backend's PTY process.

        Args:
            data: Raw string data to write.
        """
        self.lane.write_raw(data)

    async def start_streaming(self) -> None:
        """Start streaming output from the agent.

        This sets up the raw output callback to feed PTY data directly
        to the TerminalDisplay widget for proper terminal emulation.
        It also sets up the write callback so keyboard input goes to the PTY.
        """
        # Hook up raw output to terminal display
        self.lane.set_raw_output_callback(self._on_raw_output)

        # Hook up keyboard input to PTY
        try:
            terminal = self.query_one("#terminal", TerminalDisplay)
            terminal.set_write_callback(self._write_to_pty)
        except Exception:
            pass

    async def _stream_loop(self) -> None:
        """Background task to stream agent output (legacy - kept for compatibility)."""
        # This method is no longer the primary streaming mechanism.
        # Raw output is now fed directly via callback to TerminalDisplay.
        pass

    async def stop_streaming(self) -> None:
        """Stop streaming output."""
        # Clear the raw output callback
        self.lane.set_raw_output_callback(None)

        # Clear the write callback
        try:
            terminal = self.query_one("#terminal", TerminalDisplay)
            terminal.set_write_callback(None)
        except Exception:
            pass

        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None

    def append_user_message(self, message: str) -> None:
        """Append a user message to the terminal.

        For TerminalDisplay, we feed text that will be rendered by pyte.
        Since the terminal emulates a real terminal, we send the text
        with ANSI color codes.

        Args:
            message: User's message.
        """
        try:
            terminal = self.query_one("#terminal", TerminalDisplay)
            # Feed text with ANSI color codes for cyan bold
            user_text = f"\n\x1b[1;36mYou:\x1b[0m {message}\n"
            terminal.feed(user_text)
        except Exception:
            pass

    def append_separator(self) -> None:
        """Append a visual separator to the terminal."""
        try:
            terminal = self.query_one("#terminal", TerminalDisplay)
            # Feed dim text separator
            separator = "\x1b[2m" + "─" * 40 + "\x1b[0m\n"
            terminal.feed(separator)
        except Exception:
            pass

    def clear_terminal(self) -> None:
        """Clear the terminal output."""
        try:
            terminal = self.query_one("#terminal", TerminalDisplay)
            terminal.clear()
        except Exception:
            pass

    async def send_message(self, message: str) -> None:
        """Send a message to this agent.

        Args:
            message: Message to send.
        """
        from hafs.models.agent import AgentMessage

        self.is_busy = True
        self.append_user_message(message)

        try:
            agent_message = AgentMessage(
                content=message,
                sender="user",
                recipient=self.lane.agent.name,
            )
            await self.lane.receive_message(agent_message)
            await self.lane.process_next_message()
            await self.start_streaming()
        finally:
            self.is_busy = False
