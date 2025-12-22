"""High-Performance Session Workspace.

This is the next-generation chat interface for HAFS, designed to replace
the legacy ChatScreen. It features a 3-column layout, virtualized rendering,
and deep integration with the Agentic File System.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, Mapping

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Input

from tui.core.event_bus import (
    get_event_bus,
    StreamTokenEvent,
    ToolResultEvent,
    Event,
    PhaseEvent,
)
from tui.core.navigation_controller import get_navigation_controller
from tui.core.state_store import get_state_store
from tui.mixins.which_key import WhichKeyMixin, WhichKeyNode
from tui.widgets.workspace_widgets import (
    AgentRoster,
    SessionExplorer,
    ContextTree,
    SharedStateInspector,
    PlanTracker,
)
from tui.widgets.virtual_chat_stream import VirtualChatStream
from tui.widgets.streaming_message import StreamingMessage
from tui.widgets.tool_card import ToolCard

if TYPE_CHECKING:
    from agents.core.coordinator import AgentCoordinator


class NavigatorPanel(Vertical):
    """Left sidebar: Session explorer, Agent roster, Context tree."""

    DEFAULT_CSS = """
    NavigatorPanel {
        width: 25%;
        height: 100%;
        background: $surface;
        border-right: solid $primary;
    }
    """

    def compose(self) -> ComposeResult:
        yield SessionExplorer()
        yield AgentRoster()
        yield ContextTree()


class InspectorPanel(Vertical):
    """Right sidebar: Shared state, Plan tracker, Synergy metrics."""

    DEFAULT_CSS = """
    InspectorPanel {
        width: 25%;
        height: 100%;
        background: $surface;
        border-left: solid $primary;
    }
    """

    def compose(self) -> ComposeResult:
        yield SharedStateInspector()
        yield PlanTracker()
        yield Static("Synergy (TODO)", id="synergy-metrics")


class StagePanel(Vertical):
    """Center stage: Unified chat stream and composer."""

    DEFAULT_CSS = """
    StagePanel {
        width: 1fr;
        height: 100%;
        background: $background;
    }
    
    #composer-area {
        height: auto;
        min-height: 3;
        padding: 0 1;
        background: $surface;
        border-top: solid $primary;
    }
    
    #composer-input {
        width: 100%;
        border: none;
    }
    """

    def compose(self) -> ComposeResult:
        yield VirtualChatStream(id="chat-stream")
        with Vertical(id="composer-area"):
            yield Input(placeholder="Message agents... (@name, /command)", id="composer-input")


class SessionWorkspace(WhichKeyMixin, Screen):
    """The main workspace screen for agentic sessions."""

    BINDINGS = [
        Binding("ctrl+b", "toggle_navigator", "Toggle Nav", show=True),
        Binding("ctrl+i", "toggle_inspector", "Toggle Inspector", show=True),
        Binding("escape", "back", "Back"),
    ]

    DEFAULT_CSS = """
    SessionWorkspace {
        layout: horizontal;
        background: $background;
    }
    
    .hidden {
        display: none;
    }
    """

    def __init__(self, coordinator: AgentCoordinator | None = None) -> None:
        super().__init__()
        from tui.core.screen_router import get_screen_router

        self._router = get_screen_router()
        self._coordinator = coordinator
        self._nav_visible = True
        self._inspector_visible = True
        self._active_streams: dict[str, StreamingMessage] = {}

    def compose(self) -> ComposeResult:
        yield NavigatorPanel(id="navigator")
        yield StagePanel(id="stage")
        yield InspectorPanel(id="inspector")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize workspace on mount."""
        self._bus = get_event_bus()
        self._bus.subscribe("chat.stream_token", self._on_stream_token)
        self._bus.subscribe("tool.result", self._on_tool_result)
        self._bus.subscribe("agent.status", self._on_agent_status)
        self._bus.subscribe("phase.*", self._on_phase_event)

        if self._coordinator:
            self.run_worker(self._sync_initial_state())

    async def _sync_initial_state(self) -> None:
        """Sync initial state from coordinator."""
        if not self._coordinator:
            return

        # Update agent roster
        roster = cast(AgentRoster, self.query_one(AgentRoster))
        agents = []
        for name, lane in self._coordinator.agents.items():
            agents.append({"name": name, "status": "idle"})
        roster.agents = agents

        # Update shared state
        inspector = cast(SharedStateInspector, self.query_one(SharedStateInspector))
        inspector.state_data = self._coordinator.shared_context.model_dump()

    def _on_phase_event(self, event: Event) -> None:
        """Handle phase/plan updates."""
        if not isinstance(event, PhaseEvent):
            return

        tracker = cast(PlanTracker, self.query_one(PlanTracker))
        steps = list(tracker.steps)
        steps.append({"text": event.message or event.phase, "done": False})
        tracker.steps = steps

    def _on_stream_token(self, event: Event) -> None:
        """Handle streaming tokens."""
        if not isinstance(event, StreamTokenEvent):
            return
        stream = cast(VirtualChatStream, self.query_one(VirtualChatStream))

        msg_id = event.message_id
        agent_id = event.agent_id

        msg = stream.get_message(msg_id)

        if not msg:
            msg = StreamingMessage(
                agent_id=agent_id,
                agent_name=agent_id.title() if agent_id else "Assistant",
                role="assistant",
            )
            stream.append_message(msg)
            msg.start_streaming(msg_id)

        if event.is_final:
            msg.complete_streaming()
        else:
            msg.append_token(event.token)

    def _on_tool_result(self, event: Event) -> None:
        """Handle tool results."""
        if not isinstance(event, ToolResultEvent):
            return
        stream = cast(VirtualChatStream, self.query_one(VirtualChatStream))
        card = ToolCard.from_event(event)
        stream.append_message(card)

    def _on_agent_status(self, event: Event) -> None:
        """Update agent roster."""
        # TODO: Implement roster updates
        pass

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle composer input."""
        if event.input.id == "composer-input":
            message = event.value
            if not message.strip():
                return

            event.input.value = ""

            # Add user message to stream
            stream = cast(VirtualChatStream, self.query_one(VirtualChatStream))
            msg = StreamingMessage(agent_id="user", agent_name="You", role="user")
            stream.append_message(msg)
            msg.set_content(message)

            # Route to coordinator
            if self._coordinator:
                await self._coordinator.route_message(message, sender="user")

    def action_toggle_navigator(self) -> None:
        """Toggle the left navigator panel."""
        nav = self.query_one("#navigator", NavigatorPanel)
        self._nav_visible = not self._nav_visible
        nav.set_class(not self._nav_visible, "hidden")

    def action_toggle_inspector(self) -> None:
        """Toggle the right inspector panel."""
        inspector = self.query_one("#inspector", InspectorPanel)
        self._inspector_visible = not self._inspector_visible
        inspector.set_class(not self._inspector_visible, "hidden")

    def action_back(self) -> None:
        """Go back to dashboard."""
        self.app.pop_screen()

    async def on_header_bar_navigation_requested(self, event: Any) -> None:
        """Handle header bar navigation requests."""
        route_map = {
            "dashboard": "/dashboard",
            "chat": "/chat",
            "workspace": "/workspace",
            "logs": "/logs",
            "services": "/services",
            "analysis": "/analysis",
            "config": "/config",
        }
        route = route_map.get(event.screen)
        if route:
            await self._router.navigate(route)

    def get_which_key_map(self) -> WhichKeyNode:
        """Return which-key bindings."""
        return {
            "q": ("quit", self.app.exit),
            "b": ("toggle nav", self.action_toggle_navigator),
            "i": ("toggle inspector", self.action_toggle_inspector),
        }
