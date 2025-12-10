"""Synergy panel widget for displaying Theory of Mind metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections import deque

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import ProgressBar, Static

if TYPE_CHECKING:
    from hafs.models.synergy import SynergyScore
    from hafs.models.agent import AgentMessage


class MessageFlowWidget(Widget):
    """Visual representation of agent message flow.

    Shows recent messages between agents with direction indicators.

    Example:
        flow = MessageFlowWidget(max_messages=10)
        flow.add_message(agent_message)
    """

    DEFAULT_CSS = """
    MessageFlowWidget {
        height: 100%;
        width: 25;
        background: $surface;
        border-left: solid $primary;
        padding: 0 1;
    }

    MessageFlowWidget .flow-title {
        text-style: bold;
        color: $text-muted;
        margin-bottom: 1;
    }

    MessageFlowWidget .flow-item {
        height: 1;
    }

    MessageFlowWidget .flow-sender {
        color: $info;
    }

    MessageFlowWidget .flow-recipient {
        color: $secondary;
    }

    MessageFlowWidget .flow-arrow {
        color: $text-muted;
    }

    MessageFlowWidget .flow-broadcast {
        color: $warning;
    }
    """

    def __init__(
        self,
        max_messages: int = 8,
        id: str | None = None,
        classes: str | None = None,
    ):
        """Initialize message flow widget.

        Args:
            max_messages: Maximum messages to display.
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._max_messages = max_messages
        self._messages: deque[dict] = deque(maxlen=max_messages)

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        yield Static("Flow", classes="flow-title")
        yield Static("[dim]No messages yet[/dim]", id="flow-list")

    def add_message(self, sender: str, recipient: str | None = None) -> None:
        """Add a message to the flow visualization.

        Args:
            sender: Name of the message sender.
            recipient: Name of the recipient (None for broadcast).
        """
        self._messages.append({"sender": sender, "recipient": recipient})
        self._update_display()

    def add_agent_message(self, msg: "AgentMessage") -> None:
        """Add an AgentMessage to the flow visualization.

        Args:
            msg: The AgentMessage to display.
        """
        self.add_message(msg.sender, msg.recipient)

    def _update_display(self) -> None:
        """Update the flow display."""
        flow_list = self.query_one("#flow-list", Static)

        if not self._messages:
            flow_list.update("[dim]No messages yet[/dim]")
            return

        lines = []
        for msg in self._messages:
            sender = msg["sender"][:6]
            recipient = msg["recipient"]

            if recipient:
                # Direct message
                recip_short = recipient[:6]
                lines.append(f"[cyan]{sender}[/] > [magenta]{recip_short}[/]")
            else:
                # Broadcast
                lines.append(f"[cyan]{sender}[/] >> [yellow]all[/]")

        flow_list.update("\n".join(lines))

    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()
        self._update_display()


class SynergyPanel(Widget):
    """Panel displaying Theory of Mind metrics and synergy score.

    Shows:
    - Overall synergy score (0-100)
    - ToM markers score
    - Response quality score
    - User alignment score
    - Context utilization score

    Example:
        panel = SynergyPanel(id="synergy-panel")
        panel.update_score(synergy_score)
    """

    DEFAULT_CSS = """
    SynergyPanel {
        height: 9;
        dock: bottom;
        background: $surface;
        border-top: solid $primary;
        padding: 1;
    }

    SynergyPanel .panel-title {
        text-style: bold;
        color: $text;
        padding-bottom: 1;
    }

    SynergyPanel .score-section {
        width: 12;
        height: 100%;
        padding-right: 1;
    }

    SynergyPanel .score-value {
        text-style: bold;
        color: $accent;
    }

    SynergyPanel .score-label {
        color: $text-muted;
    }

    SynergyPanel .metrics-section {
        width: 1fr;
        height: 100%;
    }

    SynergyPanel .metric-row {
        height: 1;
        width: 100%;
    }

    SynergyPanel .metric-label {
        width: 18;
        color: $text-muted;
    }

    SynergyPanel .metric-bar {
        width: 1fr;
    }

    SynergyPanel .score-high {
        color: $success;
    }

    SynergyPanel .score-medium {
        color: $warning;
    }

    SynergyPanel .score-low {
        color: $error;
    }

    SynergyPanel .flow-section {
        width: 22;
        height: 100%;
        border-left: solid $primary;
        padding-left: 1;
    }
    """

    synergy_total: reactive[float] = reactive(0.0)

    def __init__(
        self,
        id: str | None = None,
        classes: str | None = None,
    ):
        """Initialize synergy panel.

        Args:
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._score: "SynergyScore | None" = None

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        with Horizontal():
            # Overall score section
            with Vertical(classes="score-section"):
                yield Static("Synergy", classes="score-label")
                yield Static("--", id="score-value", classes="score-value")

            # Metrics breakdown section
            with Vertical(classes="metrics-section"):
                # ToM Markers
                with Horizontal(classes="metric-row"):
                    yield Static("ToM Markers", classes="metric-label")
                    yield ProgressBar(id="tom-bar", total=100, show_percentage=True)

                # Response Quality
                with Horizontal(classes="metric-row"):
                    yield Static("Quality", classes="metric-label")
                    yield ProgressBar(id="quality-bar", total=100, show_percentage=True)

                # User Alignment
                with Horizontal(classes="metric-row"):
                    yield Static("Alignment", classes="metric-label")
                    yield ProgressBar(id="alignment-bar", total=100, show_percentage=True)

                # Context Utilization
                with Horizontal(classes="metric-row"):
                    yield Static("Context", classes="metric-label")
                    yield ProgressBar(id="context-bar", total=100, show_percentage=True)

            # Message flow section
            with Vertical(classes="flow-section"):
                yield MessageFlowWidget(id="message-flow", max_messages=6)

    def update_score(self, score: "SynergyScore") -> None:
        """Update displayed score.

        Args:
            score: The SynergyScore to display.
        """
        self._score = score
        self.synergy_total = score.total

        # Update score value with color based on level
        score_widget = self.query_one("#score-value", Static)
        score_class = self._get_score_class(score.total)
        score_widget.update(f"[{score_class}]{score.total:.0f}[/]")

        # Update progress bars
        self._update_bar("tom-bar", score.breakdown.get("tom_markers", 0))
        self._update_bar("quality-bar", score.breakdown.get("response_quality", 0))
        self._update_bar("alignment-bar", score.breakdown.get("user_alignment", 0))
        self._update_bar("context-bar", score.breakdown.get("context_utilization", 0))

    def _update_bar(self, bar_id: str, value: float) -> None:
        """Update a progress bar.

        Args:
            bar_id: ID of the progress bar.
            value: Value (0-100).
        """
        try:
            bar = self.query_one(f"#{bar_id}", ProgressBar)
            bar.progress = value
        except Exception:
            pass

    def _get_score_class(self, score: float) -> str:
        """Get CSS class for score color.

        Args:
            score: Score value (0-100).

        Returns:
            CSS class name.
        """
        if score >= 70:
            return "green"
        elif score >= 40:
            return "yellow"
        else:
            return "red"

    def reset(self) -> None:
        """Reset all scores to zero."""
        self.synergy_total = 0.0
        self._score = None

        score_widget = self.query_one("#score-value", Static)
        score_widget.update("--")

        for bar_id in ["tom-bar", "quality-bar", "alignment-bar", "context-bar"]:
            self._update_bar(bar_id, 0)

    @property
    def current_score(self) -> "SynergyScore | None":
        """Get the current synergy score."""
        return self._score

    def add_message(self, sender: str, recipient: str | None = None) -> None:
        """Add a message to the flow visualization.

        Args:
            sender: Name of the message sender.
            recipient: Name of the recipient (None for broadcast).
        """
        try:
            flow = self.query_one("#message-flow", MessageFlowWidget)
            flow.add_message(sender, recipient)
        except Exception:
            pass

    def add_agent_message(self, msg: "AgentMessage") -> None:
        """Add an AgentMessage to the flow visualization.

        Args:
            msg: The AgentMessage to display.
        """
        try:
            flow = self.query_one("#message-flow", MessageFlowWidget)
            flow.add_agent_message(msg)
        except Exception:
            pass

    def clear_flow(self) -> None:
        """Clear the message flow."""
        try:
            flow = self.query_one("#message-flow", MessageFlowWidget)
            flow.clear()
        except Exception:
            pass
