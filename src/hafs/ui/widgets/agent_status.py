"""Agent Status widget for real-time agent monitoring and quick controls."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Button, Label, Static, RichLog
from textual.reactive import reactive
from textual.message import Message
from textual.timer import Timer


class AgentStatusWidget(Container):
    """Widget showing active agents, recent thoughts, and quick launch controls."""

    DEFAULT_CSS = """
    AgentStatusWidget {
        height: auto;
        min-height: 10;
        max-height: 30;
        background: $surface;
        border-top: solid $primary;
    }

    AgentStatusWidget .section-header {
        background: $primary;
        padding: 0 1;
        height: 1;
    }

    AgentStatusWidget #agent-list {
        height: auto;
        max-height: 6;
        padding: 0 1;
    }

    AgentStatusWidget .agent-row {
        height: 1;
        layout: horizontal;
    }

    AgentStatusWidget .agent-name {
        width: 1fr;
    }

    AgentStatusWidget .agent-status {
        width: auto;
        min-width: 8;
        text-align: right;
    }

    AgentStatusWidget .status-idle {
        color: $text-disabled;
    }

    AgentStatusWidget .status-active {
        color: $success;
    }

    AgentStatusWidget .status-thinking {
        color: $warning;
    }

    AgentStatusWidget #quick-actions {
        height: auto;
        padding: 0 1;
        layout: horizontal;
    }

    AgentStatusWidget #quick-actions Button {
        width: auto;
        min-width: 8;
        margin-right: 1;
        height: 1;
    }

    AgentStatusWidget #thought-feed {
        height: auto;
        max-height: 8;
        padding: 0 1;
        background: $background;
    }

    AgentStatusWidget .thought-entry {
        color: $text-disabled;
    }

    AgentStatusWidget .thought-entry-new {
        color: $warning;
    }
    """

    # Reactive state
    active_agents: reactive[list] = reactive(list)
    recent_thoughts: reactive[list] = reactive(list)

    class AgentLaunchRequested(Message):
        """Emitted when quick launch is requested."""
        def __init__(self, agent_type: str) -> None:
            self.agent_type = agent_type
            super().__init__()

    class ChatRequested(Message):
        """Emitted when chat is requested."""
        pass

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._refresh_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Label("[bold]Agents[/]", classes="section-header")

        with Vertical(id="agent-list"):
            yield Static("[dim]No active agents[/]", id="agents-placeholder")

        with Horizontal(id="quick-actions"):
            yield Button("Chat", variant="primary", id="btn-chat")
            yield Button("Swarm", id="btn-swarm")
            yield Button("Embed", id="btn-embed")

        yield Label("[bold]Thoughts[/]", classes="section-header", id="thoughts-header")
        yield RichLog(id="thought-feed", highlight=True, markup=True, max_lines=50)

    async def on_mount(self) -> None:
        """Start periodic refresh."""
        await self.refresh_status()
        # Refresh every 5 seconds
        self._refresh_timer = self.set_interval(5.0, self._periodic_refresh)

    def on_unmount(self) -> None:
        """Stop timer on unmount."""
        if self._refresh_timer:
            self._refresh_timer.stop()

    async def _periodic_refresh(self) -> None:
        """Periodic refresh handler."""
        await self.refresh_status()

    async def refresh_status(self) -> None:
        """Refresh agent status and recent thoughts."""
        # Get active agents from coordinator if available
        try:
            coordinator = getattr(self.app, "_coordinator", None)
            if coordinator and hasattr(coordinator, "get_agent_status"):
                self.active_agents = await coordinator.get_agent_status()
            else:
                self.active_agents = []
        except Exception:
            self.active_agents = []

        # Get recent thought traces from history
        try:
            from hafs.core.history.logger import HistoryLogger
            from hafs.core.history.models import HistoryQuery, OperationType

            history_dir = Path.home() / ".context" / "history"
            if history_dir.exists():
                logger = HistoryLogger(history_dir)
                recent = logger.query(HistoryQuery(
                    operation_types=[OperationType.THOUGHT_TRACE],
                    limit=10
                ))

                self.recent_thoughts = [
                    {
                        "time": entry.timestamp,
                        "model": entry.operation.name,
                        "preview": str(entry.operation.input.get("thought_content", ""))[:100]
                    }
                    for entry in recent
                ]
        except Exception:
            self.recent_thoughts = []

        self._update_display()

    def _update_display(self) -> None:
        """Update the display with current state."""
        try:
            # Update agent list
            placeholder = self.query_one("#agents-placeholder", Static)

            if self.active_agents:
                lines = []
                for agent in self.active_agents[:5]:
                    name = agent.get("name", "Unknown")
                    status = agent.get("status", "idle")
                    status_class = f"status-{status}"
                    lines.append(f"  [bold]{name}[/] [{status_class}]{status}[/]")
                placeholder.update("\n".join(lines))
            else:
                placeholder.update(
                    "[dim]No active agents[/]\n"
                    "[dim]Press [bold]c[/] for chat, [bold]4[/] for full chat screen[/]"
                )

            # Update thought feed
            thought_feed = self.query_one("#thought-feed", RichLog)

            # Only add new thoughts (check timestamp)
            for thought in self.recent_thoughts[:5]:
                time_str = thought.get("time", "")[:19]
                model = thought.get("model", "unknown")
                preview = thought.get("preview", "")

                if preview and len(preview) > 0:
                    # Clean up binary data display
                    if preview.startswith("b'"):
                        preview = "[binary thought signature]"

                    thought_feed.write(
                        f"[dim]{time_str}[/] [bold]{model}[/]: {preview[:60]}..."
                    )

            # Update header with count
            header = self.query_one("#thoughts-header", Label)
            count = len(self.recent_thoughts)
            header.update(f"[bold]Thoughts[/] [dim]({count} recent)[/]")

        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle quick action buttons."""
        if event.button.id == "btn-chat":
            self.post_message(self.ChatRequested())
        elif event.button.id == "btn-swarm":
            self.post_message(self.AgentLaunchRequested("swarm"))
        elif event.button.id == "btn-embed":
            self.post_message(self.AgentLaunchRequested("embed"))

    def add_thought(self, model: str, content: str) -> None:
        """Add a new thought to the feed (for real-time updates)."""
        try:
            thought_feed = self.query_one("#thought-feed", RichLog)
            time_str = datetime.now().strftime("%H:%M:%S")
            preview = content[:60] if content else "[empty]"
            thought_feed.write(
                f"[warning]{time_str}[/] [bold]{model}[/]: {preview}..."
            )
        except Exception:
            pass
