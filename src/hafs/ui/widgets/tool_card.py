"""Tool Card Widget - Collapsible display for tool execution results.

This widget provides a rich display for tool execution results with:
- Collapsible/expandable view
- Syntax-highlighted output using Rich markup
- Duration and status indicators
- Copy button support
- Error state visualization
- Artifact links (if applicable)

Features:
- Compact collapsed state showing summary
- Expanded state with full output
- Color-coded status (success/error/warning)
- Execution duration display
- Standard output and error separation
- Support for copying output to clipboard

Usage:
    # Create tool card
    card = ToolCard(
        tool_name="pytest",
        stdout="All tests passed",
        duration_ms=1523,
        success=True
    )

    # Subscribe to tool events
    bus.subscribe("tool.result", self._on_tool_result)

    def _on_tool_result(self, event):
        card = ToolCard.from_event(event)
        container.mount(card)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from rich.syntax import Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Label, RichLog, Static

from hafs.ui.core.event_bus import Event, ToolResultEvent, get_event_bus

logger = logging.getLogger(__name__)


class ToolCard(Widget):
    """A collapsible card displaying tool execution results.

    Shows tool name, execution status, duration, and output with
    syntax highlighting and expand/collapse functionality.

    Attributes:
        tool_name: Name of the executed tool
        stdout: Standard output from the tool
        stderr: Standard error from the tool
        duration_ms: Execution duration in milliseconds
        success: Whether execution succeeded
        artifacts: List of artifact file paths
        agent_id: Agent that executed the tool
        is_expanded: Whether the card is expanded
    """

    DEFAULT_CSS = """
    ToolCard {
        width: 100%;
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        border: solid $primary;
        background: $panel;
    }

    ToolCard.success {
        border: solid $success;
    }

    ToolCard.error {
        border: solid $error;
    }

    ToolCard .card-header {
        height: 3;
        width: 100%;
        padding: 1;
        background: $primary;
    }

    ToolCard.success .card-header {
        background: $success-darken-2;
    }

    ToolCard.error .card-header {
        background: $error-darken-2;
    }

    ToolCard .header-row {
        width: 100%;
        height: auto;
    }

    ToolCard .tool-name {
        width: 1fr;
        color: $text;
    }

    ToolCard .duration {
        width: auto;
        color: $text-disabled;
        padding: 0 1;
    }

    ToolCard .status-icon {
        width: 3;
        content-align: center middle;
    }

    ToolCard .toggle-button {
        width: auto;
        min-width: 12;
        height: 1;
    }

    ToolCard .copy-button {
        width: auto;
        min-width: 8;
        height: 1;
        margin-left: 1;
    }

    ToolCard .card-content {
        width: 100%;
        height: auto;
        padding: 1;
    }

    ToolCard .output-section {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    ToolCard .section-label {
        height: 1;
        width: 100%;
        color: $text-disabled;
    }

    ToolCard .output-log {
        width: 100%;
        height: auto;
        max-height: 20;
        background: $surface;
        border: solid $primary;
        padding: 1;
    }

    ToolCard .artifacts-list {
        width: 100%;
        height: auto;
        padding: 1;
        background: $surface-lighten-1;
        border-left: thick $accent;
    }

    ToolCard .collapsed {
        display: none;
    }
    """

    is_expanded: reactive[bool] = reactive(False)

    class CopyRequested(Message):
        """Message sent when copy button is pressed."""

        def __init__(self, content: str):
            self.content = content
            super().__init__()

    def __init__(
        self,
        tool_name: str,
        stdout: str = "",
        stderr: str = "",
        duration_ms: int = 0,
        success: bool = True,
        artifacts: Optional[List[Path]] = None,
        agent_id: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ):
        """Initialize tool card widget.

        Args:
            tool_name: Name of the executed tool
            stdout: Standard output
            stderr: Standard error
            duration_ms: Execution duration in milliseconds
            success: Whether execution succeeded
            artifacts: List of artifact file paths
            agent_id: Agent that executed the tool
            id: Widget ID
            classes: CSS classes
        """
        super().__init__(id=id, classes=classes)
        self.tool_name = tool_name
        self.stdout = stdout
        self.stderr = stderr
        self.duration_ms = duration_ms
        self.success = success
        self.artifacts = artifacts or []
        self.agent_id = agent_id or ""

        # Add status class
        self.add_class("success" if success else "error")

    @classmethod
    def from_event(cls, event: ToolResultEvent) -> "ToolCard":
        """Create a ToolCard from a ToolResultEvent.

        Args:
            event: ToolResultEvent from EventBus

        Returns:
            New ToolCard instance
        """
        artifacts = [Path(p) for p in event.artifacts]

        return cls(
            tool_name=event.tool_name,
            stdout=event.stdout,
            stderr=event.stderr,
            duration_ms=event.duration_ms,
            success=event.success,
            artifacts=artifacts,
            agent_id=event.agent_id,
        )

    def compose(self) -> ComposeResult:
        """Compose the tool card layout."""
        with Container(classes="card-header"):
            with Horizontal(classes="header-row"):
                yield Static(self._get_status_icon(), classes="status-icon")
                yield Label(self._get_tool_label(), classes="tool-name")
                yield Static(self._get_duration_text(), classes="duration")
                yield Button(
                    "▼ Expand" if not self.is_expanded else "▲ Collapse",
                    id="toggle",
                    classes="toggle-button",
                    variant="default",
                )
                yield Button("Copy", id="copy", classes="copy-button", variant="primary")

        with Container(classes="card-content" + (" collapsed" if not self.is_expanded else "")):
            # Standard output section
            if self.stdout:
                with Vertical(classes="output-section"):
                    yield Static("[bold]Standard Output[/]", classes="section-label")
                    yield RichLog(
                        id="stdout-log",
                        classes="output-log",
                        highlight=True,
                        markup=True,
                        wrap=True,
                    )

            # Standard error section
            if self.stderr:
                with Vertical(classes="output-section"):
                    yield Static("[bold red]Standard Error[/]", classes="section-label")
                    yield RichLog(
                        id="stderr-log",
                        classes="output-log",
                        highlight=True,
                        markup=True,
                        wrap=True,
                    )

            # Artifacts section
            if self.artifacts:
                with Vertical(classes="output-section"):
                    yield Static("[bold]Artifacts[/]", classes="section-label")
                    yield Static(self._get_artifacts_text(), classes="artifacts-list")

    def on_mount(self) -> None:
        """Populate output logs on mount."""
        if self.is_expanded:
            self._populate_output()

    def _get_status_icon(self) -> str:
        """Get status icon based on success state."""
        if self.success:
            return "[green]✓[/]"
        return "[red]✗[/]"

    def _get_tool_label(self) -> str:
        """Get formatted tool name label."""
        agent_prefix = f"[dim]{self.agent_id}:[/] " if self.agent_id else ""
        return f"{agent_prefix}[bold]{self.tool_name}[/]"

    def _get_duration_text(self) -> str:
        """Format duration for display."""
        if self.duration_ms < 1000:
            return f"{self.duration_ms}ms"
        seconds = self.duration_ms / 1000
        return f"{seconds:.2f}s"

    def _get_artifacts_text(self) -> str:
        """Format artifacts list for display."""
        lines = []
        for artifact in self.artifacts:
            lines.append(f"  [cyan]→[/] {artifact}")
        return "\n".join(lines) if lines else "[dim]No artifacts[/]"

    def _populate_output(self) -> None:
        """Populate output logs with content."""
        # Populate stdout
        if self.stdout:
            try:
                stdout_log = self.query_one("#stdout-log", RichLog)
                stdout_log.clear()

                # Try to detect language for syntax highlighting
                if self._looks_like_code(self.stdout):
                    syntax = Syntax(
                        self.stdout,
                        "python",  # Default to Python, could be enhanced
                        theme="monokai",
                        line_numbers=False,
                    )
                    stdout_log.write(syntax)
                else:
                    # Plain text with potential ANSI colors preserved
                    stdout_log.write(self.stdout)

            except Exception as e:
                logger.debug(f"Failed to populate stdout: {e}")

        # Populate stderr
        if self.stderr:
            try:
                stderr_log = self.query_one("#stderr-log", RichLog)
                stderr_log.clear()

                # Error output - highlight in red
                stderr_text = Text(self.stderr, style="red")
                stderr_log.write(stderr_text)

            except Exception as e:
                logger.debug(f"Failed to populate stderr: {e}")

    def _looks_like_code(self, text: str) -> bool:
        """Heuristic to detect if text looks like code.

        Args:
            text: Text to check

        Returns:
            True if text appears to be code
        """
        code_indicators = [
            "def ",
            "class ",
            "import ",
            "function ",
            "const ",
            "let ",
            "var ",
            "=>",
            "{",
            "}",
        ]
        return any(indicator in text for indicator in code_indicators)

    def watch_is_expanded(self, expanded: bool) -> None:
        """React to expansion state changes."""
        # Update toggle button
        try:
            toggle = self.query_one("#toggle", Button)
            toggle.label = "▲ Collapse" if expanded else "▼ Expand"
        except Exception as e:
            logger.debug(f"Failed to update toggle button: {e}")

        # Show/hide content
        try:
            content = self.query_one(".card-content", Container)
            content.set_class(not expanded, "collapsed")

            if expanded:
                self._populate_output()

        except Exception as e:
            logger.debug(f"Failed to toggle content: {e}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "toggle":
            self.toggle_expanded()
        elif button_id == "copy":
            self.copy_output()

    def toggle_expanded(self) -> None:
        """Toggle the expanded state."""
        self.is_expanded = not self.is_expanded

    def expand_card(self) -> None:
        """Expand the card."""
        self.is_expanded = True

    def collapse_card(self) -> None:
        """Collapse the card."""
        self.is_expanded = False

    def copy_output(self) -> None:
        """Copy the output to clipboard (posts message for parent to handle)."""
        # Combine stdout and stderr
        content_parts = []

        if self.stdout:
            content_parts.append(f"=== {self.tool_name} Output ===")
            content_parts.append(self.stdout)

        if self.stderr:
            content_parts.append(f"=== {self.tool_name} Errors ===")
            content_parts.append(self.stderr)

        content = "\n\n".join(content_parts)

        # Post message for parent to handle clipboard
        self.post_message(self.CopyRequested(content))

        # Visual feedback
        try:
            copy_btn = self.query_one("#copy", Button)
            original_label = copy_btn.label
            copy_btn.label = "Copied!"

            # Reset label after delay
            def reset_label():
                try:
                    copy_btn.label = original_label
                except Exception:
                    pass

            self.set_timer(2.0, reset_label)

        except Exception as e:
            logger.debug(f"Failed to update copy button: {e}")

    def get_summary(self) -> str:
        """Get a one-line summary of the tool execution.

        Returns:
            Summary string
        """
        status = "✓" if self.success else "✗"
        duration = self._get_duration_text()
        output_preview = ""

        if self.stdout:
            preview = self.stdout[:50].replace("\n", " ")
            if len(self.stdout) > 50:
                preview += "..."
            output_preview = f" - {preview}"

        return f"{status} {self.tool_name} ({duration}){output_preview}"
