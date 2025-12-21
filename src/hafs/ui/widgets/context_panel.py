"""Context panel widget for displaying shared transactive memory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

if TYPE_CHECKING:
    from hafs.models.agent import SharedContext


class ContextPanel(Widget):
    """Panel displaying shared context (transactive memory).

    Shows:
    - Active task
    - Current plan
    - Recent findings
    - Team decisions

    Example:
        panel = ContextPanel(id="context-panel")
        panel.update_context(shared_context)
    """

    DEFAULT_CSS = """
    ContextPanel {
        width: 30;
        height: 100%;
        background: $surface;
        border-left: solid $primary;
        padding: 1;
    }

    ContextPanel .panel-title {
        text-style: bold;
        color: $text;
        padding-bottom: 1;
    }

    ContextPanel .section-title {
        text-style: bold;
        color: $primary;
        padding-top: 1;
    }

    ContextPanel .section-content {
        color: $text;
        padding-left: 1;
    }

    ContextPanel .item {
        color: $text-disabled;
    }

    ContextPanel .empty-text {
        color: $text-disabled;
        text-style: italic;
    }

    ContextPanel .task-text {
        color: $accent;
        text-style: bold;
    }

    ContextPanel .scroll-area {
        height: 1fr;
    }
    """

    active_task: reactive[str] = reactive("")

    def __init__(
        self,
        id: str | None = None,
        classes: str | None = None,
    ):
        """Initialize context panel.

        Args:
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._shared_context_data: "SharedContext | None" = None

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        with Vertical():
            yield Static("[bold]Shared Context[/]", classes="panel-title")

            with VerticalScroll(classes="scroll-area"):
                # Active Task
                yield Static("Task", classes="section-title")
                yield Static("[dim]No active task[/]", id="task-content", classes="section-content")

                # Plan
                yield Static("Plan", classes="section-title")
                yield Static("[dim]No plan[/]", id="plan-content", classes="section-content")

                # Context items
                yield Static("Context", classes="section-title")
                yield Static(
                    "[dim]No context items[/]", id="context-items", classes="section-content"
                )

                # Findings
                yield Static("Findings", classes="section-title")
                yield Static(
                    "[dim]No findings[/]", id="findings-content", classes="section-content"
                )

                # Decisions
                yield Static("Decisions", classes="section-title")
                yield Static(
                    "[dim]No decisions[/]", id="decisions-content", classes="section-content"
                )

                # Permissions
                yield Static("Permissions", classes="section-title")
                yield Static(
                    "[dim]Policies not loaded[/]", id="policies-content", classes="section-content"
                )

    def update_context(self, context: "SharedContext") -> None:
        """Update displayed context.

        Args:
            context: The SharedContext to display.
        """
        self._shared_context_data = context
        self.active_task = context.active_task or ""

        # Update task
        task_widget = self.query_one("#task-content", Static)
        if context.active_task:
            task_widget.update(f"[bold cyan]{context.active_task}[/]")
        else:
            task_widget.update("[dim]No active task[/]")

        # Update plan
        plan_widget = self.query_one("#plan-content", Static)
        if context.plan:
            plan_text = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(context.plan[:5]))
            if len(context.plan) > 5:
                plan_text += f"\n  [dim]+{len(context.plan) - 5} more...[/]"
            plan_widget.update(plan_text)
        else:
            plan_widget.update("[dim]No plan[/]")

        # Update context items
        context_widget = self.query_one("#context-items", Static)
        if getattr(context, "context_items", None):
            lines = []
            for path in context.context_items[:6]:
                lines.append(f"  • {path}")
            if len(context.context_items) > 6:
                lines.append(f"  [dim]+{len(context.context_items) - 6} more...[/]")
            context_widget.update("\n".join(lines))
        else:
            context_widget.update("[dim]No context items[/]")

        # Update findings
        findings_widget = self.query_one("#findings-content", Static)
        if context.findings:
            findings_text = "\n".join(f"  • {f}" for f in context.findings[-5:])
            if len(context.findings) > 5:
                findings_text = (
                    f"  [dim]+{len(context.findings) - 5} older...[/]\n"
                    + findings_text
                )
            findings_widget.update(findings_text)
        else:
            findings_widget.update("[dim]No findings[/]")

        # Update decisions
        decisions_widget = self.query_one("#decisions-content", Static)
        if context.decisions:
            decisions_text = "\n".join(f"  ✓ {d}" for d in context.decisions[-3:])
            if len(context.decisions) > 3:
                decisions_text = (
                    f"  [dim]+{len(context.decisions) - 3} older...[/]\n"
                    + decisions_text
                )
            decisions_widget.update(decisions_text)
        else:
            decisions_widget.update("[dim]No decisions[/]")

    def add_finding(self, finding: str) -> None:
        """Add a finding to the display.

        Args:
            finding: Finding text.
        """
        if self._shared_context_data:
            self._shared_context_data.add_finding(finding)
            self.update_context(self._shared_context_data)

    def add_decision(self, decision: str) -> None:
        """Add a decision to the display.

        Args:
            decision: Decision text.
        """
        if self._shared_context_data:
            self._shared_context_data.add_decision(decision)
            self.update_context(self._shared_context_data)

    def set_task(self, task: str) -> None:
        """Set the active task.

        Args:
            task: Task description.
        """
        if self._shared_context_data:
            self._shared_context_data.active_task = task
            self.update_context(self._shared_context_data)

    def reset(self) -> None:
        """Reset the context display."""
        self._shared_context_data = None
        self.active_task = ""

        self.query_one("#task-content", Static).update("[dim]No active task[/]")
        self.query_one("#plan-content", Static).update("[dim]No plan[/]")
        self.query_one("#context-items", Static).update("[dim]No context items[/]")
        self.query_one("#findings-content", Static).update("[dim]No findings[/]")
        self.query_one("#decisions-content", Static).update("[dim]No decisions[/]")
        self.query_one("#policies-content", Static).update("[dim]Policies not loaded[/]")

    def update_policies(self, directory_configs) -> None:  # type: ignore[no-untyped-def]
        """Display configured AFS directory policies."""
        policies_widget = self.query_one("#policies-content", Static)
        if not directory_configs:
            policies_widget.update("[dim]No directory policies[/]")
            return

        policy_colors = {
            "read_only": "policy-readonly",
            "writable": "policy-writable",
            "executable": "policy-executable",
        }

        lines = []
        for cfg in directory_configs:
            policy_value = getattr(cfg, "policy", "read_only")
            policy_key = getattr(policy_value, "value", policy_value)
            color = policy_colors.get(str(policy_key), "text")
            lines.append(f"  [{color}]{cfg.name}[/{color}] - {policy_key.replace('_', ' ')}")

        policies_widget.update("\n".join(lines))

    @property
    def current_context(self) -> "SharedContext | None":
        """Get the current shared context."""
        return self._shared_context_data
