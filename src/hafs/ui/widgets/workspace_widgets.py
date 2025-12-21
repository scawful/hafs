"""Workspace-specific widgets for the high-performance chat interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widget import Widget
from textual.widgets import Static, Label, ListView, ListItem, Tree
from textual.reactive import reactive


try:
    from hafs.core.sessions import SessionStore, SessionMetadata
except ImportError:
    # Fallback for different path environments
    import sys
    import os

    sys.path.append(os.path.join(os.getcwd(), "src"))
    from hafs.core.sessions import SessionStore, SessionMetadata


class SessionExplorer(Widget):
    """Tree view of past and saved agentic sessions."""

    DEFAULT_CSS = """
    SessionExplorer {
        height: 1fr;
        background: $surface;
        padding: 0 1;
        border-bottom: solid $primary;
    }
    """

    def __init__(self, id: Optional[str] = None) -> None:
        super().__init__(id=id)
        self._store = SessionStore()

    def compose(self) -> ComposeResult:
        yield Label("[bold]SESSIONS[/]")
        tree: Tree[SessionMetadata | str] = Tree("History", id="session-tree")
        tree.root.expand()
        yield tree

    def on_mount(self) -> None:
        """Load sessions on mount."""
        self.refresh_sessions()

    def refresh_sessions(self) -> None:
        """Refresh the session tree."""
        try:
            tree = self.query_one("#session-tree", Tree)
            tree.root.remove_children()

            sessions = self._store.list_sessions()
            if not sessions:
                tree.root.add_leaf("[dim]No saved sessions[/]")
                return

            for session in sessions:
                tree.root.add_leaf(session.name, data=session)
        except Exception:
            pass


class ContextTree(Widget):
    """Tree view of files and resources in the current AFS context."""

    DEFAULT_CSS = """
    ContextTree {
        height: 1fr;
        background: $surface;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("[bold]CONTEXT[/]")
        tree: Tree[Path | str] = Tree(".context", id="context-tree")
        tree.root.expand()
        yield tree

    def on_mount(self) -> None:
        """Load context on mount."""
        self.refresh_context()

    def refresh_context(self) -> None:
        """Refresh the context tree."""
        from hafs.core.afs.discovery import find_context_root

        try:
            tree = self.query_one("#context-tree", Tree)
            tree.root.remove_children()

            root_path = find_context_root()
            if not root_path:
                tree.root.add_leaf("[red]No .context found[/]")
                return

            self._add_directory(tree.root, root_path)
        except Exception:
            pass

    def _add_directory(self, node: Any, path: Path) -> None:
        """Recursively add directory contents to the tree."""
        try:
            for item in sorted(path.iterdir()):
                if item.is_dir():
                    child = node.add(item.name, data=item)
                    # Don't recurse too deep for performance
                    if item.name not in (".git", "__pycache__", "node_modules"):
                        self._add_directory(child, item)
                else:
                    node.add_leaf(item.name, data=item)
        except Exception:
            pass


class AgentRoster(Widget):
    """Compact list of active agents with status indicators."""

    DEFAULT_CSS = """
    AgentRoster {
        height: auto;
        background: $surface;
        padding: 0 1;
    }
    
    .agent-item {
        height: 1;
        layout: horizontal;
    }
    
    .status-dot {
        width: 2;
    }
    
    .status-idle { color: $text-disabled; }
    .status-thinking { color: $warning; }
    .status-active { color: $success; }
    .status-error { color: $error; }
    """

    agents: reactive[list[dict[str, Any]]] = reactive(list)

    def compose(self) -> ComposeResult:
        yield Label("[bold]AGENTS[/]")
        yield Vertical(id="agent-list")

    def watch_agents(self, agents: list[dict[str, Any]]) -> None:
        """Update the agent list when reactive state changes."""
        try:
            container = self.query_one("#agent-list", Vertical)
            container.remove_children()

            if not agents:
                container.mount(Static("[dim]No agents[/]"))
                return

            for agent in agents:
                name = agent.get("name", "Unknown")
                status = agent.get("status", "idle")

                dot = "●"
                status_class = f"status-{status}"

                row = Horizontal(classes="agent-item")
                row.mount(Static(f"[{status_class}]{dot}[/] ", classes="status-dot"))
                row.mount(Static(f"{name}"))
                container.mount(row)
        except Exception:
            pass


class SharedStateInspector(Widget):
    """Key-value view of the shared agent context."""

    DEFAULT_CSS = """
    SharedStateInspector {
        height: 1fr;
        background: $surface;
        padding: 0 1;
    }
    
    .state-key {
        color: $primary;
        text-style: bold;
    }
    
    .state-value {
        color: $text;
        padding-left: 2;
        margin-bottom: 1;
    }
    """

    state_data: reactive[dict[str, Any]] = reactive(dict)

    def compose(self) -> ComposeResult:
        yield Label("[bold]SHARED STATE[/]")
        yield Vertical(id="state-container")

    def watch_state_data(self, data: dict[str, Any]) -> None:
        """Update the state view."""
        try:
            container = self.query_one("#state-container", Vertical)
            container.remove_children()

            if not data:
                container.mount(Static("[dim]Empty context[/]"))
                return

            for key, value in data.items():
                container.mount(Static(f"{key}:", classes="state-key"))
                container.mount(Static(f"{value}", classes="state-value"))
        except Exception:
            pass


class PlanTracker(Widget):
    """Checklist of the current mission plan."""

    DEFAULT_CSS = """
    PlanTracker {
        height: 1fr;
        background: $surface;
        padding: 0 1;
        border-top: solid $primary;
    }
    
    .plan-item {
        padding-left: 1;
    }
    
    .plan-done {
        color: $success;
        text-decoration: line-through;
    }
    
    .plan-pending {
        color: $text;
    }
    """

    steps: reactive[list[dict[str, Any]]] = reactive(list)

    def compose(self) -> ComposeResult:
        yield Label("[bold]MISSION PLAN[/]")
        yield Vertical(id="plan-container")

    def watch_steps(self, steps: list[dict[str, Any]]) -> None:
        """Update the plan list."""
        try:
            container = self.query_one("#plan-container", Vertical)
            container.remove_children()

            if not steps:
                container.mount(Static("[dim]No active plan[/]"))
                return

            for step in steps:
                done = step.get("done", False)
                text = step.get("text", "")
                icon = "☑" if done else "☐"
                style = "plan-done" if done else "plan-pending"

                container.mount(Static(f"{icon} {text}", classes=f"plan-item {style}"))
        except Exception:
            pass


from textual.widget import Widget
