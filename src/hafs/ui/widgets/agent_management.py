"""Agent Management widget for the HAFS TUI."""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import DataTable, Label, TextArea, Button, Static
from textual.binding import Binding

class AgentManagementWidget(Container):
    """Widget for managing agents and their personas."""
    
    DEFAULT_CSS = """
    AgentManagementWidget {
        layout: horizontal;
        padding: 0;
        height: 100%;
    }

    AgentManagementWidget #agent-list-panel {
        width: 30;
        border-right: solid $primary-darken-2;
        height: 100%;
    }

    AgentManagementWidget #agent-edit-panel {
        width: 1fr;
        height: 100%;
        padding: 1;
    }

    AgentManagementWidget Label.section-title {
        background: $primary-darken-2;
        width: 100%;
        padding: 0 1;
        margin-bottom: 1;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self.manager = None
        self._selected_agent = None
        self._load_manager()

    def _load_manager(self):
        try:
            import sys
            # Ensure internal src is in path
            internal_src = os.path.expanduser("~/Code/Experimental/hafs_google_internal/src")
            if internal_src not in sys.path:
                sys.path.append(internal_src)
                
            from hafs_google_internal.core.agent_manager import AgentManager
            self.manager = AgentManager()
        except ImportError:
            self.manager = None

    def compose(self) -> ComposeResult:
        with Vertical(id="agent-list-panel"):
            yield Label("Agents", classes="section-title")
            yield DataTable(id="agent-table", cursor_type="row")
            
        with Vertical(id="agent-edit-panel"):
            yield Label("Edit Persona", classes="section-title", id="edit-title")
            yield TextArea(id="prompt-editor", language="markdown")
            with Horizontal(id="action-row"):
                yield Button("Save Changes", variant="primary", id="save-agent-btn")
                yield Button("Refresh", id="refresh-agents-btn")

    def on_mount(self) -> None:
        table = self.query_one("#agent-table", DataTable)
        table.add_columns("Name", "Module")
        self.refresh_agent_list()

    def refresh_agent_list(self) -> None:
        table = self.query_one("#agent-table", DataTable)
        table.clear()
        
        if not self.manager:
            table.add_row("Internal Manager Not Found", "")
            return
            
        agents = self.manager.discover_agents()
        for agent in agents:
            table.add_row(agent['name'], agent['module'], key=agent['name'])

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key.value and self.manager:
            self._selected_agent = event.row_key.value
            prompt = self.manager.get_agent_prompt(self._selected_agent)
            
            editor = self.query_one("#prompt-editor", TextArea)
            editor.load_text(prompt)
            
            self.query_one("#edit-title", Label).update(f"Edit Persona: {self._selected_agent}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-agent-btn" and self._selected_agent and self.manager:
            content = self.query_one("#prompt-editor", TextArea).text
            self.manager.set_agent_prompt(self._selected_agent, content)
            self.app.notify(f"Updated persona for {self._selected_agent}")
        elif event.button.id == "refresh-agents-btn":
            self.refresh_agent_list()
