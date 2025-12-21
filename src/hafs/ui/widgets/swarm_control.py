"""Swarm Control widget for the HAFS TUI."""

import os
import subprocess
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Label, Input, Static, Log
from textual.binding import Binding

class SwarmControlWidget(Container):
    """Widget for controlling background swarms and maintenance."""
    
    DEFAULT_CSS = """
    SwarmControlWidget {
        layout: horizontal;
        padding: 1;
        height: 100%;
    }

    SwarmControlWidget #controls-panel {
        width: 40;
        border-right: solid $primary;
        padding-right: 1;
    }

    SwarmControlWidget #status-log {
        width: 1fr;
        height: 100%;
    }

    SwarmControlWidget .action-btn {
        margin-bottom: 1;
        width: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="controls-panel"):
            yield Label("[bold]Swarm Actions[/]")
            yield Input(placeholder="Topic (optional)", id="swarm-topic-input")
            yield Button("🚀 Launch Swarm", variant="primary", id="btn-swarm", classes="action-btn")
            separator = Static("---")
            separator.styles.margin = (1, 0)
            yield separator
            yield Label("[bold]Maintenance[/]")
            yield Button("🌱 Run Gardener", id="btn-garden", classes="action-btn")
            yield Button("🧹 Clean Logs", id="btn-clean", classes="action-btn")
            yield Button("☕ Daily Briefing", id="btn-brief", classes="action-btn")

        with Vertical(id="status-log"):
            yield Label("[bold]Activity Feed[/]")
            yield Log(id="swarm-activity-log")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        log = self.query_one("#swarm-activity-log", Log)
        
        if event.button.id == "btn-swarm":
            topic = self.query_one("#swarm-topic-input", Input).value or "Auto-Discovery"
            log.write(f"Launching swarm on topic: {topic}...")
            # Use shell helper
            subprocess.Popen(["bash", "-c", f"source ~/.zshrc; hafs-swarm '{topic}' > /dev/null 2>&1"])
            self.app.notify("Swarm Launched in background")
            
        elif event.button.id == "btn-garden":
            log.write("Triggering Context Gardener...")
            subprocess.Popen(["bash", "-c", "source ~/.zshrc; python3 -c 'import asyncio; from hafs_google_internal.agents.gardener import ContextGardener; asyncio.run(ContextGardener().run_task())' > /dev/null 2>&1"])
            self.app.notify("Gardener started")
            
        elif event.button.id == "btn-brief":
            log.write("Generating Daily Briefing...")
            # We can't easily wait for output here without blocking, so just launch
            subprocess.Popen(["bash", "-c", "source ~/.zshrc; hafs-brief > /dev/null 2>&1"])
            self.app.notify("Daily Briefing triggered")
            
        elif event.button.id == "btn-clean":
            log.write("Cleaning reports and logs...")
            subprocess.Popen(["bash", "-c", "source ~/.zshrc; hafs-clean > /dev/null 2>&1"])
            self.app.notify("Cleanup initiated")
