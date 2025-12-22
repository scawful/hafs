"""Analysis Dashboard Screen - Display research analysis results.

This screen displays results from the research analysis modes:
- synergy_tom: Theory of Mind and agent synergy analysis
- scaling_metrics: Scaling patterns and performance metrics
- review_quality: Code review quality assessment
- doc_quality: Documentation quality evaluation

The screen subscribes to AnalysisEvent from the EventBus and displays
results in organized panels with trend visualization.

Usage:
    screen = AnalysisDashboardScreen()
    app.push_screen(screen)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Static

from tui.core.command_registry import Command, CommandCategory, get_command_registry
from tui.core.event_bus import AnalysisEvent, Event, get_event_bus
from tui.core.navigation_controller import get_navigation_controller
from tui.core.standard_keymaps import get_standard_keymap
from tui.core.state_store import get_state_store
from tui.mixins.which_key import WhichKeyMixin
from tui.widgets.header_bar import HeaderBar
from tui.widgets.sparkline import LabeledSparkline
from tui.widgets.which_key_bar import WhichKeyBar


class AnalysisResultPanel(Static):
    """Panel for displaying a single analysis result.

    Shows the analysis mode, summary, triggers, and score with
    trend visualization if applicable.
    """

    DEFAULT_CSS = """
    AnalysisResultPanel {
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
        margin-bottom: 1;
    }

    AnalysisResultPanel .panel-header {
        height: 1;
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }

    AnalysisResultPanel .panel-section {
        height: auto;
        margin-bottom: 1;
    }

    AnalysisResultPanel .section-label {
        color: $text-disabled;
        text-style: bold;
        height: 1;
    }

    AnalysisResultPanel .section-content {
        color: $text;
        height: auto;
        padding-left: 2;
    }

    AnalysisResultPanel .score-row {
        layout: horizontal;
        height: 1;
        margin-top: 1;
    }

    AnalysisResultPanel .score-label {
        width: 12;
        color: $text-disabled;
    }

    AnalysisResultPanel .score-value {
        width: 10;
        color: $success;
        text-style: bold;
    }

    AnalysisResultPanel .score-bar {
        width: 1fr;
        color: $accent;
    }
    """

    def __init__(
        self,
        mode: str,
        summary: Dict[str, Any],
        triggers: List[str],
        score: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Initialize the analysis result panel.

        Args:
            mode: Analysis mode name
            summary: Summary dictionary
            triggers: List of trigger messages
            score: Optional score value (0-1)
            **kwargs: Additional widget parameters
        """
        super().__init__(**kwargs)
        self._mode = mode
        self._summary = summary
        self._triggers = triggers
        self._score = score

    def compose(self) -> ComposeResult:
        """Compose the panel layout."""
        # Header with mode name
        mode_title = self._format_mode_title(self._mode)
        yield Static(mode_title, classes="panel-header")

        # Summary section
        if self._summary:
            yield Static("Summary:", classes="section-label")
            summary_text = self._format_summary(self._summary)
            yield Static(summary_text, classes="section-content")

        # Triggers section
        if self._triggers:
            yield Static("Triggers:", classes="section-label")
            triggers_text = "\n".join(f"• {t}" for t in self._triggers[:5])
            if len(self._triggers) > 5:
                triggers_text += f"\n• ... and {len(self._triggers) - 5} more"
            yield Static(triggers_text, classes="section-content")

        # Score section with visualization
        if self._score is not None:
            with Horizontal(classes="score-row"):
                yield Static("Score:", classes="score-label")
                yield Static(f"{self._score:.2%}", classes="score-value")
                yield Static(self._render_score_bar(self._score), classes="score-bar")

    def _format_mode_title(self, mode: str) -> str:
        """Format the mode name as a title.

        Args:
            mode: Mode identifier

        Returns:
            Formatted title string
        """
        mode_titles = {
            "synergy_tom": "🤝 Synergy & Theory of Mind",
            "scaling_metrics": "📈 Scaling Metrics",
            "review_quality": "🔍 Review Quality",
            "doc_quality": "📚 Documentation Quality",
        }
        return mode_titles.get(mode, mode.replace("_", " ").title())

    def _format_summary(self, summary: Dict[str, Any]) -> str:
        """Format summary dictionary as readable text.

        Args:
            summary: Summary data

        Returns:
            Formatted summary string
        """
        lines = []
        for key, value in summary.items():
            key_formatted = key.replace("_", " ").title()
            if isinstance(value, (int, float)):
                lines.append(f"{key_formatted}: {value}")
            elif isinstance(value, bool):
                lines.append(f"{key_formatted}: {'Yes' if value else 'No'}")
            elif isinstance(value, str):
                lines.append(f"{key_formatted}: {value}")
            elif isinstance(value, list):
                lines.append(f"{key_formatted}: {', '.join(str(v) for v in value[:3])}")
        return "\n".join(lines) if lines else "No summary data"

    def _render_score_bar(self, score: float) -> str:
        """Render a score bar visualization.

        Args:
            score: Score value (0-1)

        Returns:
            ASCII bar visualization
        """
        bar_width = 20
        filled = int(score * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        return bar


class AnalysisDashboardScreen(WhichKeyMixin, Screen):
    """Analysis dashboard screen for displaying research analysis results.

    Features:
    - Display results from all analysis modes
    - Subscribe to AnalysisEvent updates
    - Show trend visualizations
    - Navigate between analysis types
    - Export results

    WhichKey bindings:
    - SPC g → goto (navigation)
    - SPC r → refresh
    - SPC c → clear
    - SPC e → export
    """

    BINDINGS = [
        Binding("escape", "pop_screen", "Back"),
        Binding("r", "refresh", "Refresh"),
        Binding("c", "clear", "Clear"),
        Binding("e", "export", "Export"),
        Binding("ctrl+p", "command_palette", "Commands", show=False),
        Binding("ctrl+k", "command_palette", "Commands", show=False),
    ]

    DEFAULT_CSS = """
    AnalysisDashboardScreen {
        layout: vertical;
    }

    AnalysisDashboardScreen #main-content {
        height: 1fr;
        padding: 1 2;
    }

    AnalysisDashboardScreen .screen-title {
        height: 1;
        color: $accent;
        text-style: bold;
        background: $surface-darken-1;
        padding: 0 2;
        margin-bottom: 1;
    }

    AnalysisDashboardScreen .empty-state {
        height: 100%;
        align: center middle;
        color: $text-disabled;
    }

    AnalysisDashboardScreen #footer-area {
        height: auto;
        background: $surface;
        border-top: solid $primary;
    }

    AnalysisDashboardScreen #which-key-bar {
        width: 100%;
    }
    """

    # Reactive state
    results_count: reactive[int] = reactive(0)

    def __init__(self) -> None:
        """Initialize the analysis dashboard screen."""
        super().__init__()
        self._state = get_state_store()
        self._bus = get_event_bus()
        self._nav = get_navigation_controller()
        self._commands = get_command_registry()
        self._subscription = None

        # Analysis results storage
        self._results: Dict[str, Dict[str, Any]] = {}

        # Register commands
        self._register_commands()

    def get_which_key_map(self):
        """Return which-key bindings for this screen."""
        keymap = get_standard_keymap(self)
        # Add analysis-specific bindings
        keymap["r"] = ("refresh", self.action_refresh)
        keymap["c"] = ("clear", self.action_clear)
        keymap["e"] = ("export", self.action_export)
        return keymap

    def _register_commands(self) -> None:
        """Register screen-specific commands."""
        try:
            self._commands.register(Command(
                id="analysis.refresh",
                name="Refresh Analysis",
                description="Refresh analysis results from state",
                handler=self.action_refresh,
                category=CommandCategory.VIEW,
                keybinding="r",
            ))
        except ValueError:
            pass

        try:
            self._commands.register(Command(
                id="analysis.clear",
                name="Clear Analysis",
                description="Clear all analysis results",
                handler=self.action_clear,
                category=CommandCategory.VIEW,
                keybinding="c",
            ))
        except ValueError:
            pass

        try:
            self._commands.register(Command(
                id="analysis.export",
                name="Export Analysis",
                description="Export analysis results to file",
                handler=self.action_export,
                category=CommandCategory.ANALYSIS,
                keybinding="e",
            ))
        except ValueError:
            pass

    def compose(self) -> ComposeResult:
        """Compose the screen layout."""
        yield HeaderBar(id="header-bar")

        yield Static("📊 Analysis Dashboard", classes="screen-title")

        with VerticalScroll(id="main-content"):
            yield Container(id="results-container")

        # Footer area
        with Container(id="footer-area"):
            yield WhichKeyBar(id="which-key-bar")

    def on_mount(self) -> None:
        """Initialize screen on mount."""
        # Set navigation context
        self._nav.set_screen_context("analysis")

        # Subscribe to analysis events
        self._subscription = self._bus.subscribe(
            "analysis.*",
            self._on_analysis_event,
        )

        # Initialize which-key hints
        self.init_which_key_hints()

        # Set breadcrumb path
        try:
            header = self.query_one(HeaderBar)
            header.set_path("/analysis")
        except Exception:
            pass

        # Load existing results from state
        self._load_state()
        self._render_results()

    def on_unmount(self) -> None:
        """Clean up on unmount."""
        if self._subscription:
            self._subscription.unsubscribe()

    def _load_state(self) -> None:
        """Load analysis results from state store."""
        analysis_state = self._state.get("analysis", {})

        for mode in ["synergy_tom", "scaling_metrics", "review_quality", "doc_quality"]:
            mode_data = analysis_state.get(mode, {})
            if mode_data:
                self._results[mode] = mode_data

        self.results_count = len(self._results)

    def _on_analysis_event(self, event: Event) -> None:
        """Handle analysis events.

        Args:
            event: The analysis event
        """
        if not isinstance(event, AnalysisEvent):
            return

        mode = event.mode
        if not mode:
            return

        # Store result
        self._results[mode] = {
            "summary": event.summary,
            "triggers": event.triggers,
            "score": event.score,
            "timestamp": event.timestamp,
        }

        # Update state
        self._state.set(f"analysis.{mode}", self._results[mode])
        self.results_count = len(self._results)

        # Re-render
        self._render_results()

    def _render_results(self) -> None:
        """Render all analysis results."""
        try:
            container = self.query_one("#results-container", Container)
            container.remove_children()

            if not self._results:
                container.mount(
                    Static(
                        "No analysis results available.\nResults will appear here when analysis events are published.",
                        classes="empty-state",
                    )
                )
                return

            # Render results in order
            mode_order = ["synergy_tom", "scaling_metrics", "review_quality", "doc_quality"]

            for mode in mode_order:
                if mode in self._results:
                    result = self._results[mode]
                    panel = AnalysisResultPanel(
                        mode=mode,
                        summary=result.get("summary", {}),
                        triggers=result.get("triggers", []),
                        score=result.get("score"),
                    )
                    container.mount(panel)

        except Exception as e:
            self.notify(f"Error rendering results: {e}", severity="error")

    # Actions

    def action_refresh(self) -> None:
        """Refresh analysis results from state."""
        self._load_state()
        self._render_results()
        self.notify("Analysis results refreshed", timeout=1)

    def action_command_palette(self) -> None:
        """Open command palette."""
        from tui.screens.command_palette import CommandPalette
        self.app.push_screen(CommandPalette())

    def action_clear(self) -> None:
        """Clear all analysis results."""
        self._results.clear()
        self.results_count = 0

        # Clear from state
        for mode in ["synergy_tom", "scaling_metrics", "review_quality", "doc_quality"]:
            self._state.set(f"analysis.{mode}", {})

        self._render_results()
        self.notify("Analysis results cleared", timeout=1)

    def action_export(self) -> None:
        """Export analysis results to file."""
        import json
        from datetime import datetime
        from pathlib import Path

        try:
            # Create exports directory
            export_dir = Path.home() / ".context" / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_file = export_dir / f"analysis_results_{timestamp}.json"

            # Export results
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "results": self._results,
            }

            with open(export_file, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            self.notify(f"Exported to {export_file.name}", timeout=3)

        except Exception as e:
            self.notify(f"Export failed: {e}", severity="error")

    def action_pop_screen(self) -> None:
        """Return to previous screen."""
        self.app.pop_screen()

    async def on_header_bar_navigation_requested(self, event: HeaderBar.NavigationRequested) -> None:
        """Handle header bar navigation requests."""
        from tui.core.screen_router import get_screen_router

        route_map = {
            "dashboard": "/dashboard",
            "chat": "/chat",
            "logs": "/logs",
            "services": "/services",
            "analysis": "/analysis",
            "config": "/config",
        }
        route = route_map.get(event.screen)
        if route:
            router = get_screen_router()
            await router.navigate(route)
