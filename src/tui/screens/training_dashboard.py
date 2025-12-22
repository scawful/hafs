"""Training Data Dashboard Screen."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import Label, Static

from tui.mixins.which_key import WhichKeyMixin
from tui.widgets.header_bar import HeaderBar
from tui.widgets.training_charts import (
    EmbeddingCoverageChart,
    GeneratorStatsChart,
    QualityTrendChart,
    RejectionReasonsChart,
    TrainingLossChart,
)
from tui.widgets.training_data_loader import TrainingDataLoader
from tui.widgets.which_key_bar import WhichKeyBar


class TrainingDashboardScreen(WhichKeyMixin, Screen):
    """Dashboard for training data visualization."""

    BINDINGS = [
        Binding("r", "refresh", "Refresh", show=True),
        Binding("q", "back", "Back", show=True),
        Binding("escape", "back", "Back", show=False),
        Binding("1", "focus_quality", "Quality"),
        Binding("2", "focus_generators", "Generators"),
        Binding("3", "focus_coverage", "Coverage"),
        Binding("4", "focus_runs", "Runs"),
    ]

    DEFAULT_CSS = """
    TrainingDashboardScreen {
        layout: vertical;
    }

    .chart-container {
        layout: grid;
        grid-size: 2 2;
        grid-gutter: 1;
        padding: 1;
    }

    .chart-panel {
        border: solid $primary;
        padding: 0 1;
    }

    .chart-panel:focus {
        border: double $accent;
    }

    .panel-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    .status-bar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-disabled;
        padding: 0 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.loader = TrainingDataLoader()
        self._quality_chart: QualityTrendChart | None = None
        self._generator_chart: GeneratorStatsChart | None = None
        self._coverage_chart: EmbeddingCoverageChart | None = None
        self._loss_chart: TrainingLossChart | None = None

    def get_which_key_map(self) -> dict:
        """Return which-key bindings for this screen."""
        return {
            "r": ("Refresh", "refresh"),
            "q": ("Back", "back"),
            "1": ("Quality", "focus_quality"),
            "2": ("Generators", "focus_generators"),
            "3": ("Coverage", "focus_coverage"),
            "4": ("Runs", "focus_runs"),
        }

    def compose(self) -> ComposeResult:
        yield HeaderBar("Training Data Dashboard")

        with Container(classes="chart-container"):
            with Vertical(classes="chart-panel", id="quality-panel"):
                yield Label("Quality Trends", classes="panel-title")
                self._quality_chart = QualityTrendChart()
                yield self._quality_chart

            with Vertical(classes="chart-panel", id="generator-panel"):
                yield Label("Generator Stats", classes="panel-title")
                self._generator_chart = GeneratorStatsChart()
                yield self._generator_chart

            with Vertical(classes="chart-panel", id="coverage-panel"):
                yield Label("Embedding Coverage", classes="panel-title")
                self._coverage_chart = EmbeddingCoverageChart()
                yield self._coverage_chart

            with Vertical(classes="chart-panel", id="runs-panel"):
                yield Label("Training Runs", classes="panel-title")
                self._loss_chart = TrainingLossChart()
                yield self._loss_chart

        yield Static("", classes="status-bar", id="status")
        yield WhichKeyBar()

    async def on_mount(self) -> None:
        """Load data when screen mounts."""
        self.init_which_key_hints()
        await self.action_refresh()

    async def action_refresh(self) -> None:
        """Reload data from disk."""
        status = self.query_one("#status", Static)
        status.update("Loading training data...")

        if self.loader.refresh():
            # Update charts with new data
            if self._quality_chart:
                self._quality_chart.data = self.loader.quality_trends
            if self._generator_chart:
                self._generator_chart.data = self.loader.generator_stats
            if self._coverage_chart:
                self._coverage_chart.data = self.loader.embedding_regions
                self._coverage_chart.coverage_score = self.loader.coverage_score
            if self._loss_chart:
                self._loss_chart.data = self.loader.training_runs

            status.update(
                f"Loaded: {len(self.loader.generator_stats)} generators, "
                f"{len(self.loader.embedding_regions)} regions, "
                f"{len(self.loader.training_runs)} runs"
            )
        else:
            status.update("No training data found in ~/.context/training/")

    async def action_back(self) -> None:
        """Go back to previous screen."""
        self.app.pop_screen()

    async def action_focus_quality(self) -> None:
        self.query_one("#quality-panel").focus()

    async def action_focus_generators(self) -> None:
        self.query_one("#generator-panel").focus()

    async def action_focus_coverage(self) -> None:
        self.query_one("#coverage-panel").focus()

    async def action_focus_runs(self) -> None:
        self.query_one("#runs-panel").focus()

    def on_header_bar_navigation_requested(
        self, event: HeaderBar.NavigationRequested
    ) -> None:
        """Handle header bar navigation requests."""
        if event.target == "back":
            self.app.pop_screen()
