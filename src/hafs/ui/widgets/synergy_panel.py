"""Synergy panel widget for displaying Theory of Mind metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import ProgressBar, Static

if TYPE_CHECKING:
    from hafs.models.synergy import SynergyScore


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
        height: 7;
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
        width: 15;
        height: 100%;
        padding-right: 2;
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
        width: 20;
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
