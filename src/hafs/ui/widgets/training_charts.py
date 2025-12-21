"""Chart widgets for training data visualization using plotext."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotext as plt
from textual.reactive import reactive
from textual.widgets import Static

if TYPE_CHECKING:
    from hafs.ui.widgets.training_data_loader import (
        EmbeddingRegionData,
        GeneratorStatsData,
        QualityTrendData,
        TrainingRunData,
    )


class PlotextChart(Static):
    """Base class for plotext-based charts."""

    DEFAULT_CSS = """
    PlotextChart {
        height: auto;
        min-height: 12;
        padding: 1;
        border: solid $primary;
    }
    """

    chart_width: reactive[int] = reactive(80)
    chart_height: reactive[int] = reactive(15)

    def _build_chart(self) -> str:
        """Override in subclasses to build the chart."""
        return "No data"

    def render(self) -> str:
        """Render the chart to ASCII."""
        return self._build_chart()

    def on_resize(self, event) -> None:
        """Update dimensions on resize."""
        self.chart_width = max(40, event.size.width - 4)  # Account for padding/border
        self.chart_height = max(10, event.size.height - 2)


class QualityTrendChart(PlotextChart):
    """Line chart showing quality metrics over time."""

    data: reactive[list["QualityTrendData"]] = reactive([], always_update=True)

    def _build_chart(self) -> str:
        if not self.data:
            return "No quality trend data available"

        plt.clear_data()
        plt.clear_figure()
        plt.plotsize(self.chart_width, self.chart_height)
        plt.theme("dark")

        # Plot each metric
        colors = ["cyan", "green", "yellow", "red", "magenta"]
        for i, trend in enumerate(self.data[:5]):  # Max 5 series
            if trend.values:
                label = f"{trend.domain}/{trend.metric}"
                plt.plot(
                    trend.values[-50:], label=label, color=colors[i % len(colors)]
                )

        plt.title("Quality Metrics Trend")
        plt.xlabel("Recent Samples")
        plt.ylabel("Score")

        return plt.build()


class GeneratorStatsChart(PlotextChart):
    """Bar chart showing generator acceptance rates."""

    data: reactive[list["GeneratorStatsData"]] = reactive([], always_update=True)

    def _build_chart(self) -> str:
        if not self.data:
            return "No generator stats available"

        plt.clear_data()
        plt.clear_figure()
        plt.plotsize(self.chart_width, self.chart_height)
        plt.theme("dark")

        # Prepare data
        names = [g.name.replace("DataGenerator", "") for g in self.data]
        rates = [g.acceptance_rate * 100 for g in self.data]

        plt.bar(names, rates, color="cyan")
        plt.title("Generator Acceptance Rates")
        plt.ylabel("Acceptance %")
        plt.ylim(0, 100)

        return plt.build()


class EmbeddingCoverageChart(PlotextChart):
    """Scatter-like visualization of embedding region coverage."""

    data: reactive[list["EmbeddingRegionData"]] = reactive([], always_update=True)
    coverage_score: reactive[float] = reactive(0.0)

    def _build_chart(self) -> str:
        if not self.data:
            return "No embedding coverage data available"

        plt.clear_data()
        plt.clear_figure()
        plt.plotsize(self.chart_width, self.chart_height)
        plt.theme("dark")

        # Plot region sample counts
        counts = [r.sample_count for r in self.data]
        avg_count = sum(counts) / len(counts) if counts else 0

        # Split into sparse and dense
        sparse_x = [
            i for i, r in enumerate(self.data) if r.sample_count < avg_count * 0.5
        ]
        sparse_y = [self.data[i].sample_count for i in sparse_x]
        dense_x = [
            i for i, r in enumerate(self.data) if r.sample_count >= avg_count * 0.5
        ]
        dense_y = [self.data[i].sample_count for i in dense_x]

        if dense_x:
            plt.scatter(dense_x, dense_y, label="Dense", color="green", marker="dot")
        if sparse_x:
            plt.scatter(
                sparse_x, sparse_y, label="Sparse", color="red", marker="diamond"
            )

        plt.title(f"Embedding Coverage (Score: {self.coverage_score:.1%})")
        plt.xlabel("Region Index")
        plt.ylabel("Sample Count")

        return plt.build()


class TrainingLossChart(PlotextChart):
    """Bar chart comparing training run losses."""

    data: reactive[list["TrainingRunData"]] = reactive([], always_update=True)

    def _build_chart(self) -> str:
        if not self.data:
            return "No training run data available"

        plt.clear_data()
        plt.clear_figure()
        plt.plotsize(self.chart_width, self.chart_height)
        plt.theme("dark")

        # Sort by final loss
        sorted_runs = sorted(self.data, key=lambda r: r.final_loss)[:10]

        names = [r.run_id[:15] for r in sorted_runs]
        losses = [r.final_loss for r in sorted_runs]

        plt.bar(names, losses, color="cyan")
        plt.title("Training Run Losses (Lower is Better)")
        plt.ylabel("Final Loss")

        return plt.build()


class RejectionReasonsChart(PlotextChart):
    """Horizontal bar chart of rejection reasons."""

    data: reactive[dict[str, int]] = reactive({}, always_update=True)

    def _build_chart(self) -> str:
        if not self.data:
            return "No rejection data available"

        plt.clear_data()
        plt.clear_figure()
        plt.plotsize(self.chart_width, self.chart_height)
        plt.theme("dark")

        # Sort by count
        sorted_reasons = sorted(self.data.items(), key=lambda x: x[1], reverse=True)[:8]

        reasons = [r[0].replace("_", " ").title() for r in sorted_reasons]
        counts = [r[1] for r in sorted_reasons]

        plt.bar(reasons, counts, color="yellow", orientation="horizontal")
        plt.title("Top Rejection Reasons")

        return plt.build()
