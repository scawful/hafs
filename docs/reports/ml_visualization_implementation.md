# ML/AI Training Data Visualization - Implementation Guide

> Generated: 2025-12-21
> Status: Planned
> Components: Textual TUI + ImGui/ImPlot C++ Application

This document provides detailed implementation guidance for adding ML/AI training data visualization to hafs.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Sources](#data-sources)
3. [Part 1: Textual TUI Enhancement](#part-1-textual-tui-enhancement)
4. [Part 2: ImGui/ImPlot C++ Application](#part-2-imguiimplot-c-application)
5. [Implementation Checklist](#implementation-checklist)

---

## Overview

### Goals

- Visualize training data quality metrics in real-time
- Track generator performance and rejection patterns
- Monitor embedding space coverage for active learning
- Compare training runs and model improvements

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Data Sources                                  │
│  ~/.context/training/                                                │
│  ├── quality_feedback.json   (Quality trends, generator stats)      │
│  ├── training_feedback.json  (Training runs, domain effectiveness)  │
│  └── active_learning.json    (Embedding regions, coverage)          │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
            ┌───────▼───────┐          ┌────────▼────────┐
            │  Textual TUI  │          │  ImGui/ImPlot   │
            │  (ASCII)      │          │  (Native GUI)   │
            │               │          │                 │
            │ • plotext     │          │ • GLFW window   │
            │ • Quick view  │          │ • Interactive   │
            │ • Terminal    │          │ • Export PNG    │
            └───────────────┘          └─────────────────┘
```

---

## Data Sources

### 1. Quality Feedback (`~/.context/training/quality_feedback.json`)

**Source:** `src/agents/training/feedback/quality_tracker.py`

```json
{
  "thresholds": {
    "min_quality_score": 0.7,
    "min_diversity": 0.3,
    "max_hallucination_risk": 0.5,
    "min_kg_consistency": 0.5,
    "min_coherence": 0.4,
    "similarity_threshold": 0.95
  },
  "generator_stats": {
    "AsmDataGenerator": {
      "samples_generated": 500,
      "samples_accepted": 425,
      "samples_rejected": 75,
      "rejection_reasons": {
        "low_diversity": 25,
        "kg_inconsistent": 15,
        "high_hallucination": 20,
        "low_coherence": 10,
        "duplicate": 5
      },
      "avg_quality_score": 0.82,
      "total_quality_sum": 348.5
    },
    "CppDataGenerator": { ... },
    "TextDataGenerator": { ... }
  },
  "rejection_history": [
    {
      "sample_id": "01JFXYZ...",
      "domain": "asm",
      "generator": "AsmDataGenerator",
      "reason": "low_diversity",
      "scores": {
        "diversity": 0.25,
        "kg_consistency": 0.8,
        "hallucination_risk": 0.3,
        "coherence": 0.7,
        "overall": 0.58
      },
      "timestamp": "2025-12-21T10:30:00Z"
    }
  ],
  "last_updated": "2025-12-21T12:00:00Z"
}
```

**Visualization Use:**
- Line chart: Quality score trends over time
- Bar chart: Generator acceptance rates
- Pie chart: Rejection reason distribution
- Heatmap: Generator × Rejection reason matrix

### 2. Training Feedback (`~/.context/training/training_feedback.json`)

**Source:** `src/agents/training/feedback/training_feedback.py`

```json
{
  "training_runs": {
    "run_2025_12_20_qwen": {
      "model_name": "qwen2.5-coder-finetuned",
      "base_model": "qwen2.5-coder:7b",
      "dataset_path": "~/.context/training/asm/output/dataset.jsonl",
      "samples_count": 1500,
      "domain_distribution": {
        "asm": 800,
        "cpp": 400,
        "text": 300
      },
      "start_time": "2025-12-20T14:00:00Z",
      "end_time": "2025-12-20T18:30:00Z",
      "final_loss": 0.089,
      "eval_metrics": {
        "accuracy": 0.92,
        "f1_score": 0.88,
        "perplexity": 3.2
      },
      "notes": "First ASM-heavy run with new validators"
    }
  },
  "domain_effectiveness": {
    "asm": 0.025,
    "cpp": 0.018,
    "text": 0.012
  },
  "quality_threshold_effectiveness": {
    "0.7": 0.015,
    "0.8": 0.022,
    "0.9": 0.008
  },
  "last_updated": "2025-12-21T12:00:00Z"
}
```

**Visualization Use:**
- Line chart: Loss curves across training runs
- Bar chart: Domain effectiveness comparison
- Grouped bars: Run comparison (loss, metrics)
- Area chart: Domain distribution over runs

### 3. Active Learning (`~/.context/training/active_learning.json`)

**Source:** `src/agents/training/active_learning.py`

```json
{
  "regions": [
    {
      "centroid": [0.123, -0.456, 0.789, ...],  // 768 dims
      "sample_count": 45,
      "sample_ids": ["01JFX...", "01JFY...", ...],
      "domain": "asm",
      "avg_quality": 0.78
    },
    {
      "centroid": [...],
      "sample_count": 12,  // Sparse region!
      "sample_ids": [...],
      "domain": "cpp",
      "avg_quality": 0.65
    }
  ],
  "embedding_dim": 768,
  "num_regions": 50,
  "last_updated": "2025-12-21T12:00:00Z"
}
```

**Derived Metrics (from `get_coverage_report()`):**
```python
CoverageReport:
  total_samples: 2500
  num_regions: 50
  avg_region_density: 50.0
  min_region_density: 12
  max_region_density: 95
  sparse_regions: 15  # Below average
  coverage_score: 0.72  # 0-1, higher = more even
  domain_coverage: {
    "asm": 0.45,
    "cpp": 0.30,
    "text": 0.25
  }
```

**Visualization Use:**
- Scatter plot: Regions by density (highlight sparse)
- Bubble chart: Region size by sample count, color by domain
- Pie chart: Domain coverage distribution
- Progress bar: Overall coverage score

---

## Part 1: Textual TUI Enhancement

### Dependencies

Add to `pyproject.toml`:
```toml
dependencies = [
    # ... existing ...
    "plotext>=5.2.8",
]
```

### File: `src/hafs/ui/widgets/training_data_loader.py`

```python
"""Data loading layer for training visualization."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class QualityTrendData:
    """Quality trend for a single domain/metric."""
    domain: str
    metric: str
    values: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    @property
    def trend_direction(self) -> str:
        if len(self.values) < 5:
            return "insufficient"
        recent = sum(self.values[-5:]) / 5
        older = sum(self.values[:5]) / 5
        diff = recent - older
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        return "stable"


@dataclass
class GeneratorStatsData:
    """Statistics for a single generator."""
    name: str
    samples_generated: int = 0
    samples_accepted: int = 0
    samples_rejected: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    avg_quality: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        total = self.samples_accepted + self.samples_rejected
        return self.samples_accepted / total if total > 0 else 0.0


@dataclass
class EmbeddingRegionData:
    """Embedding space region data."""
    index: int
    sample_count: int
    domain: str
    avg_quality: float


@dataclass
class TrainingRunData:
    """Training run metadata."""
    run_id: str
    model_name: str
    final_loss: float
    samples_count: int
    domain_distribution: dict[str, int] = field(default_factory=dict)
    eval_metrics: dict[str, float] = field(default_factory=dict)


class TrainingDataLoader:
    """Loads and caches training data from JSON files."""

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or Path.home() / ".context" / "training"

        # Cached data
        self._quality_trends: list[QualityTrendData] = []
        self._generator_stats: list[GeneratorStatsData] = []
        self._embedding_regions: list[EmbeddingRegionData] = []
        self._training_runs: list[TrainingRunData] = []
        self._coverage_score: float = 0.0

        self._last_load: Optional[datetime] = None

    def refresh(self) -> bool:
        """Reload all data from disk."""
        try:
            self._load_quality_feedback()
            self._load_active_learning()
            self._load_training_feedback()
            self._last_load = datetime.now()
            return True
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return False

    def _load_quality_feedback(self) -> None:
        """Load quality_feedback.json."""
        path = self.data_path / "quality_feedback.json"
        if not path.exists():
            return

        data = json.loads(path.read_text())

        # Parse generator stats
        self._generator_stats = []
        for name, stats in data.get("generator_stats", {}).items():
            self._generator_stats.append(GeneratorStatsData(
                name=name,
                samples_generated=stats.get("samples_generated", 0),
                samples_accepted=stats.get("samples_accepted", 0),
                samples_rejected=stats.get("samples_rejected", 0),
                rejection_reasons=stats.get("rejection_reasons", {}),
                avg_quality=stats.get("avg_quality_score", 0.0),
            ))

        # Parse rejection history for trends
        history = data.get("rejection_history", [])
        trends_by_key: dict[tuple[str, str], QualityTrendData] = {}

        for entry in history:
            domain = entry.get("domain", "unknown")
            scores = entry.get("scores", {})

            for metric, value in scores.items():
                key = (domain, metric)
                if key not in trends_by_key:
                    trends_by_key[key] = QualityTrendData(domain=domain, metric=metric)
                trends_by_key[key].values.append(value)

        self._quality_trends = list(trends_by_key.values())

    def _load_active_learning(self) -> None:
        """Load active_learning.json."""
        path = self.data_path / "active_learning.json"
        if not path.exists():
            return

        data = json.loads(path.read_text())

        self._embedding_regions = []
        for i, region in enumerate(data.get("regions", [])):
            self._embedding_regions.append(EmbeddingRegionData(
                index=i,
                sample_count=region.get("sample_count", 0),
                domain=region.get("domain", "unknown"),
                avg_quality=region.get("avg_quality", 0.0),
            ))

        # Calculate coverage score
        if self._embedding_regions:
            counts = [r.sample_count for r in self._embedding_regions]
            avg = sum(counts) / len(counts)
            if avg > 0:
                std = (sum((c - avg) ** 2 for c in counts) / len(counts)) ** 0.5
                cv = std / avg
                self._coverage_score = max(0.0, min(1.0, 1.0 - cv))

    def _load_training_feedback(self) -> None:
        """Load training_feedback.json."""
        path = self.data_path / "training_feedback.json"
        if not path.exists():
            return

        data = json.loads(path.read_text())

        self._training_runs = []
        for run_id, run_data in data.get("training_runs", {}).items():
            if run_data.get("final_loss") is not None:
                self._training_runs.append(TrainingRunData(
                    run_id=run_id,
                    model_name=run_data.get("model_name", ""),
                    final_loss=run_data.get("final_loss", 0.0),
                    samples_count=run_data.get("samples_count", 0),
                    domain_distribution=run_data.get("domain_distribution", {}),
                    eval_metrics=run_data.get("eval_metrics", {}),
                ))

    # Properties for accessing data
    @property
    def quality_trends(self) -> list[QualityTrendData]:
        return self._quality_trends

    @property
    def generator_stats(self) -> list[GeneratorStatsData]:
        return self._generator_stats

    @property
    def embedding_regions(self) -> list[EmbeddingRegionData]:
        return self._embedding_regions

    @property
    def training_runs(self) -> list[TrainingRunData]:
        return self._training_runs

    @property
    def coverage_score(self) -> float:
        return self._coverage_score

    @property
    def has_data(self) -> bool:
        return bool(self._generator_stats or self._embedding_regions or self._training_runs)
```

### File: `src/hafs/ui/widgets/training_charts.py`

```python
"""Chart widgets for training data visualization using plotext."""

from __future__ import annotations

from typing import Any

import plotext as plt
from textual.reactive import reactive
from textual.widgets import Static

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

    width: reactive[int] = reactive(80)
    height: reactive[int] = reactive(15)

    def _build_chart(self) -> str:
        """Override in subclasses to build the chart."""
        return "No data"

    def render(self) -> str:
        """Render the chart to ASCII."""
        return self._build_chart()

    def on_resize(self, event) -> None:
        """Update dimensions on resize."""
        self.width = event.size.width - 4  # Account for padding/border
        self.height = max(10, event.size.height - 2)


class QualityTrendChart(PlotextChart):
    """Line chart showing quality metrics over time."""

    data: reactive[list[QualityTrendData]] = reactive([], always_update=True)

    def _build_chart(self) -> str:
        if not self.data:
            return "No quality trend data available"

        plt.clear_data()
        plt.clear_figure()
        plt.plotsize(self.width, self.height)
        plt.theme("dark")

        # Plot each metric
        colors = ["cyan", "green", "yellow", "red", "magenta"]
        for i, trend in enumerate(self.data[:5]):  # Max 5 series
            if trend.values:
                label = f"{trend.domain}/{trend.metric}"
                plt.plot(trend.values[-50:], label=label, color=colors[i % len(colors)])

        plt.title("Quality Metrics Trend")
        plt.xlabel("Recent Samples")
        plt.ylabel("Score")

        return plt.build()


class GeneratorStatsChart(PlotextChart):
    """Bar chart showing generator acceptance rates."""

    data: reactive[list[GeneratorStatsData]] = reactive([], always_update=True)

    def _build_chart(self) -> str:
        if not self.data:
            return "No generator stats available"

        plt.clear_data()
        plt.clear_figure()
        plt.plotsize(self.width, self.height)
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

    data: reactive[list[EmbeddingRegionData]] = reactive([], always_update=True)
    coverage_score: reactive[float] = reactive(0.0)

    def _build_chart(self) -> str:
        if not self.data:
            return "No embedding coverage data available"

        plt.clear_data()
        plt.clear_figure()
        plt.plotsize(self.width, self.height)
        plt.theme("dark")

        # Plot region sample counts
        indices = [r.index for r in self.data]
        counts = [r.sample_count for r in self.data]

        avg_count = sum(counts) / len(counts) if counts else 0

        # Split into sparse and dense
        sparse_x = [i for i, r in enumerate(self.data) if r.sample_count < avg_count * 0.5]
        sparse_y = [self.data[i].sample_count for i in sparse_x]
        dense_x = [i for i, r in enumerate(self.data) if r.sample_count >= avg_count * 0.5]
        dense_y = [self.data[i].sample_count for i in dense_x]

        if dense_x:
            plt.scatter(dense_x, dense_y, label="Dense", color="green", marker="dot")
        if sparse_x:
            plt.scatter(sparse_x, sparse_y, label="Sparse", color="red", marker="diamond")

        plt.title(f"Embedding Coverage (Score: {self.coverage_score:.1%})")
        plt.xlabel("Region Index")
        plt.ylabel("Sample Count")

        return plt.build()


class TrainingLossChart(PlotextChart):
    """Bar chart comparing training run losses."""

    data: reactive[list[TrainingRunData]] = reactive([], always_update=True)

    def _build_chart(self) -> str:
        if not self.data:
            return "No training run data available"

        plt.clear_data()
        plt.clear_figure()
        plt.plotsize(self.width, self.height)
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
        plt.plotsize(self.width, self.height)
        plt.theme("dark")

        # Sort by count
        sorted_reasons = sorted(self.data.items(), key=lambda x: x[1], reverse=True)[:8]

        reasons = [r[0].replace("_", " ").title() for r in sorted_reasons]
        counts = [r[1] for r in sorted_reasons]

        plt.bar(reasons, counts, color="yellow", orientation="horizontal")
        plt.title("Top Rejection Reasons")

        return plt.build()
```

### File: `src/hafs/ui/screens/training_dashboard.py`

```python
"""Training Data Dashboard Screen."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Label, Static

from hafs.ui.mixins.which_key import WhichKeyMixin
from hafs.ui.widgets.header_bar import HeaderBar
from hafs.ui.widgets.which_key_bar import WhichKeyBar
from hafs.ui.widgets.training_charts import (
    EmbeddingCoverageChart,
    GeneratorStatsChart,
    QualityTrendChart,
    RejectionReasonsChart,
    TrainingLossChart,
)
from hafs.ui.widgets.training_data_loader import TrainingDataLoader


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

    .status-bar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
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

            status.update(f"Loaded: {len(self.loader.generator_stats)} generators, "
                         f"{len(self.loader.embedding_regions)} regions, "
                         f"{len(self.loader.training_runs)} runs")
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
```

---

## Part 2: ImGui/ImPlot C++ Application

### Directory Structure

```
src/cc/viz/
├── CMakeLists.txt           # Viz-specific CMake config
├── main.cc                  # Entry point
├── app.h                    # Application class header
├── app.cc                   # Application implementation
├── data_loader.h            # JSON data loader header
├── data_loader.cc           # JSON data loader implementation
├── charts/
│   ├── charts.h             # Common chart headers
│   ├── quality_chart.cc     # Quality trends line chart
│   ├── generator_chart.cc   # Generator stats bar chart
│   ├── embedding_chart.cc   # Embedding coverage scatter
│   └── training_chart.cc    # Training loss comparison
└── themes/
    └── hafs_theme.h         # Custom ImGui theme
```

### CMake Addition to `src/cc/CMakeLists.txt`

```cmake
# =============================================================================
# Optional: ImGui Visualization Application
# =============================================================================

option(HAFS_BUILD_VIZ "Build ImGui visualization application" OFF)

if(HAFS_BUILD_VIZ)
  message(STATUS "Building visualization application with ImGui/ImPlot")

  # Fetch Dear ImGui
  FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.90.1
  )
  FetchContent_MakeAvailable(imgui)

  # Fetch ImPlot
  FetchContent_Declare(
    implot
    GIT_REPOSITORY https://github.com/epezent/implot.git
    GIT_TAG v0.16
  )
  FetchContent_MakeAvailable(implot)

  # Find dependencies
  find_package(glfw3 3.3 REQUIRED)
  find_package(OpenGL REQUIRED)

  # Collect ImGui sources
  set(IMGUI_SOURCES
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
  )

  # Collect ImPlot sources
  set(IMPLOT_SOURCES
    ${implot_SOURCE_DIR}/implot.cpp
    ${implot_SOURCE_DIR}/implot_items.cpp
    ${implot_SOURCE_DIR}/implot_demo.cpp
  )

  # Visualization application
  add_executable(hafs_viz
    viz/main.cc
    viz/app.cc
    viz/data_loader.cc
    viz/charts/quality_chart.cc
    viz/charts/generator_chart.cc
    viz/charts/embedding_chart.cc
    viz/charts/training_chart.cc
    ${IMGUI_SOURCES}
    ${IMPLOT_SOURCES}
  )

  target_include_directories(hafs_viz PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/viz
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
    ${implot_SOURCE_DIR}
  )

  target_compile_definitions(hafs_viz PRIVATE
    IMGUI_IMPL_OPENGL_LOADER_GLAD=0
  )

  target_link_libraries(hafs_viz PRIVATE
    glfw
    OpenGL::GL
  )

  # Link simdjson if available
  if(simdjson_FOUND)
    target_link_libraries(hafs_viz PRIVATE simdjson::simdjson)
    target_compile_definitions(hafs_viz PRIVATE HAFS_HAS_SIMDJSON=1)
  endif()

  # macOS specific
  if(APPLE)
    target_link_libraries(hafs_viz PRIVATE
      "-framework Cocoa"
      "-framework IOKit"
      "-framework CoreVideo"
    )
  endif()

  # Install
  install(TARGETS hafs_viz RUNTIME DESTINATION bin)

  message(STATUS "Visualization app will be built as 'hafs_viz'")
endif()
```

### File: `src/cc/viz/data_loader.h`

```cpp
#pragma once

#include <map>
#include <string>
#include <vector>

namespace hafs::viz {

// Data structures matching Python side
struct QualityTrendData {
  std::string domain;
  std::string metric;
  std::vector<float> values;
  float mean = 0.0f;
  std::string trend_direction;
};

struct GeneratorStatsData {
  std::string name;
  int samples_generated = 0;
  int samples_accepted = 0;
  int samples_rejected = 0;
  std::map<std::string, int> rejection_reasons;
  float acceptance_rate = 0.0f;
  float avg_quality = 0.0f;
};

struct EmbeddingRegionData {
  int index = 0;
  int sample_count = 0;
  std::string domain;
  float avg_quality = 0.0f;
};

struct TrainingRunData {
  std::string run_id;
  std::string model_name;
  float final_loss = 0.0f;
  int samples_count = 0;
  std::map<std::string, int> domain_distribution;
  std::map<std::string, float> eval_metrics;
};

struct CoverageData {
  int total_samples = 0;
  int num_regions = 0;
  float coverage_score = 0.0f;
  int sparse_regions = 0;
  std::map<std::string, float> domain_coverage;
};

class DataLoader {
 public:
  explicit DataLoader(const std::string& data_path);

  // Reload all data from disk
  bool Refresh();

  // Accessors
  const std::vector<QualityTrendData>& GetQualityTrends() const {
    return quality_trends_;
  }
  const std::vector<GeneratorStatsData>& GetGeneratorStats() const {
    return generator_stats_;
  }
  const std::vector<EmbeddingRegionData>& GetEmbeddingRegions() const {
    return embedding_regions_;
  }
  const std::vector<TrainingRunData>& GetTrainingRuns() const {
    return training_runs_;
  }
  const CoverageData& GetCoverage() const { return coverage_; }

  bool HasData() const { return has_data_; }
  std::string GetLastError() const { return last_error_; }

 private:
  bool LoadQualityFeedback();
  bool LoadActiveLearning();
  bool LoadTrainingFeedback();

  std::string data_path_;
  bool has_data_ = false;
  std::string last_error_;

  std::vector<QualityTrendData> quality_trends_;
  std::vector<GeneratorStatsData> generator_stats_;
  std::vector<EmbeddingRegionData> embedding_regions_;
  std::vector<TrainingRunData> training_runs_;
  CoverageData coverage_;
};

}  // namespace hafs::viz
```

### Build Instructions

```bash
# Install GLFW (macOS)
brew install glfw

# Build with visualization
cd /Users/scawful/Code/hafs
cmake -B build -S src/cc -DHAFS_BUILD_VIZ=ON
cmake --build build

# Run
./build/hafs_viz ~/.context/training

# Or with explicit path
./build/hafs_viz /path/to/training/data
```

---

## Implementation Checklist

### Phase 1: TUI Foundation
- [ ] Add plotext>=5.2.8 to pyproject.toml
- [ ] Create `src/hafs/ui/widgets/training_data_loader.py`
- [ ] Create `src/hafs/ui/widgets/training_charts.py`
- [ ] Create `src/hafs/ui/screens/training_dashboard.py`

### Phase 2: TUI Integration
- [ ] Register `/training` route in `screen_router.py`
- [ ] Add keybinding `9` in `app.py` for training dashboard
- [ ] Add navigation link from main dashboard

### Phase 3: C++ Foundation
- [ ] Create `src/cc/viz/` directory structure
- [ ] Update `src/cc/CMakeLists.txt` with ImGui/ImPlot config
- [ ] Implement `data_loader.h/cc`
- [ ] Implement `main.cc` with GLFW/ImGui setup

### Phase 4: C++ Charts
- [ ] Implement `charts/quality_chart.cc`
- [ ] Implement `charts/generator_chart.cc`
- [ ] Implement `charts/embedding_chart.cc`
- [ ] Implement `charts/training_chart.cc`

### Phase 5: C++ App
- [ ] Implement `app.h/cc` with layout
- [ ] Add menu bar (File, View, Data, Help)
- [ ] Add refresh functionality (F5)
- [ ] Add theming support

### Phase 6: Polish
- [ ] Add chart export (PNG for ImGui)
- [ ] Add data export (CSV)
- [ ] Add zoom/pan for charts
- [ ] Add time range filtering
- [ ] Write user documentation

---

## References

| Resource | Location |
|----------|----------|
| Existing Sparkline widget | `src/hafs/ui/widgets/sparkline.py` |
| Analysis screen pattern | `src/hafs/ui/screens/analysis_screen.py` |
| C++ JSON loading | `src/cc/io/json_loader.cc` |
| Quality tracker data | `src/agents/training/feedback/quality_tracker.py` |
| Training feedback data | `src/agents/training/feedback/training_feedback.py` |
| Active learning data | `src/agents/training/active_learning.py` |
| plotext documentation | https://github.com/piccolomo/plotext |
| ImGui documentation | https://github.com/ocornut/imgui |
| ImPlot documentation | https://github.com/epezent/implot |
