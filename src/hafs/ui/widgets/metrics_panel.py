"""Metrics Panel - Display token usage, cost tracking, and response latency.

This widget subscribes to MetricsEvent from the EventBus and displays
real-time metrics including:
- Token usage (total and by agent)
- Cost tracking (cumulative and per-request)
- Response latency (with sparkline trends)
- Agent activity status

Usage:
    panel = MetricsPanel()
    # Panel automatically subscribes to metrics.* events
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from hafs.ui.core.event_bus import Event, MetricsEvent, get_event_bus
from hafs.ui.core.state_store import get_state_store
from hafs.ui.widgets.sparkline import LabeledSparkline, Sparkline


class MetricsPanel(Widget):
    """Panel for displaying real-time metrics and trends.

    Subscribes to MetricsEvent from the EventBus and maintains
    running statistics for tokens, costs, and latency.

    Displays:
    - Total tokens used (with sparkline)
    - Cost breakdown by agent
    - Response latency trends
    - Agent activity indicators
    """

    DEFAULT_CSS = """
    MetricsPanel {
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    MetricsPanel .metrics-title {
        color: $accent;
        text-style: bold;
        height: 1;
        margin-bottom: 1;
    }

    MetricsPanel .metrics-section {
        height: auto;
        margin-bottom: 1;
    }

    MetricsPanel .metric-row {
        height: 1;
        layout: horizontal;
    }

    MetricsPanel .metric-label {
        width: 20;
        color: $text-disabled;
    }

    MetricsPanel .metric-value {
        width: 15;
        color: $accent;
        text-style: bold;
    }

    MetricsPanel .agent-activity {
        height: auto;
        margin-top: 1;
    }

    MetricsPanel .agent-row {
        height: 1;
        layout: horizontal;
    }

    MetricsPanel .agent-name {
        width: 15;
        color: $text;
    }

    MetricsPanel .agent-tokens {
        width: 10;
        color: $secondary;
        text-align: right;
    }

    MetricsPanel .agent-cost {
        width: 10;
        color: $warning;
        text-align: right;
    }
    """

    # Reactive state
    total_tokens: reactive[int] = reactive(0)
    total_cost: reactive[float] = reactive(0.0)
    avg_latency: reactive[float] = reactive(0.0)

    def __init__(self, max_history: int = 50, **kwargs) -> None:
        """Initialize the metrics panel.

        Args:
            max_history: Maximum history entries for sparklines
            **kwargs: Additional widget parameters
        """
        super().__init__(**kwargs)
        self._bus = get_event_bus()
        self._store = get_state_store()
        self._subscription = None

        # Metrics tracking
        self._max_history = max_history
        self._token_history: deque[float] = deque(maxlen=max_history)
        self._latency_history: deque[float] = deque(maxlen=max_history)
        self._cost_history: deque[float] = deque(maxlen=max_history)

        # Per-agent metrics
        self._agent_tokens: Dict[str, int] = defaultdict(int)
        self._agent_costs: Dict[str, float] = defaultdict(float)
        self._agent_requests: Dict[str, int] = defaultdict(int)

        # Load initial state from store
        self._load_state()

    def _load_state(self) -> None:
        """Load metrics from state store."""
        self.total_tokens = self._store.get("metrics.tokens_used", 0)
        self.total_cost = self._store.get("metrics.cost_total", 0.0)

        # Load latency history
        latency_hist = self._store.get("metrics.latency_history", [])
        for val in latency_hist[-self._max_history:]:
            self._latency_history.append(val)

        # Load per-agent metrics
        agent_tokens = self._store.get("metrics.tokens_by_agent", {})
        for agent_id, tokens in agent_tokens.items():
            self._agent_tokens[agent_id] = tokens

    def compose(self) -> ComposeResult:
        """Compose the metrics panel layout."""
        yield Static("📊 Metrics", classes="metrics-title")

        # Token metrics
        with Container(classes="metrics-section"):
            with Horizontal(classes="metric-row"):
                yield Static("Total Tokens:", classes="metric-label")
                yield Static(f"{self.total_tokens:,}", id="total-tokens", classes="metric-value")

            # Token sparkline
            yield LabeledSparkline(
                label="Token Trend:",
                values=list(self._token_history),
                width=30,
                show_value=False,
                id="token-sparkline",
            )

        # Cost metrics
        with Container(classes="metrics-section"):
            with Horizontal(classes="metric-row"):
                yield Static("Total Cost:", classes="metric-label")
                yield Static(f"${self.total_cost:.4f}", id="total-cost", classes="metric-value")

            # Cost sparkline
            yield LabeledSparkline(
                label="Cost Trend:",
                values=list(self._cost_history),
                width=30,
                show_value=False,
                id="cost-sparkline",
            )

        # Latency metrics
        with Container(classes="metrics-section"):
            with Horizontal(classes="metric-row"):
                yield Static("Avg Latency:", classes="metric-label")
                yield Static(
                    f"{self.avg_latency:.0f} ms" if self.avg_latency > 0 else "-",
                    id="avg-latency",
                    classes="metric-value",
                )

            # Latency sparkline
            yield LabeledSparkline(
                label="Latency Trend:",
                values=list(self._latency_history),
                width=30,
                show_value=False,
                unit="ms",
                id="latency-sparkline",
            )

        # Agent activity section
        with Vertical(classes="agent-activity"):
            yield Static("Agent Activity:", classes="metric-label")
            yield Container(id="agent-list")

    def on_mount(self) -> None:
        """Subscribe to metrics events on mount."""
        self._subscription = self._bus.subscribe(
            "metrics.*",
            self._on_metrics_event,
        )
        self._refresh_agent_list()

    def on_unmount(self) -> None:
        """Unsubscribe from events on unmount."""
        if self._subscription:
            self._subscription.unsubscribe()

    def _on_metrics_event(self, event: Event) -> None:
        """Handle incoming metrics events.

        Args:
            event: The metrics event
        """
        if not isinstance(event, MetricsEvent):
            return

        metric_type = event.metric_type
        value = event.value
        agent_id = event.agent_id

        # Update metrics based on type
        if metric_type == "tokens":
            self._update_tokens(value, agent_id)
        elif metric_type == "latency":
            self._update_latency(value)
        elif metric_type == "cost":
            self._update_cost(value, agent_id)

        # Refresh display
        self._update_display()

    def _update_tokens(self, tokens: float, agent_id: Optional[str]) -> None:
        """Update token metrics.

        Args:
            tokens: Number of tokens
            agent_id: Agent that used the tokens (optional)
        """
        # Update total
        self.total_tokens += int(tokens)
        self._token_history.append(float(self.total_tokens))

        # Update per-agent
        if agent_id:
            self._agent_tokens[agent_id] += int(tokens)
            self._agent_requests[agent_id] += 1

        # Update store
        self._store.set("metrics.tokens_used", self.total_tokens)
        if agent_id:
            self._store.set(
                f"metrics.tokens_by_agent.{agent_id}",
                self._agent_tokens[agent_id],
            )

    def _update_latency(self, latency: float) -> None:
        """Update latency metrics.

        Args:
            latency: Response latency in milliseconds
        """
        self._latency_history.append(latency)

        # Calculate average
        if self._latency_history:
            self.avg_latency = sum(self._latency_history) / len(self._latency_history)

        # Update store
        self._store.set(
            "metrics.latency_history",
            list(self._latency_history),
        )

    def _update_cost(self, cost: float, agent_id: Optional[str]) -> None:
        """Update cost metrics.

        Args:
            cost: Cost amount
            agent_id: Agent that incurred the cost (optional)
        """
        # Update total
        self.total_cost += cost
        self._cost_history.append(float(self.total_cost))

        # Update per-agent
        if agent_id:
            self._agent_costs[agent_id] += cost

        # Update store
        self._store.set("metrics.cost_total", self.total_cost)

    def _update_display(self) -> None:
        """Update all display components."""
        # Update token display
        try:
            token_display = self.query_one("#total-tokens", Static)
            token_display.update(f"{self.total_tokens:,}")
        except Exception:
            pass

        # Update cost display
        try:
            cost_display = self.query_one("#total-cost", Static)
            cost_display.update(f"${self.total_cost:.4f}")
        except Exception:
            pass

        # Update latency display
        try:
            latency_display = self.query_one("#avg-latency", Static)
            text = f"{self.avg_latency:.0f} ms" if self.avg_latency > 0 else "-"
            latency_display.update(text)
        except Exception:
            pass

        # Update sparklines
        try:
            token_spark = self.query_one("#token-sparkline", LabeledSparkline)
            token_spark.set_values(list(self._token_history))
        except Exception:
            pass

        try:
            cost_spark = self.query_one("#cost-sparkline", LabeledSparkline)
            cost_spark.set_values(list(self._cost_history))
        except Exception:
            pass

        try:
            latency_spark = self.query_one("#latency-sparkline", LabeledSparkline)
            latency_spark.set_values(list(self._latency_history))
        except Exception:
            pass

        # Refresh agent list
        self._refresh_agent_list()

    def _refresh_agent_list(self) -> None:
        """Refresh the agent activity list."""
        try:
            container = self.query_one("#agent-list", Container)
            container.remove_children()

            # Sort agents by token usage
            agents = sorted(
                self._agent_tokens.items(),
                key=lambda x: x[1],
                reverse=True,
            )

            # Display top agents
            for agent_id, tokens in agents[:5]:
                cost = self._agent_costs.get(agent_id, 0.0)
                requests = self._agent_requests.get(agent_id, 0)

                # Create agent row
                with container:
                    row = Horizontal(classes="agent-row")
                    with row:
                        Static(agent_id[:14], classes="agent-name")
                        Static(f"{tokens:,}t", classes="agent-tokens")
                        Static(f"${cost:.3f}", classes="agent-cost")

        except Exception:
            pass

    def reset_metrics(self) -> None:
        """Reset all metrics to zero."""
        self.total_tokens = 0
        self.total_cost = 0.0
        self.avg_latency = 0.0

        self._token_history.clear()
        self._latency_history.clear()
        self._cost_history.clear()

        self._agent_tokens.clear()
        self._agent_costs.clear()
        self._agent_requests.clear()

        # Clear store
        self._store.set("metrics.tokens_used", 0)
        self._store.set("metrics.cost_total", 0.0)
        self._store.set("metrics.latency_history", [])
        self._store.set("metrics.tokens_by_agent", {})

        self._update_display()
