"""Sparkline Widget - ASCII sparkline visualization for trends.

This widget provides a simple ASCII sparkline using block characters
to visualize trends in numeric data. Commonly used for metrics like
token usage, latency, or agent activity over time.

Characters used: ▁▂▃▄▅▆▇█ (from lowest to highest)

Usage:
    sparkline = Sparkline(width=20, min_value=0, max_value=100)
    sparkline.set_values([10, 20, 30, 25, 35, 50, 45, 60, 70, 65])
"""

from __future__ import annotations

from typing import List, Optional

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


# Sparkline characters from lowest to highest
SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"


class Sparkline(Widget):
    """ASCII sparkline widget for visualizing numeric trends.

    Displays a compact trend visualization using Unicode block characters.
    Automatically scales values to fit the available character range.

    Attributes:
        values: List of numeric values to visualize
        width: Maximum width of the sparkline (default: 40)
        min_value: Minimum value for scaling (auto-calculated if None)
        max_value: Maximum value for scaling (auto-calculated if None)
    """

    DEFAULT_CSS = """
    Sparkline {
        height: auto;
        width: auto;
    }

    Sparkline Static {
        color: $accent;
        text-style: bold;
    }
    """

    # Reactive attributes
    values: reactive[List[float]] = reactive(list, init=False)
    width: reactive[int] = reactive(40)
    min_value: reactive[Optional[float]] = reactive(None)
    max_value: reactive[Optional[float]] = reactive(None)

    def __init__(
        self,
        values: Optional[List[float]] = None,
        width: int = 40,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Initialize the sparkline widget.

        Args:
            values: Initial values to display
            width: Maximum width of the sparkline
            min_value: Minimum value for scaling (auto if None)
            max_value: Maximum value for scaling (auto if None)
            **kwargs: Additional widget parameters
        """
        super().__init__(**kwargs)
        self.values = values or []
        self.width = width
        self.min_value = min_value
        self.max_value = max_value

    def compose(self) -> ComposeResult:
        """Compose the sparkline display."""
        yield Static(self._render_sparkline(), id="sparkline-display")

    def set_values(self, values: List[float]) -> None:
        """Update the sparkline values.

        Args:
            values: New list of numeric values to display
        """
        self.values = values

    def append_value(self, value: float) -> None:
        """Append a new value to the sparkline.

        Automatically trims to width if values exceed width.

        Args:
            value: New value to append
        """
        new_values = self.values + [value]
        if len(new_values) > self.width:
            new_values = new_values[-self.width:]
        self.values = new_values

    def watch_values(self, new_values: List[float]) -> None:
        """React to values changing."""
        self._update_display()

    def _update_display(self) -> None:
        """Update the sparkline display."""
        try:
            display = self.query_one("#sparkline-display", Static)
            display.update(self._render_sparkline())
        except Exception:
            pass

    def _render_sparkline(self) -> str:
        """Render the sparkline string.

        Returns:
            String of sparkline characters representing the data
        """
        if not self.values:
            return "─" * min(self.width, 20)

        # Determine value range
        min_val = self.min_value if self.min_value is not None else min(self.values)
        max_val = self.max_value if self.max_value is not None else max(self.values)

        # Handle edge case where all values are the same
        if min_val == max_val:
            return SPARKLINE_CHARS[len(SPARKLINE_CHARS) // 2] * len(self.values)

        # Scale values to character indices
        chars = []
        for value in self.values:
            # Normalize to 0-1 range
            normalized = (value - min_val) / (max_val - min_val)
            # Scale to character index
            char_idx = int(normalized * (len(SPARKLINE_CHARS) - 1))
            # Clamp to valid range
            char_idx = max(0, min(char_idx, len(SPARKLINE_CHARS) - 1))
            chars.append(SPARKLINE_CHARS[char_idx])

        return "".join(chars)

    def clear(self) -> None:
        """Clear all values from the sparkline."""
        self.values = []


class LabeledSparkline(Widget):
    """Sparkline with a label and optional value display.

    Convenience widget that combines a label, sparkline, and current value.

    Usage:
        labeled = LabeledSparkline(
            label="Tokens",
            values=[100, 150, 200, 175, 225],
            show_value=True,
        )
    """

    DEFAULT_CSS = """
    LabeledSparkline {
        height: auto;
        width: 100%;
        layout: horizontal;
    }

    LabeledSparkline .sparkline-label {
        width: 16;
        color: $text-disabled;
    }

    LabeledSparkline Sparkline {
        width: 1fr;
    }

    LabeledSparkline .sparkline-value {
        width: 12;
        color: $accent;
        text-align: right;
        text-style: bold;
    }
    """

    def __init__(
        self,
        label: str = "",
        values: Optional[List[float]] = None,
        width: int = 30,
        show_value: bool = True,
        unit: str = "",
        **kwargs,
    ) -> None:
        """Initialize labeled sparkline.

        Args:
            label: Label text to display
            values: Initial sparkline values
            width: Width of the sparkline component
            show_value: Whether to show current value
            unit: Unit suffix for value display (e.g., "ms", "tokens")
            **kwargs: Additional widget parameters
        """
        super().__init__(**kwargs)
        self._label = label
        self._values = values or []
        self._width = width
        self._show_value = show_value
        self._unit = unit

    def compose(self) -> ComposeResult:
        """Compose the labeled sparkline."""
        yield Static(self._label, classes="sparkline-label")
        yield Sparkline(values=self._values, width=self._width)

        if self._show_value:
            value_text = self._format_value()
            yield Static(value_text, classes="sparkline-value")

    def _format_value(self) -> str:
        """Format the current value for display."""
        if not self._values:
            return "-"
        current = self._values[-1]
        if self._unit:
            return f"{current:.1f} {self._unit}"
        return f"{current:.1f}"

    def set_values(self, values: List[float]) -> None:
        """Update sparkline values.

        Args:
            values: New list of values
        """
        self._values = values

        try:
            sparkline = self.query_one(Sparkline)
            sparkline.set_values(values)
        except Exception:
            pass

        if self._show_value:
            try:
                value_display = self.query_one(".sparkline-value", Static)
                value_display.update(self._format_value())
            except Exception:
                pass

    def append_value(self, value: float) -> None:
        """Append a new value.

        Args:
            value: Value to append
        """
        self._values.append(value)
        if len(self._values) > self._width:
            self._values = self._values[-self._width:]

        try:
            sparkline = self.query_one(Sparkline)
            sparkline.append_value(value)
        except Exception:
            pass

        if self._show_value:
            try:
                value_display = self.query_one(".sparkline-value", Static)
                value_display.update(self._format_value())
            except Exception:
                pass
