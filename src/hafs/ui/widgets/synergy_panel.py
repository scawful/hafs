"""Synergy panel widget for displaying Theory of Mind metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, ProgressBar, Static

if TYPE_CHECKING:
    from hafs.models.synergy import SynergyScore
    from hafs.services.synergy_service import SynergySummary


class SectionToggled(Message):
    """Emitted when a collapsible section header is toggled."""

    def __init__(self, section_id: str, collapsed: bool) -> None:
        self.section_id = section_id
        self.collapsed = collapsed
        super().__init__()


class CollapsibleHeader(Static):
    """A clickable header that toggles visibility of sibling content."""

    def __init__(self, title: str, collapsed: bool = False, **kwargs) -> None:
        self._title = title
        self._collapsed = collapsed
        display = "▸" if collapsed else "▾"
        super().__init__(f"{display} {title}", **kwargs)

    def on_click(self) -> None:
        self._collapsed = not self._collapsed
        display = "▸" if self._collapsed else "▾"
        self.update(f"{display} {self._title}")
        # Notify parent to toggle content visibility
        self.post_message(SectionToggled(self.id or "", self._collapsed))


class MetacognitionWidget(Vertical):
    """Widget displaying metacognitive state indicators."""

    DEFAULT_CSS = """
    MetacognitionWidget {
        width: 18;
        height: 100%;
        padding: 0 1;
        border-right: solid $primary;
    }
    MetacognitionWidget .meta-title {
        text-style: bold;
        color: $text-disabled;
    }
    MetacognitionWidget .meta-title:hover {
        background: $primary;
    }
    MetacognitionWidget .meta-row {
        height: 1;
    }
    MetacognitionWidget .meta-label {
        color: $text-disabled;
    }
    MetacognitionWidget .status-ok {
        color: $success;
    }
    MetacognitionWidget .status-warn {
        color: $warning;
    }
    MetacognitionWidget .status-error {
        color: $error;
    }
    MetacognitionWidget .flow-active {
        color: $success;
        text-style: bold;
    }
    MetacognitionWidget .flow-inactive {
        color: $text-disabled;
    }
    MetacognitionWidget .meta-content.hidden {
        display: none;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        yield CollapsibleHeader("Metacognition", id="meta-header", classes="meta-title")
        with Vertical(id="meta-content", classes="meta-content"):
            yield Static("Progress: [green]✓[/]", id="progress-status", classes="meta-row")
            yield Static("Load: [dim]0%[/]", id="cognitive-load", classes="meta-row")
            yield Static("Strategy: [dim]--[/]", id="strategy", classes="meta-row")
            yield Static("[dim]FLOW[/]", id="flow-indicator", classes="meta-row flow-inactive")

    def update_metacognition(
        self,
        progress_status: str,
        cognitive_load: float,
        strategy: str,
        strategy_effectiveness: float,
        is_spinning: bool,
        flow_state: bool,
    ) -> None:
        """Update the metacognition display.

        Args:
            progress_status: Current progress status (making_progress, spinning, blocked).
            cognitive_load: Current cognitive load as percentage (0-100).
            strategy: Current strategy name.
            strategy_effectiveness: Strategy effectiveness (0-1).
            is_spinning: Whether spinning is detected.
            flow_state: Whether in flow state.
        """
        try:
            # Progress indicator
            progress_widget = self.query_one("#progress-status", Static)
            if progress_status == "making_progress":
                progress_widget.update("Progress: [green]✓[/]")
            elif progress_status == "spinning":
                progress_widget.update("Progress: [yellow]⟳[/]")
            else:  # blocked
                progress_widget.update("Progress: [red]✗[/]")

            # Cognitive load
            load_widget = self.query_one("#cognitive-load", Static)
            load_pct = int(cognitive_load * 100)
            if load_pct < 50:
                load_widget.update(f"Load: [green]{load_pct}%[/]")
            elif load_pct < 80:
                load_widget.update(f"Load: [yellow]{load_pct}%[/]")
            else:
                load_widget.update(f"Load: [red]{load_pct}%[/]")

            # Strategy with effectiveness indicator
            strategy_widget = self.query_one("#strategy", Static)
            short_strategy = strategy[:8] if len(strategy) > 8 else strategy
            if strategy_effectiveness >= 0.6:
                strategy_widget.update(f"Strategy: [green]{short_strategy}[/]")
            elif strategy_effectiveness >= 0.4:
                strategy_widget.update(f"Strategy: [yellow]{short_strategy}[/]")
            else:
                strategy_widget.update(f"Strategy: [red]{short_strategy}[/]")

            # Flow state indicator
            flow_widget = self.query_one("#flow-indicator", Static)
            if flow_state:
                flow_widget.update("[bold green]⚡ FLOW[/]")
                flow_widget.remove_class("flow-inactive")
                flow_widget.add_class("flow-active")
            else:
                flow_widget.update("[dim]FLOW[/]")
                flow_widget.remove_class("flow-active")
                flow_widget.add_class("flow-inactive")

        except Exception:
            pass


class CognitiveStateWidget(Vertical):
    """A widget to display the agent's cognitive and emotional state."""

    DEFAULT_CSS = """
    CognitiveStateWidget {
        width: 28;
        height: 100%;
        padding-left: 1;
        border-left: solid $primary;
    }
    CognitiveStateWidget .cognitive-title {
        text-style: bold;
        color: $text-disabled;
        margin-bottom: 1;
    }
    CognitiveStateWidget .cognitive-title:hover {
        background: $primary;
    }
    CognitiveStateWidget .concern-list {
        height: auto;
    }
    CognitiveStateWidget .confidence-label {
        color: $text-disabled;
    }
    CognitiveStateWidget #mitigation {
        height: auto;
    }
    CognitiveStateWidget .cognitive-content.hidden {
        display: none;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the widget's layout."""
        yield CollapsibleHeader("Cognitive State", id="cognitive-header", classes="cognitive-title")
        with Vertical(id="cognitive-content", classes="cognitive-content"):
            yield Static("Concerns: [dim]None[/dim]", id="concerns", classes="concern-list")
            yield Label("Confidence", classes="confidence-label")
            yield ProgressBar(id="confidence-bar", total=1.0, show_percentage=True)
            yield Static("Mitigation: [dim]N/A[/dim]", id="mitigation")

    def update_state(
        self,
        concerns: list[str],
        confidence: float,
        mitigation: str,
    ) -> None:
        """Update the displayed cognitive state."""
        try:
            concerns_widget = self.query_one("#concerns", Static)
            if concerns:
                concern_text = "\n".join(f"- {c}" for c in concerns)
                concerns_widget.update(f"Concerns:\n[yellow]{concern_text}[/]")
            else:
                concerns_widget.update("Concerns: [green]None[/]")

            confidence_bar = self.query_one("#confidence-bar", ProgressBar)
            confidence_bar.progress = confidence

            mitigation_widget = self.query_one("#mitigation", Static)
            if mitigation:
                mitigation_widget.update(f"Mitigation: {mitigation}")
            else:
                mitigation_widget.update("Mitigation: [dim]N/A[/dim]")
        except Exception:
            pass


class AbilityTrackingWidget(Vertical):
    """Widget displaying IRT-based ability estimates.

    Shows θ (individual), κ (collaborative), and synergy gain (κ - θ)
    based on "Quantifying Human-AI Synergy" research.
    """

    DEFAULT_CSS = """
    AbilityTrackingWidget {
        width: 22;
        height: 100%;
        padding: 0 1;
        border-left: solid $primary;
    }
    AbilityTrackingWidget .ability-title {
        text-style: bold;
        color: $text-disabled;
    }
    AbilityTrackingWidget .ability-title:hover {
        background: $primary;
    }
    AbilityTrackingWidget .ability-row {
        height: 1;
    }
    AbilityTrackingWidget .ability-label {
        color: $text-disabled;
        width: 8;
    }
    AbilityTrackingWidget .synergy-positive {
        color: $success;
        text-style: bold;
    }
    AbilityTrackingWidget .synergy-negative {
        color: $error;
        text-style: bold;
    }
    AbilityTrackingWidget .synergy-neutral {
        color: $text-disabled;
    }
    AbilityTrackingWidget .reliable {
        color: $success;
    }
    AbilityTrackingWidget .unreliable {
        color: $warning;
    }
    AbilityTrackingWidget .ability-content.hidden {
        display: none;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        yield CollapsibleHeader("IRT Ability", id="ability-header", classes="ability-title")
        with Vertical(id="ability-content", classes="ability-content"):
            yield Static("θ: [dim]--[/]", id="theta-value", classes="ability-row")
            yield Static("κ: [dim]--[/]", id="kappa-value", classes="ability-row")
            yield Static("Δ: [dim]--[/]", id="synergy-gain", classes="ability-row")
            yield Static("ToM: [dim]--[/]", id="tom-score", classes="ability-row")
            yield Static("[dim]--[/]", id="benefit-label", classes="ability-row")

    def update_abilities(
        self,
        theta: float,
        theta_reliable: bool,
        kappa: float,
        kappa_reliable: bool,
        synergy_gain: float,
        ai_benefit: str,
        tom_score: float | None = None,
    ) -> None:
        """Update the ability display.

        Args:
            theta: Individual ability estimate (θ).
            theta_reliable: Whether θ estimate is reliable.
            kappa: Collaborative ability estimate (κ).
            kappa_reliable: Whether κ estimate is reliable.
            synergy_gain: κ - θ value.
            ai_benefit: Benefit category string.
            tom_score: Optional recent ToM score.
        """
        try:
            # Theta (individual ability)
            theta_widget = self.query_one("#theta-value", Static)
            theta_class = "reliable" if theta_reliable else "unreliable"
            theta_widget.update(f"θ: [{theta_class}]{theta:+.2f}[/]")

            # Kappa (collaborative ability)
            kappa_widget = self.query_one("#kappa-value", Static)
            kappa_class = "reliable" if kappa_reliable else "unreliable"
            kappa_widget.update(f"κ: [{kappa_class}]{kappa:+.2f}[/]")

            # Synergy gain (κ - θ)
            synergy_widget = self.query_one("#synergy-gain", Static)
            if synergy_gain > 0.1:
                synergy_widget.update(f"Δ: [green]+{synergy_gain:.2f}[/]")
            elif synergy_gain < -0.1:
                synergy_widget.update(f"Δ: [red]{synergy_gain:.2f}[/]")
            else:
                synergy_widget.update(f"Δ: [dim]{synergy_gain:+.2f}[/]")

            # ToM score
            tom_widget = self.query_one("#tom-score", Static)
            if tom_score is not None:
                if tom_score >= 3.5:
                    tom_widget.update(f"ToM: [green]{tom_score:.1f}[/]")
                elif tom_score >= 2.5:
                    tom_widget.update(f"ToM: [yellow]{tom_score:.1f}[/]")
                else:
                    tom_widget.update(f"ToM: [red]{tom_score:.1f}[/]")
            else:
                tom_widget.update("ToM: [dim]--[/]")

            # Benefit label
            benefit_widget = self.query_one("#benefit-label", Static)
            benefit_map = {
                "significant_benefit": "[green]AI helps![/]",
                "moderate_benefit": "[green]AI helps[/]",
                "neutral": "[dim]Neutral[/]",
                "slight_hindrance": "[yellow]Mild issue[/]",
                "significant_hindrance": "[red]Impaired[/]",
            }
            benefit_widget.update(benefit_map.get(ai_benefit, "[dim]--[/]"))

        except Exception:
            pass

    def reset(self) -> None:
        """Reset to default values."""
        try:
            self.query_one("#theta-value", Static).update("θ: [dim]--[/]")
            self.query_one("#kappa-value", Static).update("κ: [dim]--[/]")
            self.query_one("#synergy-gain", Static).update("Δ: [dim]--[/]")
            self.query_one("#tom-score", Static).update("ToM: [dim]--[/]")
            self.query_one("#benefit-label", Static).update("[dim]--[/]")
        except Exception:
            pass


class SynergyPanel(Widget):
    """Panel displaying ToM metrics, metacognition, and cognitive state."""

    DEFAULT_CSS = """
    SynergyPanel {
        height: 9;
        dock: bottom;
        background: $surface;
        border-top: solid $primary;
        padding: 1;
    }
    SynergyPanel .score-section {
        width: 12;
        height: 100%;
        padding-right: 1;
    }
    SynergyPanel .score-value {
        text-style: bold;
        color: $accent;
    }
    SynergyPanel .score-label {
        color: $text-disabled;
    }
    SynergyPanel .metrics-section {
        width: 1fr;
        height: 100%;
        padding-right: 1;
    }
    SynergyPanel .metric-row {
        height: 1;
        width: 100%;
    }
    SynergyPanel .metric-label {
        width: 12;
        color: $text-disabled;
    }
    """

    synergy_total: reactive[float] = reactive(0.0)
    flow_state: reactive[bool] = reactive(False)

    def __init__(
        self,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize synergy panel."""
        super().__init__(id=id, classes=classes)
        self._last_state_mtime: float = 0.0
        self._last_meta_mtime: float = 0.0
        self._score: "SynergyScore | None" = None

    def on_mount(self) -> None:
        """Start polling for state changes."""
        self.set_interval(2.0, self._check_all_state)

    def on_section_toggled(self, event: SectionToggled) -> None:
        """Handle section toggle events from collapsible headers."""
        section_id = event.section_id
        collapsed = event.collapsed

        # Map header IDs to content container IDs
        content_map = {
            "meta-header": "#meta-content",
            "cognitive-header": "#cognitive-content",
            "ability-header": "#ability-content",
        }

        content_id = content_map.get(section_id)
        if content_id:
            try:
                content = self.query_one(content_id)
                if collapsed:
                    content.add_class("hidden")
                else:
                    content.remove_class("hidden")
            except Exception:
                pass

    def _check_all_state(self) -> None:
        """Check for changes in state.md and metacognition.json."""
        self._check_cognitive_state()
        self._check_metacognition_state()

    def _check_cognitive_state(self) -> None:
        """Periodically check for changes in state.md."""
        state_file = Path.cwd() / ".context" / "scratchpad" / "state.md"
        if not state_file.exists():
            return

        try:
            mtime = state_file.stat().st_mtime
            if mtime > self._last_state_mtime:
                self._last_state_mtime = mtime
                content = state_file.read_text()
                concerns, confidence, mitigation = self._parse_state_content(content)
                self.update_cognitive_state(concerns, confidence, mitigation)
        except (FileNotFoundError, OSError):
            pass

    def _check_metacognition_state(self) -> None:
        """Periodically check for changes in metacognition.json."""
        meta_file = Path.cwd() / ".context" / "scratchpad" / "metacognition.json"
        if not meta_file.exists():
            return

        try:
            mtime = meta_file.stat().st_mtime
            if mtime > self._last_meta_mtime:
                self._last_meta_mtime = mtime
                content = meta_file.read_text()
                data = json.loads(content)
                self._update_metacognition_from_data(data)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            pass

    def _update_metacognition_from_data(self, data: dict) -> None:
        """Update metacognition widget from parsed JSON data."""
        try:
            meta_widget = self.query_one("#metacognition", MetacognitionWidget)

            try:
                from hafs.core.protocol.metacognition_compat import normalize_metacognition

                data = normalize_metacognition(data)
            except Exception:
                pass

            progress_status = data.get("progress_status", "making_progress")
            cognitive_load = data.get("cognitive_load", {}).get("current", 0.0)
            strategy = data.get("current_strategy", "incremental")
            effectiveness = data.get("strategy_effectiveness", 0.5)
            spin_data = data.get("spin_detection", {})
            is_spinning = spin_data.get("similar_action_count", 0) >= spin_data.get(
                "spinning_threshold", 4
            )
            flow_state = data.get("flow_state", False)

            # Update reactive for external observation
            self.flow_state = flow_state

            meta_widget.update_metacognition(
                progress_status=progress_status,
                cognitive_load=cognitive_load,
                strategy=strategy,
                strategy_effectiveness=effectiveness,
                is_spinning=is_spinning,
                flow_state=flow_state,
            )
        except Exception:
            pass

    def _parse_state_content(self, content: str) -> tuple[list[str], float, str]:
        """Parse state.md to extract cognitive state."""
        concerns: list[str] = []
        confidence: float = 0.0
        mitigation: str = ""
        lines = content.splitlines()
        in_section = False
        in_hafs_risk = False

        for line in lines:
            if "## 5. Emotional State & Risk Assessment" in line:
                in_section = True
                continue
            if "## 6." in line:  # Next section
                in_section = False
                continue
            if "<!-- hafs:risk:start -->" in line:
                in_hafs_risk = True
                continue
            if "<!-- hafs:risk:end -->" in line:
                in_hafs_risk = False
                continue
            if not in_section and not in_hafs_risk:
                continue
            if in_hafs_risk and "## HAFS Risk" in line:
                continue

            clean_line = line.replace("**", "")

            if "Identified Concerns:" in clean_line:
                parts = clean_line.split("Identified Concerns:", 1)
                if len(parts) > 1:
                    val = parts[1].strip()
                    if val and val.lower() not in ("none", "n/a", ""):
                        concerns = [c.strip() for c in val.split(",") if c.strip()]
            elif "Confidence Score" in clean_line:
                parts = clean_line.split(":")
                if len(parts) > 1:
                    val = parts[-1].strip()
                    if val:
                        try:
                            confidence = float(val)
                        except ValueError:
                            confidence = 0.0
            elif "Mitigation Strategy:" in clean_line:
                parts = clean_line.split("Mitigation Strategy:", 1)
                if len(parts) > 1:
                    val = parts[1].strip()
                    if val and val.lower() not in ("n/a", "none", ""):
                        mitigation = val

        return concerns, confidence, mitigation

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        with Horizontal():
            # Synergy score section
            with Vertical(classes="score-section"):
                yield Static("Synergy", classes="score-label")
                yield Static("--", id="score-value", classes="score-value")

            # Metacognition section (NEW)
            yield MetacognitionWidget(id="metacognition")

            # ToM metrics section
            with Vertical(classes="metrics-section"):
                with Horizontal(classes="metric-row"):
                    yield Static("ToM", classes="metric-label")
                    yield ProgressBar(id="tom-bar", total=100, show_percentage=True)
                with Horizontal(classes="metric-row"):
                    yield Static("Quality", classes="metric-label")
                    yield ProgressBar(id="quality-bar", total=100, show_percentage=True)
                with Horizontal(classes="metric-row"):
                    yield Static("Alignment", classes="metric-label")
                    yield ProgressBar(id="alignment-bar", total=100, show_percentage=True)
                with Horizontal(classes="metric-row"):
                    yield Static("Context", classes="metric-label")
                    yield ProgressBar(id="context-bar", total=100, show_percentage=True)

            # Cognitive state section
            yield CognitiveStateWidget(id="cognitive-state")

            # IRT Ability tracking section (research-based)
            yield AbilityTrackingWidget(id="ability-tracking")

    def update_score(self, score: "SynergyScore") -> None:
        """Update displayed synergy score."""
        self._score = score
        self.synergy_total = score.total
        try:
            self.query_one("#score-value", Static).update(f"{score.total:.0f}")
            self._update_bar("tom-bar", score.breakdown.get("tom_markers", 0))
            self._update_bar("quality-bar", score.breakdown.get("response_quality", 0))
            self._update_bar("alignment-bar", score.breakdown.get("user_alignment", 0))
            self._update_bar("context-bar", score.breakdown.get("context_utilization", 0))
        except Exception:
            pass

    def update_cognitive_state(
        self,
        concerns: list[str],
        confidence: float,
        mitigation: str,
    ) -> None:
        """Update the cognitive state display."""
        try:
            cognitive_widget = self.query_one("#cognitive-state", CognitiveStateWidget)
            cognitive_widget.update_state(concerns, confidence, mitigation)
        except Exception:
            pass

    def update_metacognition(
        self,
        progress_status: str,
        cognitive_load: float,
        strategy: str,
        strategy_effectiveness: float,
        is_spinning: bool,
        flow_state: bool,
    ) -> None:
        """Update the metacognition display directly."""
        try:
            meta_widget = self.query_one("#metacognition", MetacognitionWidget)
            meta_widget.update_metacognition(
                progress_status=progress_status,
                cognitive_load=cognitive_load,
                strategy=strategy,
                strategy_effectiveness=strategy_effectiveness,
                is_spinning=is_spinning,
                flow_state=flow_state,
            )
            self.flow_state = flow_state
        except Exception:
            pass

    def _update_bar(self, bar_id: str, value: float) -> None:
        """Update a progress bar."""
        try:
            bar = self.query_one(f"#{bar_id}", ProgressBar)
            bar.progress = value
        except Exception:
            pass

    def update_synergy_summary(self, summary: dict) -> None:
        """Update the ability tracking from a synergy summary.

        Args:
            summary: Dictionary from SynergySummary.to_dict() with keys:
                - theta_individual, theta_individual_reliable
                - kappa_collaborative, kappa_collaborative_reliable
                - synergy_gain, ai_benefit
                - recent_tom_score
        """
        try:
            ability_widget = self.query_one("#ability-tracking", AbilityTrackingWidget)
            ability_widget.update_abilities(
                theta=summary.get("theta_individual", 0.0),
                theta_reliable=summary.get("theta_individual_reliable", False),
                kappa=summary.get("kappa_collaborative", 0.0),
                kappa_reliable=summary.get("kappa_collaborative_reliable", False),
                synergy_gain=summary.get("synergy_gain", 0.0),
                ai_benefit=summary.get("ai_benefit", "neutral"),
                tom_score=summary.get("recent_tom_score"),
            )
        except Exception:
            pass

    def reset(self) -> None:
        """Reset all state to defaults."""
        self.synergy_total = 0.0
        self.flow_state = False
        self._score = None
        try:
            self.query_one("#score-value", Static).update("--")
            for bar_id in ["tom-bar", "quality-bar", "alignment-bar", "context-bar"]:
                self._update_bar(bar_id, 0)
            self.query_one("#cognitive-state", CognitiveStateWidget).update_state([], 0.0, "")
            self.query_one("#metacognition", MetacognitionWidget).update_metacognition(
                progress_status="making_progress",
                cognitive_load=0.0,
                strategy="incremental",
                strategy_effectiveness=0.5,
                is_spinning=False,
                flow_state=False,
            )
            self.query_one("#ability-tracking", AbilityTrackingWidget).reset()
        except Exception:
            pass

    @property
    def current_score(self) -> "SynergyScore | None":
        """Get the current synergy score."""
        return self._score

    @property
    def is_in_flow_state(self) -> bool:
        """Check if currently in flow state."""
        return self.flow_state
