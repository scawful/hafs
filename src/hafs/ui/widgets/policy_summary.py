"""Compact summary of AFS directory policies for the dashboard."""

from __future__ import annotations

from typing import Iterable

from textual.app import ComposeResult
from textual.containers import HorizontalScroll
from textual.widget import Widget
from textual.widgets import Static

from hafs.config.schema import AFSDirectoryConfig


class PolicySummary(Widget):
    """Render policy chips inline so users can see permissions at a glance."""

    DEFAULT_CSS = """
    PolicySummary {
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1;
    }

    PolicySummary #policy-title {
        color: $text;
        margin-bottom: 1;
    }

    PolicySummary #policy-chips {
        height: 1;
    }

    PolicySummary .policy-chip {
        padding: 0 1;
        height: 1;
        border: solid $primary-darken-2;
        margin-right: 1;
    }

    PolicySummary .policy-read_only {
        color: $info;
    }

    PolicySummary .policy-writable {
        color: $success;
    }

    PolicySummary .policy-executable {
        color: $error;
    }
    """

    def __init__(
        self,
        policies: Iterable[AFSDirectoryConfig] | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self._policies: list[AFSDirectoryConfig] = list(policies or [])

    def compose(self) -> ComposeResult:
        """Compose the policy chip layout."""
        yield Static("[bold]Policies[/] [dim](p to edit)[/dim]", id="policy-title")
        with HorizontalScroll(id="policy-chips"):
            if not self._policies:
                yield Static("[dim]No policy data[/dim]")
            else:
                for cfg in self._policies:
                    policy_value = getattr(cfg.policy, "value", str(cfg.policy))
                    label = f"{cfg.name}: {policy_value.replace('_', ' ')}"
                    yield Static(
                        label,
                        classes=f"policy-chip policy-{policy_value}",
                    )

    def set_policies(self, policies: Iterable[AFSDirectoryConfig]) -> None:
        """Update the displayed policies."""
        self._policies = list(policies)
        self.refresh(recompose=True)
