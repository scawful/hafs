"""Heuristics for scoring risk/confidence from fears matches."""

from __future__ import annotations

from collections.abc import Sequence

from core.fears.repository import FearMatch


def compute_confidence(matches: Sequence[FearMatch]) -> float:
    """Compute a 0..1 confidence score given matched fears.

    Higher confidence means lower perceived risk.
    """
    base = 0.85

    for i, match in enumerate(matches):
        primary = i == 0
        if match.matched_by == "both":
            base -= 0.22 if primary else 0.10
        elif match.matched_by == "pattern":
            base -= 0.18 if primary else 0.08
        else:
            base -= 0.12 if primary else 0.06

    if base < 0.05:
        return 0.05
    if base > 0.95:
        return 0.95
    return float(round(base, 2))


def strongest_match(matches: Sequence[FearMatch]) -> FearMatch | None:
    """Pick the most specific/strong match for display."""
    if not matches:
        return None

    priority = {"both": 3, "pattern": 2, "keyword": 1}
    return max(matches, key=lambda m: priority.get(m.matched_by, 0))

