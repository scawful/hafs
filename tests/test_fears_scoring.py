from __future__ import annotations

from hafs.core.fears.repository import FearMatch
from hafs.core.fears.scoring import compute_confidence, strongest_match


def test_compute_confidence_keyword_vs_pattern() -> None:
    keyword = FearMatch(
        fear_id="k",
        concern="c",
        mitigation="m",
        matched_by="keyword",
    )
    pattern = FearMatch(
        fear_id="p",
        concern="c",
        mitigation="m",
        matched_by="pattern",
    )

    assert compute_confidence([pattern]) < compute_confidence([keyword])


def test_compute_confidence_multiple_matches_lowers_score() -> None:
    a = FearMatch(fear_id="a", concern="c", mitigation="m", matched_by="keyword")
    b = FearMatch(fear_id="b", concern="c", mitigation="m", matched_by="keyword")
    assert compute_confidence([a, b]) < compute_confidence([a])


def test_strongest_match_prefers_both_over_pattern_over_keyword() -> None:
    keyword = FearMatch(fear_id="k", concern="c", mitigation="m", matched_by="keyword")
    pattern = FearMatch(fear_id="p", concern="c", mitigation="m", matched_by="pattern")
    both = FearMatch(fear_id="b", concern="c", mitigation="m", matched_by="both")

    assert strongest_match([keyword, pattern]) == pattern
    assert strongest_match([keyword, both, pattern]) == both

