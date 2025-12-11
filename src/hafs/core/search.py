"""Fuzzy search utilities for HAFS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Sequence, TypeVar

from rapidfuzz import fuzz, process

T = TypeVar("T")

# Default threshold for fuzzy matching (0-100)
DEFAULT_THRESHOLD = 60


@dataclass
class SearchResult(Generic[T]):
    """Result from a fuzzy search."""

    item: T
    score: float
    matched_field: str


def fuzzy_match(query: str, text: str, threshold: float = DEFAULT_THRESHOLD) -> float | None:
    """Check if query fuzzy-matches text.

    Args:
        query: Search query.
        text: Text to match against.
        threshold: Minimum score (0-100) for a match.

    Returns:
        Match score if above threshold, None otherwise.
    """
    if not query or not text:
        return None

    # Use token_set_ratio for better partial matching
    score = fuzz.token_set_ratio(query.lower(), text.lower())
    return score if score >= threshold else None


def fuzzy_filter(
    query: str,
    items: Sequence[T],
    key: Callable[[T], str],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[SearchResult[T]]:
    """Filter items using fuzzy matching.

    Args:
        query: Search query.
        items: Items to filter.
        key: Function to extract searchable text from item.
        threshold: Minimum score (0-100) for inclusion.

    Returns:
        List of matching items with scores, sorted by score descending.
    """
    if not query:
        return [SearchResult(item=item, score=100.0, matched_field="") for item in items]

    results = []
    for item in items:
        text = key(item)
        score = fuzzy_match(query, text, threshold)
        if score is not None:
            results.append(SearchResult(item=item, score=score, matched_field=text))

    # Sort by score descending
    results.sort(key=lambda r: r.score, reverse=True)
    return results


def fuzzy_filter_multi(
    query: str,
    items: Sequence[T],
    keys: dict[str, Callable[[T], str]],
    threshold: float = DEFAULT_THRESHOLD,
    weights: dict[str, float] | None = None,
) -> list[SearchResult[T]]:
    """Filter items using fuzzy matching across multiple fields.

    Args:
        query: Search query.
        items: Items to filter.
        keys: Dictionary of field_name -> extraction function.
        threshold: Minimum score (0-100) for inclusion.
        weights: Optional weights for each field (default: equal weights).

    Returns:
        List of matching items with weighted scores, sorted descending.
    """
    if not query:
        return [SearchResult(item=item, score=100.0, matched_field="") for item in items]

    if weights is None:
        weights = {name: 1.0 for name in keys}

    results = []
    for item in items:
        best_score = 0.0
        best_field = ""

        for field_name, extractor in keys.items():
            text = extractor(item)
            if not text:
                continue

            score = fuzzy_match(query, text, 0)  # Get raw score
            if score is not None:
                weighted_score = score * weights.get(field_name, 1.0)
                if weighted_score > best_score:
                    best_score = weighted_score
                    best_field = field_name

        if best_score >= threshold:
            results.append(SearchResult(item=item, score=best_score, matched_field=best_field))

    # Sort by score descending
    results.sort(key=lambda r: r.score, reverse=True)
    return results


def fuzzy_autocomplete(
    query: str,
    choices: Sequence[str],
    limit: int = 5,
    threshold: float = 40,
) -> list[tuple[str, float]]:
    """Get fuzzy autocomplete suggestions.

    Args:
        query: Partial input to complete.
        choices: Available choices.
        limit: Maximum number of suggestions.
        threshold: Minimum score for inclusion.

    Returns:
        List of (choice, score) tuples, sorted by score descending.
    """
    if not query:
        return [(choice, 100.0) for choice in choices[:limit]]

    # Use process.extract for efficient batch matching
    results = process.extract(
        query.lower(),
        [c.lower() for c in choices],
        scorer=fuzz.token_set_ratio,
        limit=limit,
        score_cutoff=threshold,
    )

    # Map back to original case choices
    lower_to_original = {c.lower(): c for c in choices}
    return [(lower_to_original.get(match, match), score) for match, score, _ in results]


def substring_or_fuzzy(
    query: str,
    text: str,
    fuzzy_threshold: float = DEFAULT_THRESHOLD,
) -> float | None:
    """Match using substring first, falling back to fuzzy.

    Substring matches get score of 100, fuzzy matches get their actual score.

    Args:
        query: Search query.
        text: Text to match against.
        fuzzy_threshold: Minimum score for fuzzy match.

    Returns:
        Match score if matched, None otherwise.
    """
    if not query or not text:
        return None

    query_lower = query.lower()
    text_lower = text.lower()

    # Exact substring match gets perfect score
    if query_lower in text_lower:
        return 100.0

    # Fall back to fuzzy matching
    return fuzzy_match(query, text, fuzzy_threshold)
