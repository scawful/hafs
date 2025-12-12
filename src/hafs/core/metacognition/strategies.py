"""Strategy catalog and recommendation logic for metacognition."""

from __future__ import annotations

from hafs.models.metacognition import Strategy, STRATEGY_DESCRIPTIONS


# Recommendations for when to switch strategies
STRATEGY_RECOMMENDATIONS = {
    Strategy.DIVIDE_AND_CONQUER: {
        "when_to_use": [
            "Problem is complex with multiple independent parts",
            "Clear boundaries between subproblems exist",
            "Subproblems can be solved in parallel",
        ],
        "when_to_avoid": [
            "Problem is tightly coupled",
            "Overhead of decomposition exceeds benefit",
            "Time pressure is extreme",
        ],
        "switch_to_if_failing": Strategy.INCREMENTAL,
    },
    Strategy.DEPTH_FIRST: {
        "when_to_use": [
            "One solution path looks clearly promising",
            "Exploring all options is too expensive",
            "Need to validate feasibility quickly",
        ],
        "when_to_avoid": [
            "Multiple paths have similar probability",
            "Getting stuck would be costly",
            "Need to compare alternatives",
        ],
        "switch_to_if_failing": Strategy.BREADTH_FIRST,
    },
    Strategy.BREADTH_FIRST: {
        "when_to_use": [
            "Uncertainty about which approach is best",
            "Need to compare multiple options",
            "Optimal solution is important",
        ],
        "when_to_avoid": [
            "Time is limited",
            "One option is clearly superior",
            "Resource constraints are tight",
        ],
        "switch_to_if_failing": Strategy.DEPTH_FIRST,
    },
    Strategy.INCREMENTAL: {
        "when_to_use": [
            "Risk needs to be minimized",
            "Frequent validation is possible",
            "Requirements may change",
            "Building on existing system",
        ],
        "when_to_avoid": [
            "Major architectural changes needed",
            "Validation is expensive",
            "Complete rewrite is required",
        ],
        "switch_to_if_failing": Strategy.PROTOTYPE,
    },
    Strategy.RESEARCH_FIRST: {
        "when_to_use": [
            "Domain is unfamiliar",
            "Multiple unknowns exist",
            "Wrong approach would be costly",
            "Time for research is available",
        ],
        "when_to_avoid": [
            "Domain is well understood",
            "Deadline is immediate",
            "Problem is straightforward",
        ],
        "switch_to_if_failing": Strategy.INCREMENTAL,
    },
    Strategy.PROTOTYPE: {
        "when_to_use": [
            "Feasibility is uncertain",
            "Need to demonstrate concept quickly",
            "Requirements are unclear",
            "Stakeholder feedback needed early",
        ],
        "when_to_avoid": [
            "Production quality needed immediately",
            "Problem is well understood",
            "Prototype might become production code",
        ],
        "switch_to_if_failing": Strategy.INCREMENTAL,
    },
}


def get_strategy_description(strategy: Strategy) -> str:
    """Get human-readable description of a strategy.

    Args:
        strategy: The strategy to describe.

    Returns:
        Description string.
    """
    return STRATEGY_DESCRIPTIONS.get(strategy, "Unknown strategy")


def suggest_strategy_change(
    current_strategy: Strategy,
    effectiveness: float,
    effectiveness_threshold: float = 0.4,
) -> Strategy | None:
    """Suggest a strategy change if current one is not working.

    Args:
        current_strategy: The currently active strategy.
        effectiveness: Current strategy effectiveness (0.0 to 1.0).
        effectiveness_threshold: Threshold below which to suggest change.

    Returns:
        New strategy to try, or None if no change suggested.
    """
    if effectiveness >= effectiveness_threshold:
        return None

    recommendation = STRATEGY_RECOMMENDATIONS.get(current_strategy)
    if recommendation:
        return recommendation.get("switch_to_if_failing")

    return None


def get_strategy_for_situation(
    is_complex: bool = False,
    is_unfamiliar: bool = False,
    is_risky: bool = False,
    is_time_pressured: bool = False,
    has_multiple_options: bool = False,
) -> Strategy:
    """Recommend a strategy based on situation characteristics.

    Args:
        is_complex: Problem has multiple parts.
        is_unfamiliar: Domain is not well known.
        is_risky: Mistakes would be costly.
        is_time_pressured: Deadline is tight.
        has_multiple_options: Multiple valid approaches exist.

    Returns:
        Recommended strategy for the situation.
    """
    # Decision tree for strategy selection
    if is_unfamiliar and not is_time_pressured:
        return Strategy.RESEARCH_FIRST

    if is_risky:
        return Strategy.INCREMENTAL

    if is_complex:
        return Strategy.DIVIDE_AND_CONQUER

    if has_multiple_options and not is_time_pressured:
        return Strategy.BREADTH_FIRST

    if is_time_pressured:
        return Strategy.DEPTH_FIRST

    # Default to incremental for safety
    return Strategy.INCREMENTAL


def format_strategy_advice(strategy: Strategy) -> str:
    """Format strategy with usage advice as markdown.

    Args:
        strategy: Strategy to format.

    Returns:
        Markdown-formatted strategy advice.
    """
    desc = get_strategy_description(strategy)
    rec = STRATEGY_RECOMMENDATIONS.get(strategy, {})

    when_to_use = rec.get("when_to_use", [])
    when_to_avoid = rec.get("when_to_avoid", [])

    lines = [
        f"**Strategy: {strategy.value}**",
        f"_{desc}_",
        "",
    ]

    if when_to_use:
        lines.append("**When to use:**")
        for item in when_to_use:
            lines.append(f"- {item}")
        lines.append("")

    if when_to_avoid:
        lines.append("**When to avoid:**")
        for item in when_to_avoid:
            lines.append(f"- {item}")

    return "\n".join(lines)
