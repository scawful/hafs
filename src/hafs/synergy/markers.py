"""Theory of Mind marker detection patterns."""

from __future__ import annotations

import re
from typing import Pattern

from hafs.models.synergy import ToMMarkerType

# Compiled regex patterns for detecting Theory of Mind markers in text
TOM_PATTERNS: dict[ToMMarkerType, list[Pattern[str]]] = {
    ToMMarkerType.PERSPECTIVE_TAKING: [
        re.compile(r"\bfrom your perspective\b", re.IGNORECASE),
        re.compile(r"\byou might think\b", re.IGNORECASE),
        re.compile(r"\byou might feel\b", re.IGNORECASE),
        re.compile(r"\byou might want\b", re.IGNORECASE),
        re.compile(r"\byou may think\b", re.IGNORECASE),
        re.compile(r"\byou may feel\b", re.IGNORECASE),
        re.compile(r"\byou may want\b", re.IGNORECASE),
        re.compile(r"\bin your position\b", re.IGNORECASE),
        re.compile(r"\bfrom your point of view\b", re.IGNORECASE),
        re.compile(r"\bfrom where you(?:'re| are) standing\b", re.IGNORECASE),
        re.compile(r"\bhow you(?:'re| are) seeing\b", re.IGNORECASE),
        re.compile(r"\bif I were you\b", re.IGNORECASE),
    ],
    ToMMarkerType.GOAL_INFERENCE: [
        re.compile(r"\byour goal (?:is|seems)\b", re.IGNORECASE),
        re.compile(r"\byou(?:'re| are) trying to\b", re.IGNORECASE),
        re.compile(r"\bwhat you want to achieve\b", re.IGNORECASE),
        re.compile(r"\bwhat you(?:'re| are) aiming for\b", re.IGNORECASE),
        re.compile(r"\byour objective (?:is|seems)\b", re.IGNORECASE),
        re.compile(r"\byour intention (?:is|seems)\b", re.IGNORECASE),
        re.compile(r"\bwhat you(?:'re| are) working towards\b", re.IGNORECASE),
        re.compile(r"\bit seems you(?:'re| are) looking to\b", re.IGNORECASE),
        re.compile(r"\byou appear to be trying to\b", re.IGNORECASE),
    ],
    ToMMarkerType.KNOWLEDGE_GAP_DETECTION: [
        re.compile(r"\byou might not know\b", re.IGNORECASE),
        re.compile(r"\byou may not (?:be aware|know)\b", re.IGNORECASE),
        re.compile(r"\bI should mention\b", re.IGNORECASE),
        re.compile(r"\bI should point out\b", re.IGNORECASE),
        re.compile(r"\bin case you(?:'re| are) not aware\b", re.IGNORECASE),
        re.compile(r"\bin case you don't know\b", re.IGNORECASE),
        re.compile(r"\byou may not realize\b", re.IGNORECASE),
        re.compile(r"\bit's worth noting\b", re.IGNORECASE),
        re.compile(r"\bfor your information\b", re.IGNORECASE),
        re.compile(r"\byou might be unaware\b", re.IGNORECASE),
    ],
    ToMMarkerType.COMMUNICATION_REPAIR: [
        re.compile(r"\blet me clarify\b", re.IGNORECASE),
        re.compile(r"\bto be more specific\b", re.IGNORECASE),
        re.compile(r"\bto be more precise\b", re.IGNORECASE),
        re.compile(r"\bwhat I meant was\b", re.IGNORECASE),
        re.compile(r"\bto put it another way\b", re.IGNORECASE),
        re.compile(r"\bin other words\b", re.IGNORECASE),
        re.compile(r"\blet me rephrase\b", re.IGNORECASE),
        re.compile(r"\bto clarify\b", re.IGNORECASE),
        re.compile(r"\bwhat I(?:'m| am) saying is\b", re.IGNORECASE),
        re.compile(r"\bto elaborate\b", re.IGNORECASE),
    ],
    ToMMarkerType.CONFIRMATION_SEEKING: [
        re.compile(r"\bis that correct\??", re.IGNORECASE),
        re.compile(r"\bdoes that make sense\??", re.IGNORECASE),
        re.compile(r"\bdo you follow\??", re.IGNORECASE),
        re.compile(r"\bam I understanding (?:correctly|right)\??", re.IGNORECASE),
        re.compile(r"\bdid I get that right\??", re.IGNORECASE),
        re.compile(r"\bis that what you meant\??", re.IGNORECASE),
        re.compile(r"\bdo you understand\??", re.IGNORECASE),
        re.compile(r"\bwould you like me to\b", re.IGNORECASE),
        re.compile(r"\bshould I\b", re.IGNORECASE),
        re.compile(r"\bcan you confirm\b", re.IGNORECASE),
    ],
    ToMMarkerType.MENTAL_STATE_ATTRIBUTION: [
        re.compile(r"\byou(?:'re| are) good at\b", re.IGNORECASE),
        re.compile(r"\bas an AI\b", re.IGNORECASE),
        re.compile(r"\bas a(?:n)? (?:language model|assistant|AI)\b", re.IGNORECASE),
        re.compile(r"\bgiven your capabilities\b", re.IGNORECASE),
        re.compile(r"\bwith your expertise\b", re.IGNORECASE),
        re.compile(r"\byou(?:'re| are) capable of\b", re.IGNORECASE),
        re.compile(r"\byou can\b", re.IGNORECASE),
        re.compile(r"\byou(?:'re| are) able to\b", re.IGNORECASE),
        re.compile(r"\byour strength(?:s)? (?:is|are)\b", re.IGNORECASE),
    ],
    ToMMarkerType.PLAN_COORDINATION: [
        re.compile(r"\blet(?:'s| us) work together\b", re.IGNORECASE),
        re.compile(r"\bcan you handle\b", re.IGNORECASE),
        re.compile(r"\bI(?:'ll| will) focus on\b", re.IGNORECASE),
        re.compile(r"\byou take care of\b", re.IGNORECASE),
        re.compile(r"\bwe can collaborate\b", re.IGNORECASE),
        re.compile(r"\blet(?:'s| us) divide\b", re.IGNORECASE),
        re.compile(r"\bI(?:'ll| will) handle\b", re.IGNORECASE),
        re.compile(r"\byou can work on\b", re.IGNORECASE),
        re.compile(r"\bwe should coordinate\b", re.IGNORECASE),
        re.compile(r"\blet(?:'s| us) team up\b", re.IGNORECASE),
    ],
    ToMMarkerType.CHALLENGE_DISAGREE: [
        re.compile(r"\bcould you be wrong\b", re.IGNORECASE),
        re.compile(r"\bare you sure\b", re.IGNORECASE),
        re.compile(r"\bwhat if you(?:'re| are)\b", re.IGNORECASE),
        re.compile(r"\bhave you considered\b", re.IGNORECASE),
        re.compile(r"\bI(?:'m| am) not sure (?:that|if)\b", re.IGNORECASE),
        re.compile(r"\bI disagree\b", re.IGNORECASE),
        re.compile(r"\bthat doesn't seem right\b", re.IGNORECASE),
        re.compile(r"\bI think you(?:'re| are) mistaken\b", re.IGNORECASE),
        re.compile(r"\bactually,\b", re.IGNORECASE),
        re.compile(r"\bhowever,\b", re.IGNORECASE),
    ],
}


def get_all_patterns() -> dict[ToMMarkerType, list[Pattern[str]]]:
    """
    Get all Theory of Mind detection patterns.

    Returns:
        Dictionary mapping ToMMarkerType to list of compiled regex patterns.
    """
    return TOM_PATTERNS


def get_patterns_for_type(marker_type: ToMMarkerType) -> list[Pattern[str]]:
    """
    Get patterns for a specific ToM marker type.

    Args:
        marker_type: The type of marker to get patterns for.

    Returns:
        List of compiled regex patterns for the specified marker type.
    """
    return TOM_PATTERNS.get(marker_type, [])
