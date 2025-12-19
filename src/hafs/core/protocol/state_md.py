"""Helpers for safely updating state.md without clobbering user content."""

from __future__ import annotations

from pathlib import Path


def upsert_block(
    content: str,
    *,
    start_marker: str,
    end_marker: str,
    body: str,
) -> str:
    """Insert or replace a delimited block in markdown content."""
    start = content.find(start_marker)
    end = content.find(end_marker)

    block = f"{start_marker}\n{body.rstrip()}\n{end_marker}\n"

    if start != -1 and end != -1 and end > start:
        before = content[:start].rstrip()
        after = content[end + len(end_marker) :].lstrip("\n")
        joined = "\n\n".join([p for p in (before, block.rstrip(), after) if p])
        return joined.rstrip() + "\n"

    # Append block
    base = content.rstrip()
    if base:
        return base + "\n\n" + block
    return block


def update_context_block(
    state_path: Path,
    *,
    last_user_input: str,
    relevant_history: str,
    applicable_rules: str,
) -> bool:
    """Update the HAFS context block inside `state.md`."""
    if not state_path.exists():
        return False

    try:
        content = state_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False

    body = "\n".join(
        [
            "## HAFS Context",
            f"- **Last User Input:** {last_user_input}",
            f"- **Relevant History:** {relevant_history}",
            f"- **Applicable Rules:** {applicable_rules}",
        ]
    )

    updated = upsert_block(
        content,
        start_marker="<!-- hafs:context:start -->",
        end_marker="<!-- hafs:context:end -->",
        body=body,
    )

    try:
        state_path.write_text(updated, encoding="utf-8")
    except OSError:
        return False
    return True


def update_risk_block(
    state_path: Path,
    *,
    concern: str,
    confidence: float,
    mitigation: str,
) -> bool:
    """Update the HAFS risk block inside `state.md`."""
    if not state_path.exists():
        return False

    try:
        content = state_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False

    body = "\n".join(
        [
            "## HAFS Risk",
            f"- **Identified Concerns:** {concern or 'none'}",
            f"- **Confidence Score (0-1):** {confidence:.2f}",
            f"- **Mitigation Strategy:** {mitigation or 'none'}",
        ]
    )

    updated = upsert_block(
        content,
        start_marker="<!-- hafs:risk:start -->",
        end_marker="<!-- hafs:risk:end -->",
        body=body,
    )

    try:
        state_path.write_text(updated, encoding="utf-8")
    except OSError:
        return False
    return True

