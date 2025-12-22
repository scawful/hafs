"""Deterministically contextualize AFS state.md from local AFS content.

This is intentionally non-LLM and best-effort: it updates a few key fields in
`.context/scratchpad/state.md` based on recent files in `.context/history` and
`.context/memory`/`.context/knowledge`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StateContext:
    last_user_input: str
    relevant_history: str
    applicable_rules: str


def _first_interesting_line(text: str) -> str:
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            line = line.lstrip("#").strip()
        return line
    return ""


def _summarize_recent_files(
    directory: Path,
    *,
    max_files: int,
    max_line_len: int,
    exclude_names: set[str] | None = None,
) -> str:
    if not directory.exists() or not directory.is_dir():
        return "none"

    exclude_names = exclude_names or set()

    candidates: list[Path] = []
    try:
        for entry in directory.rglob("*"):
            if not entry.is_file():
                continue
            if entry.name in exclude_names:
                continue
            if entry.name.startswith("."):
                continue
            candidates.append(entry)
    except OSError:
        return "none"

    if not candidates:
        return "none"

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    items: list[str] = []
    for path in candidates[:max_files]:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        snippet = _first_interesting_line(content)
        if snippet:
            snippet = " ".join(snippet.split())
            if len(snippet) > max_line_len:
                snippet = snippet[: max_line_len - 1].rstrip() + "…"
        rel = path.name
        items.append(f"{rel}: {snippet}" if snippet else rel)

    return " | ".join(items) if items else "none"


def build_state_context(project_root: Path, last_user_input: str) -> StateContext:
    context_root = project_root / ".context"
    history_dir = context_root / "history"
    memory_dir = context_root / "memory"
    knowledge_dir = context_root / "knowledge"

    relevant_history = _summarize_recent_files(
        history_dir, max_files=4, max_line_len=90, exclude_names={".keep"}
    )
    memory_summary = _summarize_recent_files(
        memory_dir, max_files=4, max_line_len=90, exclude_names={".keep", "fears.json"}
    )
    knowledge_summary = _summarize_recent_files(
        knowledge_dir, max_files=4, max_line_len=90, exclude_names={".keep"}
    )

    applicable_parts = [p for p in (memory_summary, knowledge_summary) if p != "none"]
    applicable_rules = " | ".join(applicable_parts) if applicable_parts else "none"

    return StateContext(
        last_user_input=last_user_input,
        relevant_history=relevant_history,
        applicable_rules=applicable_rules,
    )


def update_state_md(project_root: Path, *, last_user_input: str) -> bool:
    """Update `.context/scratchpad/state.md` fields in-place."""
    state_file = project_root / ".context" / "scratchpad" / "state.md"
    if not state_file.exists():
        return False

    sanitized = " ".join(last_user_input.strip().splitlines()).strip()
    if not sanitized:
        return False

    ctx = build_state_context(project_root, sanitized)
    try:
        from core.protocol.state_md import update_context_block

        return update_context_block(
            state_file,
            last_user_input=ctx.last_user_input,
            relevant_history=ctx.relevant_history,
            applicable_rules=ctx.applicable_rules,
        )
    except Exception:
        return False
