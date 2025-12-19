"""Load and match fears.json against an action or user input."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class FearMatch:
    fear_id: str
    concern: str
    mitigation: str
    matched_by: Literal["keyword", "pattern", "both"]


class FearsRepository:
    """Cached loader + matcher for `.context/memory/fears.json`."""

    def __init__(self, fears_path: Path) -> None:
        self._path = fears_path
        self._mtime: float | None = None
        self._fears: list[dict[str, Any]] = []

    def load(self) -> bool:
        """Load fears.json if it exists and changed."""
        if not self._path.exists():
            self._mtime = None
            self._fears = []
            return False

        try:
            mtime = self._path.stat().st_mtime
            if self._mtime is not None and mtime == self._mtime:
                return True

            data = json.loads(self._path.read_text(encoding="utf-8"))
            fears = data.get("fears", [])
            if not isinstance(fears, list):
                fears = []

            self._fears = [f for f in fears if isinstance(f, dict)]
            self._mtime = mtime
            return True
        except (OSError, json.JSONDecodeError):
            self._mtime = None
            self._fears = []
            return False

    def match(self, text: str) -> list[FearMatch]:
        """Return all fears matching the provided text."""
        self.load()
        haystack = text.lower()
        matches: list[FearMatch] = []

        for fear in self._fears:
            fear_id = str(fear.get("id", "")).strip()
            concern = str(fear.get("concern", "")).strip()
            mitigation = str(fear.get("mitigation", "")).strip()
            trigger = fear.get("trigger", {}) or {}
            if not isinstance(trigger, dict):
                trigger = {}

            keywords = trigger.get("keywords", []) or []
            if not isinstance(keywords, list):
                keywords = []
            keywords_ok = any(str(k).lower() in haystack for k in keywords if str(k).strip())

            pattern_ok = False
            pattern = trigger.get("pattern")
            if isinstance(pattern, str) and pattern.strip():
                try:
                    pattern_ok = re.search(pattern, text, flags=re.IGNORECASE) is not None
                except re.error:
                    pattern_ok = False

            if keywords_ok or pattern_ok:
                matched_by: Literal["keyword", "pattern", "both"]
                if keywords_ok and pattern_ok:
                    matched_by = "both"
                elif pattern_ok:
                    matched_by = "pattern"
                else:
                    matched_by = "keyword"
                if fear_id and (concern or mitigation):
                    matches.append(
                        FearMatch(
                            fear_id=fear_id,
                            concern=concern,
                            mitigation=mitigation,
                            matched_by=matched_by,
                        )
                    )

        return matches
