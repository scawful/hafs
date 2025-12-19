from __future__ import annotations

import json
from pathlib import Path

from hafs.core.fears.repository import FearsRepository


def test_fears_repository_matches_keywords(tmp_path: Path) -> None:
    fears_path = tmp_path / "fears.json"
    fears_path.write_text(
        json.dumps(
            {
                "version": 1,
                "fears": [
                    {
                        "id": "fear-1",
                        "trigger": {"keywords": ["deploy"]},
                        "concern": "Might break prod",
                        "mitigation": "Confirm environment first",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    repo = FearsRepository(fears_path)
    matches = repo.match("Can you deploy this?")
    assert matches
    assert matches[0].fear_id == "fear-1"
    assert matches[0].matched_by == "keyword"
    assert "prod" in matches[0].concern


def test_fears_repository_matches_pattern(tmp_path: Path) -> None:
    fears_path = tmp_path / "fears.json"
    fears_path.write_text(
        json.dumps(
            {
                "version": 1,
                "fears": [
                    {
                        "id": "fear-2",
                        "trigger": {"pattern": r"rm\s+-rf"},
                        "concern": "Destructive delete",
                        "mitigation": "Ask for confirmation",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    repo = FearsRepository(fears_path)
    matches = repo.match("Please run rm -rf /tmp/test")
    assert matches
    assert matches[0].fear_id == "fear-2"
    assert matches[0].matched_by == "pattern"
