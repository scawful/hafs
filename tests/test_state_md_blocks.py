from __future__ import annotations

from hafs.core.protocol.state_md import upsert_block


def test_upsert_block_appends_when_missing() -> None:
    out = upsert_block(
        "# Title\n",
        start_marker="<!-- hafs:test:start -->",
        end_marker="<!-- hafs:test:end -->",
        body="## Block\n- a",
    )
    assert "<!-- hafs:test:start -->" in out
    assert "## Block" in out
    assert "<!-- hafs:test:end -->" in out


def test_upsert_block_replaces_existing() -> None:
    content = "\n".join(
        [
            "# Title",
            "<!-- hafs:test:start -->",
            "old",
            "<!-- hafs:test:end -->",
            "",
        ]
    )
    out = upsert_block(
        content,
        start_marker="<!-- hafs:test:start -->",
        end_marker="<!-- hafs:test:end -->",
        body="new",
    )
    assert "old" not in out
    assert "new" in out

