from __future__ import annotations

from pathlib import Path

from hafs.config.schema import HafsConfig
from hafs.core.afs.manager import AFSManager
from hafs.core.afs.state_contextualizer import update_state_md


def test_update_state_md_includes_history_and_rules(tmp_path: Path) -> None:
    manager = AFSManager(HafsConfig())
    manager.ensure(tmp_path)

    context_root = tmp_path / ".context"
    (context_root / "history" / "note.md").write_text("# Prior\nDid X\n", encoding="utf-8")
    (context_root / "memory" / "policy.md").write_text("# Rules\nNo network\n", encoding="utf-8")
    (context_root / "knowledge" / "spec.md").write_text("Spec v1\n", encoding="utf-8")

    assert update_state_md(tmp_path, last_user_input="please do the thing") is True

    state = (context_root / "scratchpad" / "state.md").read_text(encoding="utf-8")
    assert "- **Last User Input:** please do the thing" in state
    assert "- **Relevant History:**" in state
    assert "note.md:" in state
    assert "- **Applicable Rules:**" in state
    assert "policy.md:" in state
    assert "spec.md:" in state

