from __future__ import annotations

from pathlib import Path

from config.schema import HafsConfig
from core.afs.manager import AFSManager


def test_afs_manager_ensure_scaffolds_protocol_files(tmp_path: Path) -> None:
    cfg = HafsConfig()
    manager = AFSManager(cfg)

    root = manager.ensure(tmp_path)

    context_root = tmp_path / ".context"
    assert root.path == context_root.resolve()

    assert (context_root / "metadata.json").exists()

    # Cognitive protocol scaffolding
    assert (context_root / "scratchpad" / "state.md").exists()
    assert (context_root / "scratchpad" / "deferred.md").exists()
    assert (context_root / "scratchpad" / "metacognition.json").exists()
    assert (context_root / "scratchpad" / "goals.json").exists()
    assert (context_root / "memory" / "fears.json").exists()

