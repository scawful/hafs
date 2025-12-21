from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agents.analysis.deep_context_pipeline import DeepContextPipeline, SmartMLPipeline


def _write_embedding_status(path: Path, *, total: int, done: int, running: bool) -> None:
    payload = {
        "total_symbols": total,
        "total_embeddings": done,
        "running": running,
    }
    path.write_text(json.dumps(payload))


def test_deep_context_pipeline_builds_report(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("TODO: update docs")
    src_dir = repo / "src"
    src_dir.mkdir()
    (src_dir / "app.py").write_text("# FIXME: bug\nprint('hi')\n")

    context_root = tmp_path / ".context"
    embedding_root = context_root / "embedding_service"
    embedding_root.mkdir(parents=True)
    _write_embedding_status(embedding_root / "daemon_status.json", total=10, done=2, running=False)
    kb_dir = context_root / "knowledge" / "alttp"
    kb_dir.mkdir(parents=True)
    (kb_dir / "symbols.json").write_text(json.dumps([{"name": "A"}, {"name": "B"}]))
    (kb_dir / "routines.json").write_text(json.dumps([{"name": "Init"}]))
    (kb_dir / "embedding_index.json").write_text(json.dumps({"sym:A": "a.json"}))

    reports_root = tmp_path / "reports"
    pipeline = DeepContextPipeline(
        repo_root=repo,
        context_root=context_root,
        reports_root=reports_root,
        check_nodes=False,
        llm_summary=False,
    )
    result = asyncio.run(pipeline.generate_report("Test Report"))

    assert result["snapshot"]["total_files"] == 2
    assert result["signals"]["todo_count"] >= 1
    assert "Start embedding daemon to reduce backlog." in result["recommendations"]
    assert result["kb_coverage"]["summary"]["bases"] == 1
    assert result["doc_index"]["total_docs"] >= 1
    assert result["report_path"]
    assert Path(result["report_path"]).exists()


def test_smart_ml_pipeline_recommendations(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    embedding_root = context_root / "embedding_service"
    embedding_root.mkdir(parents=True)
    _write_embedding_status(embedding_root / "daemon_status.json", total=5, done=1, running=False)
    kb_dir = context_root / "knowledge" / "oracle-of-secrets"
    kb_dir.mkdir(parents=True)
    (kb_dir / "symbols.json").write_text(json.dumps([{"name": "Flag"}]))

    reports_root = tmp_path / "reports"
    pipeline = SmartMLPipeline(
        context_root=context_root,
        reports_root=reports_root,
        llm_summary=False,
    )
    result = asyncio.run(pipeline.generate_report("ML Plan"))

    assert "Start embedding daemon to reduce backlog." in result["recommendations"]
    assert result["kb_coverage"]["summary"]["bases"] == 1
    assert result["report_path"]
    assert Path(result["report_path"]).exists()
