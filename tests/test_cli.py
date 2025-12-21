from __future__ import annotations

import os
from pathlib import Path

from typer.testing import CliRunner

from hafs.cli import app

runner = CliRunner()


def test_history_status_uses_config_context_root() -> None:
    """History status should respect config context_root."""
    with runner.isolated_filesystem():
        context_root = Path("custom_context").resolve()
        Path("hafs.toml").write_text(
            f"""
[general]
context_root = "{context_root}"
"""
        )

        result = runner.invoke(app, ["history", "status"])

        assert result.exit_code == 0
        assert (context_root / "history" / "embeddings").is_dir()


def test_afs_init_mount_list_unmount() -> None:
    """AFS commands should operate on the current project root."""
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["afs", "init"])
        assert result.exit_code == 0
        assert Path(".context").is_dir()

        source_dir = Path("my_docs")
        source_dir.mkdir()
        (source_dir / "note.md").write_text("hello")

        result = runner.invoke(app, ["afs", "mount", "knowledge", str(source_dir)])
        assert result.exit_code == 0
        assert (Path(".context") / "knowledge" / "my_docs").is_symlink()

        result = runner.invoke(app, ["afs", "list"])
        assert result.exit_code == 0
        assert "knowledge" in result.stdout.lower()
        assert "my_docs" in result.stdout

        result = runner.invoke(app, ["afs", "unmount", "knowledge", "my_docs"])
        assert result.exit_code == 0
        assert not (Path(".context") / "knowledge" / "my_docs").exists()

        context_root = Path(".context").resolve()
        sub_dir = Path("subdir")
        sub_dir.mkdir()

        cwd = os.getcwd()
        os.chdir(sub_dir)
        try:
            result = runner.invoke(app, ["afs", "list"])
        finally:
            os.chdir(cwd)

        assert result.exit_code == 0
        assert "Context Root" in result.stdout
        assert str(context_root) in result.stdout
