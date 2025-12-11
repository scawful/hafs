from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from typer.testing import CliRunner

from hafs.cli import app
from hafs.core.parsers.base import BaseParser
from hafs.core.parsers.registry import ParserRegistry

runner = CliRunner()


class _StubItem:
    def __init__(self) -> None:
        self.short_id = "stub-123"
        self.start_time = datetime.now()
        self.messages = ["msg1", "msg2"]
        self.total_tokens = 100


class _StubParser(BaseParser[_StubItem]):
    """Parser stub that records initialization and max_items."""

    last_max_items: int | None = None
    init_base_paths: list[Path] = []

    def __init__(self, base_path: Path | None = None):
        super().__init__(base_path)
        type(self).init_base_paths.append(self.base_path)

    @property
    def name(self) -> str:
        return "gemini"

    def default_path(self) -> Path:
        return Path("default")

    def parse(self, max_items: int = 50) -> list[_StubItem]:
        type(self).last_max_items = max_items
        return [_StubItem()]

    def search(self, query: str, items: list[_StubItem] | None = None) -> list[_StubItem]:
        return items or []


def test_logs_respects_config(tmp_path: Path) -> None:
    """logs command should honor config base_path and max_items."""
    ParserRegistry.clear()
    ParserRegistry.register("gemini", _StubParser)

    # Prepare config and backing directory
    with runner.isolated_filesystem():
        base_dir = Path("logs")
        base_dir.mkdir(parents=True)

        Path("hafs.toml").write_text(
            f"""
[parsers.gemini]
enabled = true
base_path = "{base_dir.resolve()}"
max_items = 3
"""
        )

        result = runner.invoke(app, ["logs", "--parser", "gemini"])

    assert result.exit_code == 0
    assert _StubParser.last_max_items == 3
    # Parser should be constructed with expanded base_path
    assert _StubParser.init_base_paths, "Parser was not initialized"
    assert Path(_StubParser.init_base_paths[-1]).name == "logs"

    ParserRegistry.clear()
    ParserRegistry.load_defaults()


def test_ctx_rejects_path_escape(tmp_path: Path) -> None:
    """ctx subcommands must refuse paths outside .context."""
    context_root = tmp_path / ".context"
    for name in ["memory", "knowledge", "tools", "scratchpad", "history"]:
        target = context_root / name
        target.mkdir(parents=True, exist_ok=True)
        (target / ".keep").touch()

    note = context_root / "scratchpad" / "note.txt"
    note.write_text("hello")
    outside = tmp_path / "outside.txt"
    outside.write_text("oops")

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        ok_result = runner.invoke(app, ["ctx", "view", "scratchpad/note.txt"])
        bad_result = runner.invoke(app, ["ctx", "view", "../outside.txt"])
    finally:
        os.chdir(cwd)

    assert ok_result.exit_code == 0
    assert bad_result.exit_code != 0
    assert "outside context" in bad_result.output.lower()
