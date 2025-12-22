from pathlib import Path

from tui.utils.file_ops import (
    can_edit_file,
    duplicate_file,
    next_copy_path,
    read_text_file,
    rename_path,
)


def test_can_edit_file_detects_text_vs_binary(tmp_path: Path) -> None:
    text_file = tmp_path / "note.md"
    text_file.write_text("hello world")

    binary_file = tmp_path / "bin.dat"
    binary_file.write_bytes(b"\x00\x01\x02")

    assert can_edit_file(text_file) is True
    assert can_edit_file(binary_file) is False


def test_duplicate_file_creates_unique_copy(tmp_path: Path) -> None:
    src = tmp_path / "data.txt"
    src.write_text("sample")

    first_copy = duplicate_file(src)
    second_name = next_copy_path(src)

    assert first_copy.exists()
    assert first_copy.read_text() == "sample"
    assert "copy" in first_copy.name
    assert second_name.parent == tmp_path
    assert not second_name.exists()


def test_rename_path_moves_file(tmp_path: Path) -> None:
    src = tmp_path / "old.txt"
    src.write_text("content")
    dest = tmp_path / "new.txt"

    rename_path(src, dest)

    assert not src.exists()
    assert dest.exists()
    assert dest.read_text() == "content"


def test_read_text_file_truncates(tmp_path: Path) -> None:
    src = tmp_path / "long.txt"
    src.write_text("x" * 200)

    truncated = read_text_file(src, limit=50)

    assert len(truncated) < 200
    assert "truncated" in truncated
