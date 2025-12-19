from __future__ import annotations

from pathlib import Path

from hafs.agents.coordinator import AgentCoordinator
from hafs.config.schema import AFSDirectoryConfig, GeneralConfig, PolicyType
from hafs.context.builder import ContextPromptBuilder
from hafs.core.afs.policy import PolicyEnforcer
from hafs.models.afs import MountType
from hafs.models.agent import SharedContext


def test_shared_context_tracks_unique_paths(tmp_path: Path) -> None:
    """Context items should be unique and limited."""
    ctx = SharedContext()
    paths = [tmp_path / "a.txt", tmp_path / "a.txt", tmp_path / "b.txt"]

    stored = ctx.set_context_items(paths)

    assert stored == [tmp_path / "a.txt", tmp_path / "b.txt"]
    assert ctx.context_item_count == 2


def test_prompt_builder_includes_context_items(tmp_path: Path) -> None:
    """Pinned context paths should appear in prompts."""
    builder = ContextPromptBuilder()
    ctx = SharedContext()
    ctx.set_context_items([tmp_path / "note.md"])

    prompt = builder.build("do work", shared_context=ctx)

    assert "PINNED CONTEXT ITEMS" in prompt
    assert "note.md" in prompt


def test_policy_enforcer_blocks_writes() -> None:
    """PolicyEnforcer should enforce per-mount permissions."""
    directories = [
        AFSDirectoryConfig(name="memory", policy=PolicyType.READ_ONLY),
        AFSDirectoryConfig(name="tools", policy=PolicyType.EXECUTABLE),
    ]

    enforcer = PolicyEnforcer(directories)

    allowed, error = enforcer.validate_operation(MountType.MEMORY, "write")
    assert not allowed
    assert "read_only" in error

    assert enforcer.can_execute(MountType.TOOLS)


def test_general_config_has_workspace_defaults() -> None:
    """Defaults should include helpful workspace roots."""
    cfg = GeneralConfig()
    assert len(cfg.workspace_directories) >= 2
    assert cfg.workspace_directories[0].name == "Code"


def test_coordinator_sets_context_items(tmp_path: Path) -> None:
    """Coordinator should push pinned context into shared context."""
    coord = AgentCoordinator({"default_backend": "stub", "enabled_backends": ["stub"]})
    paths = [tmp_path / "notes.md", tmp_path / "docs" / "guide.md"]

    applied = coord.set_context_items(paths)

    assert applied == coord.shared_context.context_items
    assert coord.shared_context.context_item_count == 2
