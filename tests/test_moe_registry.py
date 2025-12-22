"""Tests for MoE routing table and model registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.moe.classifier import TaskClassifier
from agents.moe.orchestrator import MoEOrchestrator
from agents.moe.registry import ModelRegistry, RoutingTable


ROUTING_TOML = """
[metadata]
schema_version = 1
updated = "2025-01-15"
default_experts = ["oracle-robbie-toolsmith"]

[[routes]]
name = "oracle-of-secrets.lore"
experts = ["oracle-nayru-canon"]
keywords = ["lore", "canon"]
match = "any"
priority = 90
"""

REGISTRY_TOML = """
[metadata]
schema_version = 1
updated = "2025-01-15"

[models.oracle-nayru-canon]
display_name = "Oracle: Nayru Canon"
role = "lore"
group = "oracle-of-secrets"
base = "gemma3-12b-it"
status = "planned"
default_provider = "llamacpp"
tags = ["oracle-of-secrets", "lore"]
notes = "Lore bible, timeline sanity, and continuity checks."
"""


def _write_tmp(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content)
    return path


def test_routing_table_match(tmp_path: Path) -> None:
    routing_path = _write_tmp(tmp_path, "routing.toml", ROUTING_TOML)
    table = RoutingTable.load(routing_path)

    decision = table.match_intent("Need lore canon validation for the plot")
    assert decision is not None
    assert decision.experts[0] == "oracle-nayru-canon"
    assert decision.confidences[0] > 0.0


def test_model_registry_load(tmp_path: Path) -> None:
    registry_path = _write_tmp(tmp_path, "registry.toml", REGISTRY_TOML)
    registry = ModelRegistry.load(registry_path)

    record = registry.get("oracle-nayru-canon")
    assert record is not None
    assert record.display_name == "Oracle: Nayru Canon"
    assert record.default_provider == "llamacpp"


@pytest.mark.asyncio
async def test_classifier_uses_routing_table(tmp_path: Path) -> None:
    routing_path = _write_tmp(tmp_path, "routing.toml", ROUTING_TOML)
    registry_path = _write_tmp(tmp_path, "registry.toml", REGISTRY_TOML)

    routing = RoutingTable.load(routing_path)
    registry = ModelRegistry.load(registry_path)

    classifier = TaskClassifier(routing_table=routing, model_registry=registry)
    classifier.initialized = True

    result = await classifier.classify("We need lore canon checks")
    assert result.primary_expert == "oracle-nayru-canon"


def test_orchestrator_registers_registry_experts(tmp_path: Path) -> None:
    routing_path = _write_tmp(tmp_path, "routing.toml", ROUTING_TOML)
    registry_path = _write_tmp(tmp_path, "registry.toml", REGISTRY_TOML)

    routing = RoutingTable.load(routing_path)
    registry = ModelRegistry.load(registry_path)

    orchestrator = MoEOrchestrator(model_registry=registry, routing_table=routing)
    assert "oracle-nayru-canon" in orchestrator.experts
