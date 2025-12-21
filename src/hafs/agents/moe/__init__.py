"""Mixture of Experts system for ROM hacking."""

from hafs.agents.moe.classifier import Classification, TaskClassifier
from hafs.agents.moe.config import (
    EXPERT_PRESETS,
    ExpertPreset,
    TempPreset,
    TokenPreset,
    get_preset,
    list_presets,
)
from hafs.agents.moe.expert import BaseExpert, ExpertConfig, ExpertResponse
from hafs.agents.moe.experts.asm_expert import AsmExpert
from hafs.agents.moe.experts.debug_expert import DebugExpert
from hafs.agents.moe.experts.registry_expert import RegistryExpert
from hafs.agents.moe.experts.yaze_expert import YazeExpert
from hafs.agents.moe.orchestrator import MoEOrchestrator, MoEResult
from hafs.agents.moe.registry import ModelRegistry, RoutingTable
from hafs.agents.moe.synthesizer import Synthesizer

__all__ = [
    # Core components
    "MoEOrchestrator",
    "MoEResult",
    "TaskClassifier",
    "Classification",
    "Synthesizer",
    # Expert base
    "BaseExpert",
    "ExpertConfig",
    "ExpertResponse",
    # Concrete experts
    "AsmExpert",
    "YazeExpert",
    "DebugExpert",
    "RegistryExpert",
    # Registry + routing
    "ModelRegistry",
    "RoutingTable",
    # Configuration presets
    "TempPreset",
    "TokenPreset",
    "ExpertPreset",
    "EXPERT_PRESETS",
    "get_preset",
    "list_presets",
]
