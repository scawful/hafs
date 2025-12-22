"""Mixture of Experts system for ROM hacking."""

from agents.moe.classifier import Classification, TaskClassifier
from agents.moe.config import (
    EXPERT_PRESETS,
    ExpertPreset,
    TempPreset,
    TokenPreset,
    get_preset,
    list_presets,
)
from agents.moe.expert import BaseExpert, ExpertConfig, ExpertResponse
from agents.moe.experts.asm_expert import AsmExpert
from agents.moe.experts.debug_expert import DebugExpert
from agents.moe.experts.registry_expert import RegistryExpert
from agents.moe.experts.yaze_expert import YazeExpert
from agents.moe.orchestrator import MoEOrchestrator, MoEResult
from agents.moe.registry import ModelRegistry, RoutingTable
from agents.moe.synthesizer import Synthesizer

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
