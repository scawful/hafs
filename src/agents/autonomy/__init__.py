"""Autonomy agents for background improvement and safety loops."""

from agents.autonomy.base import LoopReport, MemoryAwareAgent
from agents.autonomy.curiosity import CuriosityExplorerAgent
from agents.autonomy.hallucination import HallucinationWatcherAgent
from agents.autonomy.self_healing import SelfHealingAgent
from agents.autonomy.self_improvement import SelfImprovementAgent
from agents.autonomy.swarm_watch import SwarmLogMonitorAgent
from agents.autonomy.maintenance import (
    ContextDiscoveryAgent,
    TestRunnerAgent,
    QualityAuditAgent,
)

__all__ = [
    "LoopReport",
    "MemoryAwareAgent",
    "CuriosityExplorerAgent",
    "HallucinationWatcherAgent",
    "SelfHealingAgent",
    "SelfImprovementAgent",
    "SwarmLogMonitorAgent",
    "ContextDiscoveryAgent",
    "TestRunnerAgent",
    "QualityAuditAgent",
]
