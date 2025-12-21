"""Legacy path for autonomy agents. Use agents.autonomy instead."""
import warnings
from agents.autonomy import (
    CuriosityExplorerAgent,
    HallucinationWatcherAgent,
    LoopReport,
    SelfHealingAgent,
    SelfImprovementAgent,
    SwarmLogMonitorAgent,
    ContextDiscoveryAgent,
    TestRunnerAgent,
    QualityAuditAgent,
)

warnings.warn(
    "hafs.agents.autonomy_agents is deprecated. Import from 'agents.autonomy' instead.",
    DeprecationWarning,
    stacklevel=2
)
