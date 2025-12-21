import warnings
from agents.pipeline.advanced_agents import (
    StaticAnalysisAgent,
    CodeReviewerAgent,
    IntegrationTestWriter,
    ProjectManagerAgent,
    RolloutManagerAgent,
    MetricsWatcherAgent,
)

warnings.warn(
    "hafs.agents.pipeline.advanced_agents is deprecated. Import from 'agents.pipeline.advanced_agents' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export
StaticAnalysisAgent = StaticAnalysisAgent
CodeReviewerAgent = CodeReviewerAgent
IntegrationTestWriter = IntegrationTestWriter
ProjectManagerAgent = ProjectManagerAgent
RolloutManagerAgent = RolloutManagerAgent
MetricsWatcherAgent = MetricsWatcherAgent
