import warnings
from agents.pipeline.build_test_agents import BuildAgent, TestAgent

warnings.warn(
    "hafs.agents.pipeline.build_test_agents is deprecated. Import from 'agents.pipeline.build_test_agents' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export
BuildAgent = BuildAgent
TestAgent = TestAgent
