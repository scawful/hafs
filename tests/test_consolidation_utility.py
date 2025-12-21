import pytest
import warnings
from agents.utility import (
    CartographerAgent,
    ChronosAgent,
    ContextVectorAgent,
    DailyBriefingAgent,
    DistributedObservabilityAgent,
    EpisodicMemoryAgent,
    GardenerAgent,
    GeminiHistorianAgent,
    HistoryPipelineAgent,
    MonitorAgent,
    PromptEngineerAgent,
    ReportManagerAgent,
    ScoutAgent,
    ShadowObserver,
    ShellAgent,
    ToolsmithAgent,
    TrendWatcherAgent,
    VisualizerAgent,
)

def test_canonical_imports():
    """Verify utility agents can be imported from the new canonical path."""
    assert CartographerAgent is not None
    assert ChronosAgent is not None
    assert ContextVectorAgent is not None
    assert DailyBriefingAgent is not None
    assert DistributedObservabilityAgent is not None
    assert EpisodicMemoryAgent is not None
    assert GardenerAgent is not None
    assert GeminiHistorianAgent is not None
    assert HistoryPipelineAgent is not None
    assert MonitorAgent is not None
    assert PromptEngineerAgent is not None
    assert ReportManagerAgent is not None
    assert ScoutAgent is not None
    assert ShadowObserver is not None
    assert ShellAgent is not None
    assert ToolsmithAgent is not None
    assert TrendWatcherAgent is not None
    assert VisualizerAgent is not None

def test_legacy_imports():
    """Verify legacy imports still work and emit DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        from hafs.agents.history_pipeline import HistoryPipelineAgent as LegacyHistory
        from hafs.agents.observability import DistributedObservabilityAgent as LegacyObs
        from hafs.agents.cartographer import CartographerAgent as LegacyCart
        
        assert LegacyHistory is HistoryPipelineAgent
        assert LegacyObs is DistributedObservabilityAgent
        assert LegacyCart is CartographerAgent
        
        # Verify warnings were emitted
        assert len(w) >= 3
        assert any("hafs.agents.history_pipeline is deprecated" in str(warning.message) for warning in w)
        assert any("hafs.agents.observability is deprecated" in str(warning.message) for warning in w)
        assert any("hafs.agents.cartographer is deprecated" in str(warning.message) for warning in w)

if __name__ == "__main__":
    pytest.main([__file__])
