import pytest
import warnings
from agents.mission import (
    ResearchMission,
    ResearchDiscovery,
    MissionAgent,
    ALTTPResearchAgent,
    GigaleakAnalysisAgent,
    get_mission_agent,
    DEFAULT_MISSIONS,
)

def test_canonical_imports():
    """Verify mission agents can be imported from the new canonical path."""
    assert ResearchMission is not None
    assert ResearchDiscovery is not None
    assert MissionAgent is not None
    assert ALTTPResearchAgent is not None
    assert GigaleakAnalysisAgent is not None
    assert get_mission_agent is not None
    assert len(DEFAULT_MISSIONS) > 0

def test_legacy_imports():
    """Verify legacy imports still work and emit DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        from hafs.agents.mission_agents import ALTTPResearchAgent as LegacyAgent
        from hafs.agents.mission_agents import DEFAULT_MISSIONS as LegacyMissions
        
        assert LegacyAgent is ALTTPResearchAgent
        assert LegacyMissions is DEFAULT_MISSIONS
        
        # Verify warnings were emitted
        assert len(w) >= 2
        assert any("hafs.agents.mission_agents is deprecated" in str(warning.message) for warning in w)

if __name__ == "__main__":
    pytest.main([__file__])
