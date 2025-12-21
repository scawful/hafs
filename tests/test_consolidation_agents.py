import warnings
import pytest
from pathlib import Path

# New canonical imports
def test_new_agent_core_imports():
    """Verify that new canonical imports for core agents work."""
    from agents.core import (
        BaseAgent,
        AgentCoordinator,
        CoordinatorMode,
        AgentLane,
        AgentLaneManager,
        MentionRouter,
        ROLE_DESCRIPTIONS,
    )
    assert BaseAgent is not None
    assert AgentCoordinator is not None
    assert CoordinatorMode is not None
    assert AgentLane is not None
    assert AgentLaneManager is not None
    assert MentionRouter is not None
    assert ROLE_DESCRIPTIONS is not None

def test_new_agent_top_level_imports():
    """Verify that top-level agents package re-exports core agents."""
    from agents import (
        BaseAgent,
        AgentCoordinator,
        CoordinatorMode,
        AgentLane,
        MentionRouter,
    )
    assert BaseAgent is not None
    assert AgentCoordinator is not None
    assert CoordinatorMode is not None
    assert AgentLane is not None
    assert MentionRouter is not None

# Backward compatibility imports
def test_backward_compat_agents_init():
    """Verify that hafs.agents still exports core agents with warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from hafs.agents import (
            AgentCoordinator,
            CoordinatorMode,
            AgentLane,
            AgentLaneManager,
            MentionRouter,
        )
        
        assert AgentCoordinator is not None
        assert CoordinatorMode is not None
        assert AgentLane is not None
        assert AgentLaneManager is not None
        assert MentionRouter is not None
        
        # Check for DeprecationWarning
        # Note: Depending on how many times it was already imported in the session,
        # we might get multiple or zero warnings if not filtered correctly.
        # But we expect at least one warning from the hafs.agents module.
        assert any("hafs.agents is deprecated" in str(warn.message) for warn in w)

def test_backward_compat_core_modules():
    """Verify that individual legacy modules work with warnings."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from hafs.agents.base import BaseAgent
        from hafs.agents.coordinator import AgentCoordinator
        from hafs.agents.lane import AgentLane
        from hafs.agents.router import MentionRouter
        from hafs.agents.roles import ROLE_DESCRIPTIONS
        
        assert BaseAgent is not None
        assert AgentCoordinator is not None
        assert AgentLane is not None
        assert MentionRouter is not None
        assert ROLE_DESCRIPTIONS is not None
        
        # Check for multiple DeprecationWarnings
        messages = [str(warn.message) for warn in w]
        assert any("hafs.agents.base is deprecated" in msg for msg in messages)
        assert any("hafs.agents.coordinator is deprecated" in msg for msg in messages)
        assert any("hafs.agents.lane is deprecated" in msg for msg in messages)
        assert any("hafs.agents.router is deprecated" in msg for msg in messages)
        assert any("hafs.agents.roles is deprecated" in msg for msg in messages)

if __name__ == "__main__":
    # Simple manual run
    try:
        test_new_agent_core_imports()
        test_new_agent_top_level_imports()
        test_backward_compat_agents_init()
        test_backward_compat_core_modules()
        print("All consolidation tests passed!")
    except Exception as e:
        print(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()
