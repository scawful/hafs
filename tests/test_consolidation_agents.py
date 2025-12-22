import sys
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
    """Verify that agents still exports core agents with warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for name in list(sys.modules):
            if name.startswith("agents"):
                sys.modules.pop(name)
        from agents import (
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
        # But we expect at least one warning from the agents module.
        assert any("agents is deprecated" in str(warn.message) for warn in w)

def test_backward_compat_core_modules():
    """Verify that individual legacy modules work with warnings."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for name in list(sys.modules):
            if name.startswith("agents"):
                sys.modules.pop(name)
        from agents.core.base import BaseAgent
        from agents.core.coordinator import AgentCoordinator
        from agents.core.lane import AgentLane
        from agents.core.router import MentionRouter
        from agents.core.roles import ROLE_DESCRIPTIONS
        
        assert BaseAgent is not None
        assert AgentCoordinator is not None
        assert AgentLane is not None
        assert MentionRouter is not None
        assert ROLE_DESCRIPTIONS is not None
        
        # Check for multiple DeprecationWarnings
        messages = [str(warn.message) for warn in w]
        assert any("agents.core.base is deprecated" in msg for msg in messages)
        assert any("agents.core.coordinator is deprecated" in msg for msg in messages)
        assert any("agents.core.lane is deprecated" in msg for msg in messages)
        assert any("agents.core.router is deprecated" in msg for msg in messages)
        assert any("agents.core.roles is deprecated" in msg for msg in messages)

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
