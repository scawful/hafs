#!/usr/bin/env python3
"""Test script to verify agents module imports."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all agent module components can be imported."""
    print("Testing agent module imports...")

    # Test model imports
    print("  - Importing agent models...")
    from hafs.models.agent import Agent, AgentMessage, AgentRole, SharedContext
    print("    ✓ Agent models imported successfully")

    # Test roles module
    print("  - Importing roles module...")
    from hafs.agents.roles import (
        ROLE_DESCRIPTIONS,
        ROLE_KEYWORDS,
        get_role_system_prompt,
        match_role_by_keywords,
    )
    print("    ✓ Roles module imported successfully")

    # Test router module
    print("  - Importing router module...")
    from hafs.agents.router import MentionRouter
    print("    ✓ Router module imported successfully")

    # Test lane module
    print("  - Importing lane module...")
    from hafs.agents.lane import AgentLane, AgentLaneManager
    print("    ✓ Lane module imported successfully")

    # Test coordinator module
    print("  - Importing coordinator module...")
    from hafs.agents.coordinator import AgentCoordinator
    print("    ✓ Coordinator module imported successfully")

    # Test main agents package
    print("  - Importing main agents package...")
    from hafs.agents import (
        AgentCoordinator,
        AgentLane,
        MentionRouter,
        AgentRole,
    )
    print("    ✓ Main agents package imported successfully")

    print("\n✓ All imports successful!")

    # Test basic functionality
    print("\nTesting basic functionality...")

    # Test SharedContext
    context = SharedContext()
    context.add_finding("Test finding")
    context.add_decision("Test decision")
    context.active_task = "Test task"
    prompt_text = context.to_prompt_text()
    assert "Test finding" in prompt_text
    assert "Test decision" in prompt_text
    assert "Test task" in prompt_text
    print("  ✓ SharedContext works correctly")

    # Test MentionRouter
    router = MentionRouter()
    mentions = router.extract_mentions("@planner create a roadmap")
    assert mentions == ["planner"]
    cleaned = router.strip_mentions("@planner create a roadmap")
    assert "@planner" not in cleaned
    print("  ✓ MentionRouter works correctly")

    # Test role matching
    role = match_role_by_keywords("implement a login function")
    assert role == AgentRole.CODER
    print("  ✓ Role matching works correctly")

    # Test role system prompt
    prompt = get_role_system_prompt(AgentRole.PLANNER)
    assert "planning" in prompt.lower()
    print("  ✓ Role system prompts work correctly")

    print("\n✓ All tests passed!")

if __name__ == "__main__":
    try:
        test_imports()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
