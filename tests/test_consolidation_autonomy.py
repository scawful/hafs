import warnings
import pytest
from pathlib import Path

# New canonical imports
def test_new_autonomy_imports():
    """Verify that new canonical imports for autonomy agents work."""
    from agents.autonomy import (
        LoopReport,
        MemoryAwareAgent,
        SelfImprovementAgent,
        CuriosityExplorerAgent,
        SelfHealingAgent,
        HallucinationWatcherAgent,
    )
    assert LoopReport is not None
    assert MemoryAwareAgent is not None
    assert SelfImprovementAgent is not None
    assert CuriosityExplorerAgent is not None
    assert SelfHealingAgent is not None
    assert HallucinationWatcherAgent is not None

def test_autonomy_top_level_reexports():
    """Verify that top-level agents package re-exports autonomy agents."""
    from agents import (
        CuriosityExplorerAgent,
        HallucinationWatcherAgent,
        LoopReport,
        SelfHealingAgent,
        SelfImprovementAgent,
    )
    assert CuriosityExplorerAgent is not None
    assert HallucinationWatcherAgent is not None
    assert LoopReport is not None
    assert SelfHealingAgent is not None
    assert SelfImprovementAgent is not None

# Backward compatibility imports
def test_backward_compat_autonomy():
    """Verify that agents.autonomy re-exports with warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from agents.autonomy import (
            SelfImprovementAgent,
            CuriosityExplorerAgent,
            SelfHealingAgent,
            HallucinationWatcherAgent,
            LoopReport,
        )
        
        assert SelfImprovementAgent is not None
        assert CuriosityExplorerAgent is not None
        assert SelfHealingAgent is not None
        assert HallucinationWatcherAgent is not None
        assert LoopReport is not None
        
        # Check for DeprecationWarning
        assert any("agents.autonomy is deprecated" in str(warn.message) for warn in w)

if __name__ == "__main__":
    # Simple manual run
    try:
        test_new_autonomy_imports()
        test_autonomy_top_level_reexports()
        test_backward_compat_autonomy()
        print("All autonomy consolidation tests passed!")
    except Exception as e:
        print(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()
