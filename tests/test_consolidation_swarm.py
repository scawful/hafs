import pytest
import warnings
from agents.swarm import (
    SwarmCouncil,
    SwarmStatus,
    SwarmStrategist,
    CouncilReviewer,
    DeepDiveDocumenter,
)

def test_canonical_imports():
    """Verify swarm agents can be imported from the new canonical path."""
    assert SwarmCouncil is not None
    assert SwarmStatus is not None
    assert SwarmStrategist is not None
    assert CouncilReviewer is not None
    assert DeepDiveDocumenter is not None

def test_legacy_imports():
    """Verify legacy imports still work and emit DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        from hafs.agents.specialists import SwarmStrategist as LegacyStrategist
        from hafs.agents.swarm import SwarmCouncil as LegacyCouncil
        
        assert LegacyStrategist is SwarmStrategist
        assert LegacyCouncil is SwarmCouncil
        
        # Verify warnings were emitted
        assert len(w) >= 2
        assert any("hafs.agents.specialists is deprecated" in str(warning.message) for warning in w)
        assert any("hafs.agents.swarm is deprecated" in str(warning.message) for warning in w)

if __name__ == "__main__":
    pytest.main([__file__])
