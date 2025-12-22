import pytest
import sys
import warnings
from agents.analysis import (
    CodeDescriber,
    AutonomousContextAgent,
    EmbeddingAnalyzer,
    ContextReportPipeline,
)

def test_canonical_imports():
    """Verify analysis agents can be imported from the new canonical path."""
    assert CodeDescriber is not None
    assert AutonomousContextAgent is not None
    assert EmbeddingAnalyzer is not None
    assert ContextReportPipeline is not None

def test_legacy_imports():
    """Verify legacy imports still work and emit DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for name in list(sys.modules):
            if name.startswith("agents"):
                sys.modules.pop(name)
        
        from agents.analysis.code_describer import CodeDescriber as LegacyCodeDescriber
        from agents.analysis.context_builder import AutonomousContextAgent as LegacyContextAgent
        from agents.analysis.embedding_analyzer import EmbeddingAnalyzer as LegacyEmbeddingAnalyzer
        from agents.analysis.report_pipeline import ContextReportPipeline as LegacyPipeline
        
        assert LegacyCodeDescriber is CodeDescriber
        assert LegacyContextAgent is AutonomousContextAgent
        assert LegacyEmbeddingAnalyzer is EmbeddingAnalyzer
        assert LegacyPipeline is ContextReportPipeline
        
        # Verify warnings were emitted
        assert len(w) >= 4
        assert any("agents.analysis.code_describer is deprecated" in str(warning.message) for warning in w)
        assert any("agents.analysis.context_builder is deprecated" in str(warning.message) for warning in w)
        assert any("agents.analysis.embedding_analyzer is deprecated" in str(warning.message) for warning in w)
        assert any("agents.analysis.report_pipeline is deprecated" in str(warning.message) for warning in w)

if __name__ == "__main__":
    pytest.main([__file__])
