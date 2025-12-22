import unittest
import warnings
from pathlib import Path

# New canonical imports
from agents.pipeline import (
    ArchitectCouncil,
    BuilderCouncil,
    ValidatorCouncil,
    CodeWriter,
    DocWriter,
    TestWriter,
    BuildAgent,
    TestAgent,
    ReviewUploader,
)

# Legacy imports
import agents.pipeline.architect_council as legacy_architect
import agents.pipeline.builder_council as legacy_builder
import agents.pipeline.validator_council as legacy_validator
import agents.pipeline.code_writer as legacy_code_writer
import agents.pipeline.doc_writer as legacy_doc_writer
import agents.pipeline.test_writer as legacy_test_writer
import agents.pipeline.build_test_agents as legacy_build_test
import agents.pipeline.review_uploader as legacy_uploader
import agents.pipeline.advanced_agents as legacy_advanced

class TestPipelineConsolidation(unittest.TestCase):
    """Verify Phase 6: Pipeline Agents consolidation."""

    def test_canonical_imports(self):
        """Verify that agents can be imported from their new canonical paths."""
        self.assertIsNotNone(ArchitectCouncil)
        self.assertIsNotNone(BuilderCouncil)
        self.assertIsNotNone(ValidatorCouncil)
        self.assertIsNotNone(CodeWriter)
        self.assertIsNotNone(DocWriter)
        self.assertIsNotNone(TestWriter)
        self.assertIsNotNone(BuildAgent)
        self.assertIsNotNone(TestAgent)
        self.assertIsNotNone(ReviewUploader)

    def test_legacy_reexports(self):
        """Verify that legacy modules correctly re-export from new locations."""
        self.Is(legacy_architect.ArchitectCouncil, ArchitectCouncil)
        self.Is(legacy_builder.BuilderCouncil, BuilderCouncil)
        self.Is(legacy_validator.ValidatorCouncil, ValidatorCouncil)
        self.Is(legacy_code_writer.CodeWriter, CodeWriter)
        self.Is(legacy_doc_writer.DocWriter, DocWriter)
        self.Is(legacy_test_writer.TestWriter, TestWriter)
        self.Is(legacy_build_test.BuildAgent, BuildAgent)
        self.Is(legacy_build_test.TestAgent, TestAgent)
        self.Is(legacy_uploader.ReviewUploader, ReviewUploader)
        self.assertIsNotNone(legacy_advanced.StaticAnalysisAgent)

    def test_deprecation_warnings(self):
        """Verify that legacy imports emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Re-import to trigger warning
            import importlib
            import agents.pipeline.architect_council
            importlib.reload(agents.pipeline.architect_council)
            
            self.assertTrue(len(w) > 0)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))
            self.assertIn("deprecated", str(w[-1].message))

    def Is(self, a, b):
        """Helper for identity check."""
        self.assertTrue(a is b)

if __name__ == "__main__":
    unittest.main()
