import unittest
import warnings
from pathlib import Path

# New canonical imports
from agents.knowledge import (
    ALTTPKnowledgeBase,
    ALTTPMultiKBManager,
    UnifiedALTTPKnowledge,
    ALTTPEmbeddingSpecialist,
    ALTTPModuleAnalyzer,
    OracleKnowledgeBase,
    OracleKBBuilder,
    OracleOfSecretsAnalyzer,
    GigaleakKB,
    KnowledgeGraphAgent,
    KBEnhancer,
    RomHackingSpecialist,
)

# Legacy imports
import hafs.agents.alttp_knowledge as legacy_alttp
import hafs.agents.alttp_multi_kb as legacy_multi
import hafs.agents.alttp_unified_kb as legacy_unified
import hafs.agents.alttp_embeddings as legacy_embeddings
import hafs.agents.alttp_module_analyzer as legacy_analyzer
import hafs.agents.oracle_kb_builder as legacy_oracle
import hafs.agents.oracle_analyzer as legacy_oracle_analyzer
import hafs.agents.gigaleak_kb as legacy_gigaleak
import hafs.agents.knowledge_graph as legacy_graph
import hafs.agents.kb_enhancer as legacy_enhancer
import hafs.agents.rom_specialist as legacy_rom

class TestKnowledgeConsolidation(unittest.TestCase):
    """Verify Phase 5: Knowledge Agents consolidation."""

    def test_canonical_imports(self):
        """Verify that agents can be imported from their new canonical paths."""
        self.assertIsNotNone(ALTTPKnowledgeBase)
        self.assertIsNotNone(ALTTPMultiKBManager)
        self.assertIsNotNone(UnifiedALTTPKnowledge)
        self.assertIsNotNone(ALTTPEmbeddingSpecialist)
        self.assertIsNotNone(ALTTPModuleAnalyzer)
        self.assertIsNotNone(OracleKnowledgeBase)
        self.assertIsNotNone(OracleKBBuilder)
        self.assertIsNotNone(OracleOfSecretsAnalyzer)
        self.assertIsNotNone(GigaleakKB)
        self.assertIsNotNone(KnowledgeGraphAgent)
        self.assertIsNotNone(KBEnhancer)
        self.assertIsNotNone(RomHackingSpecialist)

    def test_legacy_reexports(self):
        """Verify that legacy modules correctly re-export from new locations."""
        self.Is(legacy_alttp.ALTTPKnowledgeBase, ALTTPKnowledgeBase)
        self.Is(legacy_multi.ALTTPMultiKBManager, ALTTPMultiKBManager)
        self.Is(legacy_unified.UnifiedALTTPKnowledge, UnifiedALTTPKnowledge)
        self.Is(legacy_embeddings.ALTTPEmbeddingSpecialist, ALTTPEmbeddingSpecialist)
        self.Is(legacy_analyzer.ALTTPModuleAnalyzer, ALTTPModuleAnalyzer)
        self.Is(legacy_oracle.OracleKnowledgeBase, OracleKnowledgeBase)
        self.Is(legacy_oracle.OracleKBBuilder, OracleKBBuilder)
        self.Is(legacy_oracle_analyzer.OracleOfSecretsAnalyzer, OracleOfSecretsAnalyzer)
        self.Is(legacy_gigaleak.GigaleakKB, GigaleakKB)
        self.Is(legacy_graph.KnowledgeGraphAgent, KnowledgeGraphAgent)
        self.Is(legacy_enhancer.KBEnhancer, KBEnhancer)
        self.Is(legacy_rom.RomHackingSpecialist, RomHackingSpecialist)

    def test_deprecation_warnings(self):
        """Verify that legacy imports emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Re-import to trigger warning
            import importlib
            import hafs.agents.alttp_knowledge
            importlib.reload(hafs.agents.alttp_knowledge)
            
            self.assertTrue(len(w) > 0)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))
            self.assertIn("deprecated", str(w[-1].message))

    def Is(self, a, b):
        """Helper for identity check."""
        self.assertTrue(a is b)

if __name__ == "__main__":
    unittest.main()
