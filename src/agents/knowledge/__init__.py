"""Knowledge agents for hafs.

This package contains specialized agents for knowledge base management,
disassembly analysis, and ROM hacking.
"""

from agents.knowledge.alttp import ALTTPKnowledgeBase
from agents.knowledge.alttp_multi import ALTTPMultiKBManager
from agents.knowledge.alttp_unified import UnifiedALTTPKnowledge
from agents.knowledge.alttp_embeddings import ALTTPEmbeddingSpecialist
from agents.knowledge.alttp_analyzer import ALTTPModuleAnalyzer
from agents.knowledge.oracle import OracleKnowledgeBase, OracleKBBuilder
from agents.knowledge.oracle_analyzer import OracleOfSecretsAnalyzer
from agents.knowledge.gigaleak import GigaleakKB
from agents.knowledge.graph import KnowledgeGraphAgent
from agents.knowledge.enhancer import KBEnhancer
from agents.knowledge.rom import RomHackingSpecialist

__all__ = [
    "ALTTPKnowledgeBase",
    "ALTTPMultiKBManager",
    "UnifiedALTTPKnowledge",
    "ALTTPEmbeddingSpecialist",
    "ALTTPModuleAnalyzer",
    "OracleKnowledgeBase",
    "OracleKBBuilder",
    "OracleOfSecretsAnalyzer",
    "GigaleakKB",
    "KnowledgeGraphAgent",
    "KBEnhancer",
    "RomHackingSpecialist",
]
