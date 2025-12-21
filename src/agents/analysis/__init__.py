"""Analysis Agents Package.

Provides tools for codebase analysis, context building, and embedding inspection.
"""

from agents.analysis.code_describer import (
    CodeDescriber,
    CodeKnowledgeBase,
    CodeUnit,
    LanguagePlugin,
    register_plugin,
    get_plugin,
    detect_language,
)
from agents.analysis.context_builder import AutonomousContextAgent
from agents.analysis.deep_context_pipeline import (
    ContextSignalAgent,
    DeepContextPipeline,
    MLSignalAgent,
    MLPipelinePlannerAgent,
    RepoSnapshotAgent,
    SmartMLPipeline,
)
from agents.analysis.embedding_analyzer import EmbeddingAnalyzer
from agents.analysis.report_pipeline import ContextReportPipeline

__all__ = [
    "CodeDescriber",
    "CodeKnowledgeBase",
    "CodeUnit",
    "LanguagePlugin",
    "register_plugin",
    "get_plugin",
    "detect_language",
    "AutonomousContextAgent",
    "EmbeddingAnalyzer",
    "ContextReportPipeline",
    "RepoSnapshotAgent",
    "ContextSignalAgent",
    "MLSignalAgent",
    "MLPipelinePlannerAgent",
    "DeepContextPipeline",
    "SmartMLPipeline",
]
