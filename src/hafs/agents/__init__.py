import importlib
import warnings

_DEPRECATION_MESSAGE = "agents is deprecated. Import from 'agents' instead."

warnings.warn(
    _DEPRECATION_MESSAGE,
    DeprecationWarning,
    stacklevel=2,
)

_EXPORTS = [
    ("AgentCoordinator", "agents.core.coordinator", "AgentCoordinator"),
    ("CoordinatorMode", "agents.core.coordinator", "CoordinatorMode"),
    ("AgentLane", "agents.core.lane", "AgentLane"),
    ("AgentLaneManager", "agents.core.lane", "AgentLaneManager"),
    ("MentionRouter", "agents.core.router", "MentionRouter"),
    ("ROLE_DESCRIPTIONS", "agents.core.roles", "ROLE_DESCRIPTIONS"),
    ("ROLE_KEYWORDS", "agents.core.roles", "ROLE_KEYWORDS"),
    ("get_role_keywords", "agents.core.roles", "get_role_keywords"),
    ("get_role_system_prompt", "agents.core.roles", "get_role_system_prompt"),
    ("match_role_by_keywords", "agents.core.roles", "match_role_by_keywords"),
    ("Agent", "models.agent", "Agent"),
    ("AgentMessage", "models.agent", "AgentMessage"),
    ("AgentRole", "models.agent", "AgentRole"),
    ("SharedContext", "models.agent", "SharedContext"),
    ("HistoryPipelineAgent", "agents.utility.history_pipeline", "HistoryPipelineAgent"),
    ("with_history_logging", "agents.utility.history_pipeline", "with_history_logging"),
    ("RomHackingSpecialist", "agents.knowledge.rom", "RomHackingSpecialist"),
    ("DistributedObservabilityAgent", "agents.utility.observability", "DistributedObservabilityAgent"),
    ("ContextReportPipeline", "agents.analysis.report_pipeline", "ContextReportPipeline"),
    ("EmbeddingResearchAgent", "agents.analysis.report_pipeline", "EmbeddingResearchAgent"),
    ("AnalysisAgent", "agents.analysis.report_pipeline", "AnalysisAgent"),
    ("SynthesisAgent", "agents.analysis.report_pipeline", "SynthesisAgent"),
    ("ReviewerAgent", "agents.analysis.report_pipeline", "ReviewerAgent"),
    ("ResearchContext", "agents.analysis.report_pipeline", "ResearchContext"),
    ("ALTTPModuleAnalyzer", "agents.knowledge.alttp_analyzer", "ALTTPModuleAnalyzer"),
    ("OracleKBBuilder", "agents.knowledge.oracle", "OracleKBBuilder"),
    ("OracleKnowledgeBase", "agents.knowledge.oracle", "OracleKnowledgeBase"),
    ("OracleOfSecretsAnalyzer", "agents.knowledge.oracle_analyzer", "OracleOfSecretsAnalyzer"),
    ("ReportManager", "agents.utility.report_manager", "ReportManager"),
    ("KnowledgeGraphAgent", "agents.knowledge.graph", "KnowledgeGraphAgent"),
    ("KBEnhancer", "agents.knowledge.enhancer", "KBEnhancer"),
    ("GigaleakKB", "agents.knowledge.gigaleak", "GigaleakKB"),
    ("ALTTPKnowledgeBase", "agents.knowledge.alttp", "ALTTPKnowledgeBase"),
    ("ALTTPMultiKBManager", "agents.knowledge.alttp_multi", "ALTTPMultiKBManager"),
    ("UnifiedALTTPKnowledge", "agents.knowledge.alttp_unified", "UnifiedALTTPKnowledge"),
    ("OracleOfSecretsKB", "agents.knowledge.alttp_unified", "OracleOfSecretsKB"),
    ("UnifiedSearchResult", "agents.knowledge.alttp_unified", "UnifiedSearchResult"),
    ("ALTTPEmbeddingSpecialist", "agents.knowledge.alttp_embeddings", "ALTTPEmbeddingSpecialist"),
    ("CodeDescriber", "agents.analysis.code_describer", "CodeDescriber"),
    ("CodeKnowledgeBase", "agents.analysis.code_describer", "CodeKnowledgeBase"),
    ("CodeUnit", "agents.analysis.code_describer", "CodeUnit"),
    ("LanguagePlugin", "agents.analysis.code_describer", "LanguagePlugin"),
    ("register_plugin", "agents.analysis.code_describer", "register_plugin"),
    ("get_plugin", "agents.analysis.code_describer", "get_plugin"),
    ("detect_language", "agents.analysis.code_describer", "detect_language"),
    ("SelfImprovementAgent", "agents.autonomy.self_improvement", "SelfImprovementAgent"),
    ("CuriosityExplorerAgent", "agents.autonomy.curiosity", "CuriosityExplorerAgent"),
    ("SelfHealingAgent", "agents.autonomy.self_healing", "SelfHealingAgent"),
    ("HallucinationWatcherAgent", "agents.autonomy.hallucination", "HallucinationWatcherAgent"),
    ("SwarmLogMonitorAgent", "agents.autonomy.swarm_watch", "SwarmLogMonitorAgent"),
    ("LoopReport", "agents.autonomy.base", "LoopReport"),
]

_EXPORT_MAP = {name: (module_path, attr) for name, module_path, attr in _EXPORTS}


def __getattr__(name: str):
    if name in _EXPORT_MAP:
        warnings.warn(
            _DEPRECATION_MESSAGE,
            DeprecationWarning,
            stacklevel=2,
        )
        module_path, attr = _EXPORT_MAP[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_EXPORT_MAP.keys()))


__all__ = [name for name, _, _ in _EXPORTS]
