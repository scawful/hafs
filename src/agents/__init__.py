"""HAFS Agents system."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = [
    # Core
    ("BaseAgent", "agents.core.base", "BaseAgent"),
    ("AgentMetrics", "agents.core.base", "AgentMetrics"),
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
    # Autonomy
    ("LoopReport", "agents.autonomy.base", "LoopReport"),
    ("MemoryAwareAgent", "agents.autonomy.base", "MemoryAwareAgent"),
    ("SelfImprovementAgent", "agents.autonomy.self_improvement", "SelfImprovementAgent"),
    ("CuriosityExplorerAgent", "agents.autonomy.curiosity", "CuriosityExplorerAgent"),
    ("SelfHealingAgent", "agents.autonomy.self_healing", "SelfHealingAgent"),
    ("HallucinationWatcherAgent", "agents.autonomy.hallucination", "HallucinationWatcherAgent"),
    # Knowledge
    ("ALTTPKnowledgeBase", "agents.knowledge.alttp", "ALTTPKnowledgeBase"),
    ("ALTTPMultiKBManager", "agents.knowledge.alttp_multi", "ALTTPMultiKBManager"),
    ("UnifiedALTTPKnowledge", "agents.knowledge.alttp_unified", "UnifiedALTTPKnowledge"),
    ("ALTTPEmbeddingSpecialist", "agents.knowledge.alttp_embeddings", "ALTTPEmbeddingSpecialist"),
    ("ALTTPModuleAnalyzer", "agents.knowledge.alttp_analyzer", "ALTTPModuleAnalyzer"),
    ("OracleKnowledgeBase", "agents.knowledge.oracle", "OracleKnowledgeBase"),
    ("OracleKBBuilder", "agents.knowledge.oracle", "OracleKBBuilder"),
    ("OracleOfSecretsAnalyzer", "agents.knowledge.oracle_analyzer", "OracleOfSecretsAnalyzer"),
    ("GigaleakKB", "agents.knowledge.gigaleak", "GigaleakKB"),
    ("KnowledgeGraphAgent", "agents.knowledge.graph", "KnowledgeGraphAgent"),
    ("KBEnhancer", "agents.knowledge.enhancer", "KBEnhancer"),
    ("RomHackingSpecialist", "agents.knowledge.rom", "RomHackingSpecialist"),
    # Pipeline
    ("ArchitectCouncil", "agents.pipeline.architect_council", "ArchitectCouncil"),
    ("BuilderCouncil", "agents.pipeline.builder_council", "BuilderCouncil"),
    ("ValidatorCouncil", "agents.pipeline.validator_council", "ValidatorCouncil"),
    ("CodeWriter", "agents.pipeline.code_writer", "CodeWriter"),
    ("DocWriter", "agents.pipeline.doc_writer", "DocWriter"),
    ("TestWriter", "agents.pipeline.test_writer", "TestWriter"),
    ("BuildAgent", "agents.pipeline.build_test_agents", "BuildAgent"),
    ("TestAgent", "agents.pipeline.build_test_agents", "TestAgent"),
    ("ReviewUploader", "agents.pipeline.review_uploader", "ReviewUploader"),
    # Swarm
    ("SwarmCouncil", "agents.swarm.swarm", "SwarmCouncil"),
    ("SwarmStatus", "agents.swarm.swarm", "SwarmStatus"),
    ("SwarmStrategist", "agents.swarm.specialists", "SwarmStrategist"),
    ("CouncilReviewer", "agents.swarm.specialists", "CouncilReviewer"),
    ("DeepDiveDocumenter", "agents.swarm.specialists", "DeepDiveDocumenter"),
    # Mission
    ("ResearchMission", "agents.mission.mission_agents", "ResearchMission"),
    ("ResearchDiscovery", "agents.mission.mission_agents", "ResearchDiscovery"),
    ("MissionAgent", "agents.mission.mission_agents", "MissionAgent"),
    ("ALTTPResearchAgent", "agents.mission.mission_agents", "ALTTPResearchAgent"),
    ("GigaleakAnalysisAgent", "agents.mission.mission_agents", "GigaleakAnalysisAgent"),
    ("get_mission_agent", "agents.mission.mission_agents", "get_mission_agent"),
    ("DEFAULT_MISSIONS", "agents.mission.mission_agents", "DEFAULT_MISSIONS"),
    # Analysis
    ("AutonomousContextAgent", "agents.analysis.context_builder", "AutonomousContextAgent"),
    ("CodeDescriber", "agents.analysis.code_describer", "CodeDescriber"),
    ("CodeKnowledgeBase", "agents.analysis.code_describer", "CodeKnowledgeBase"),
    ("CodeUnit", "agents.analysis.code_describer", "CodeUnit"),
    ("LanguagePlugin", "agents.analysis.code_describer", "LanguagePlugin"),
    ("EmbeddingAnalyzer", "agents.analysis.embedding_analyzer", "EmbeddingAnalyzer"),
    ("ContextReportPipeline", "agents.analysis.report_pipeline", "ContextReportPipeline"),
    ("RepoSnapshotAgent", "agents.analysis.deep_context_pipeline", "RepoSnapshotAgent"),
    ("ContextSignalAgent", "agents.analysis.deep_context_pipeline", "ContextSignalAgent"),
    ("MLSignalAgent", "agents.analysis.deep_context_pipeline", "MLSignalAgent"),
    ("MLPipelinePlannerAgent", "agents.analysis.deep_context_pipeline", "MLPipelinePlannerAgent"),
    ("DeepContextPipeline", "agents.analysis.deep_context_pipeline", "DeepContextPipeline"),
    ("SmartMLPipeline", "agents.analysis.deep_context_pipeline", "SmartMLPipeline"),
    ("register_plugin", "agents.analysis.code_describer", "register_plugin"),
    ("get_plugin", "agents.analysis.code_describer", "get_plugin"),
    ("detect_language", "agents.analysis.code_describer", "detect_language"),
    # Utility
    ("CartographerAgent", "agents.utility.cartographer", "CartographerAgent"),
    ("ChronosAgent", "agents.utility.chronos", "ChronosAgent"),
    ("ContextVectorAgent", "agents.utility.vector_memory", "ContextVectorAgent"),
    ("DailyBriefingAgent", "agents.utility.daily_briefing", "DailyBriefingAgent"),
    ("DistributedObservabilityAgent", "agents.utility.observability", "DistributedObservabilityAgent"),
    ("EpisodicMemoryAgent", "agents.utility.episodic", "EpisodicMemoryAgent"),
    ("GardenerAgent", "agents.utility.gardener", "ContextGardener"),
    ("GeminiHistorianAgent", "agents.utility.gemini_historian", "GeminiHistorian"),
    ("HistoryPipelineAgent", "agents.utility.history_pipeline", "HistoryPipelineAgent"),
    ("MonitorAgent", "agents.utility.monitor", "ActivityMonitor"),
    ("PromptEngineerAgent", "agents.utility.prompt_engineer", "PromptEngineerAgent"),
    ("ReportManagerAgent", "agents.utility.report_manager", "ReportManager"),
    ("ScoutAgent", "agents.utility.scout", "ScoutAgent"),
    ("ShadowObserver", "agents.utility.shadow_observer", "ShadowObserver"),
    ("ShellAgent", "agents.utility.shell_agent", "ShellAgent"),
    ("ToolsmithAgent", "agents.utility.toolsmith", "Toolsmith"),
    ("TrendWatcherAgent", "agents.utility.trend_watcher", "TrendWatcher"),
    ("VisualizerAgent", "agents.utility.visualizer", "VisualizerAgent"),
]

_EXPORT_MAP = {name: (module_path, attr) for name, module_path, attr in _EXPORTS}


def __getattr__(name: str) -> Any:
    if name in _EXPORT_MAP:
        module_path, attr = _EXPORT_MAP[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_EXPORT_MAP.keys()))


__all__ = [name for name, _, _ in _EXPORTS]
