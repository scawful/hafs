"""Context Report Pipeline - Multi-agent orchestration for embedding-aware context building.

Provides a 4-phase pipeline for generating comprehensive context reports:
1. RESEARCH - Parallel semantic search using embeddings
2. ANALYZE - Clustering and pattern detection
3. REVIEW - Quality validation and gap detection
4. SYNTHESIZE - Report generation

Usage:
    pipeline = ContextReportPipeline(project="alttp")
    await pipeline.setup()
    result = await pipeline.generate_report("Underworld module")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from hafs.agents.base import BaseAgent
from hafs.core.orchestration import OrchestrationPipeline, PipelineContext, PipelineStep

logger = logging.getLogger(__name__)

REPORTS_ROOT = Path.home() / ".context" / "reports"


@dataclass
class ResearchContext(PipelineContext):
    """Extended context for research pipelines."""

    project: str = ""
    research_queries: List[str] = field(default_factory=list)
    embedding_results: Dict[str, Any] = field(default_factory=dict)
    gathered_context: Dict[str, Any] = field(default_factory=dict)
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    reviewer_feedback: str = ""
    final_report: str = ""
    report_path: Optional[Path] = None


class EmbeddingResearchAgent(BaseAgent):
    """Research agent that uses embeddings for semantic context retrieval."""

    def __init__(self, kb: Any, name_suffix: str = ""):
        super().__init__(
            f"EmbeddingResearchAgent{name_suffix}",
            "Perform semantic search over knowledge bases using embeddings."
        )
        self.kb = kb
        self._search_cache: Dict[str, List[Dict]] = {}

    async def search(
        self,
        query: str,
        limit: int = 20,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Semantic search using embeddings."""
        cache_key = f"{query}:{limit}:{category_filter}"
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        if hasattr(self.kb, 'search'):
            results = await self.kb.search(query, limit=limit, category_filter=category_filter)
        else:
            # Fallback to basic embedding search
            results = await self._basic_embedding_search(query, limit)

        self._search_cache[cache_key] = results
        return results

    async def _basic_embedding_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Basic embedding search when KB doesn't have search method."""
        if not self.orchestrator:
            await self.setup()

        if not self.orchestrator or not hasattr(self.kb, '_embeddings'):
            return []

        # Generate query embedding
        query_embedding = await self.orchestrator.embed(query)
        if not query_embedding:
            return []

        # Search against indexed embeddings
        results = []
        for emb_id, embedding in self.kb._embeddings.items():
            score = self._cosine_similarity(query_embedding, embedding)
            results.append({"id": emb_id, "name": emb_id, "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

    async def gather_related_context(
        self,
        seed_symbols: List[str],
        depth: int = 2
    ) -> Dict[str, Any]:
        """Gather context expanding from seed symbols using embeddings."""
        context = {"symbols": {}, "routines": {}, "relationships": []}

        visited = set()
        current_level = seed_symbols

        for level in range(depth):
            next_level = []
            for symbol in current_level:
                if symbol in visited:
                    continue
                visited.add(symbol)

                # Search for related symbols
                related = await self.search(symbol, limit=10)
                for r in related:
                    name = r.get("name", r.get("id", ""))
                    if name not in visited:
                        next_level.append(name)
                        context["relationships"].append({
                            "source": symbol,
                            "target": name,
                            "score": r.get("score", 0),
                            "level": level
                        })

                    if r.get("type") == "symbol":
                        context["symbols"][name] = r
                    elif r.get("type") == "routine":
                        context["routines"][name] = r
                    else:
                        context["symbols"][name] = r

            current_level = next_level[:20]  # Limit expansion

        return context

    async def run_task(self, task: str) -> Dict[str, Any]:
        """Run research task."""
        if task.startswith("search:"):
            query = task[7:].strip()
            return {"results": await self.search(query)}
        elif task.startswith("gather:"):
            seeds = task[7:].strip().split(",")
            return await self.gather_related_context(seeds)
        return {"error": "Unknown task"}


class AnalysisAgent(BaseAgent):
    """Processes gathered context and extracts insights."""

    def __init__(self, analyzer: Optional[Any] = None):
        super().__init__("AnalysisAgent", "Analyze gathered context and extract insights.")
        self.analyzer = analyzer
        self.model_tier = "reasoning"

    async def analyze_module_structure(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze relationships between symbols and routines."""
        symbols = context.get("symbols", {})
        routines = context.get("routines", {})
        relationships = context.get("relationships", [])

        # Cluster if analyzer available
        clusters = []
        if self.analyzer and hasattr(self.analyzer, '_embeddings') and self.analyzer._embeddings:
            try:
                cluster_results = await self.analyzer.cluster(
                    n_clusters=min(10, len(symbols) // 3 or 1)
                )
                clusters = self.analyzer.get_cluster_summary() if hasattr(self.analyzer, 'get_cluster_summary') else []
            except Exception as e:
                logger.debug(f"Clustering failed: {e}")

        # Generate analysis with LLM
        analysis_prompt = f"""Analyze this ALTTP code structure:

SYMBOLS ({len(symbols)} total):
{json.dumps(list(symbols.values())[:20], indent=2, default=str)}

ROUTINES ({len(routines)} total):
{json.dumps(list(routines.values())[:20], indent=2, default=str)}

RELATIONSHIPS:
{json.dumps(relationships[:30], indent=2)}

Provide:
1. Overview of the module/subsystem
2. Key data structures and their purposes
3. Important routines and their roles
4. Architectural patterns observed
5. Suggestions for further investigation"""

        analysis = await self.generate_thought(analysis_prompt)

        return {
            "symbol_count": len(symbols),
            "routine_count": len(routines),
            "relationship_count": len(relationships),
            "clusters": clusters,
            "llm_analysis": analysis,
        }

    async def run_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return await self.analyze_module_structure(context)


class SynthesisAgent(BaseAgent):
    """Generates reports from analyzed context."""

    def __init__(self, project: str):
        super().__init__("SynthesisAgent", "Generate comprehensive reports from analysis.")
        self.project = project
        self.reports_dir = REPORTS_ROOT / project
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.model_tier = "reasoning"

    async def generate_report(
        self,
        topic: str,
        analysis: Dict[str, Any],
        gathered_context: Dict[str, Any],
        reviewer_feedback: str = "",
        report_type: str = "module"
    ) -> str:
        """Generate a comprehensive report."""

        report_prompt = f"""Generate a comprehensive technical report on: {topic}

PROJECT: {self.project}

ANALYSIS RESULTS:
{json.dumps(analysis, indent=2, default=str)[:5000]}

GATHERED CONTEXT (sample):
- {len(gathered_context.get('symbols', {}))} symbols analyzed
- {len(gathered_context.get('routines', {}))} routines analyzed

{f"REVIEWER FEEDBACK: {reviewer_feedback}" if reviewer_feedback else ""}

Generate a detailed report with:
1. **Executive Summary**: Key findings in 2-3 sentences
2. **Technical Overview**: Architecture and design patterns
3. **Key Components**: Important symbols, routines, memory addresses
4. **Implementation Details**: How the subsystem works
5. **Cross-References**: Related modules and dependencies
6. **ROM Hacking Implications**: How this knowledge aids modification
7. **Open Questions**: Areas needing further investigation

Format as clean Markdown."""

        report = await self.generate_thought(report_prompt)
        return report

    def save_report(self, topic: str, content: str, report_type: str = "module") -> Path:
        """Save report to organized directory."""
        subdir = self.reports_dir / report_type
        subdir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        safe_topic = topic.replace(" ", "_").replace("/", "_")[:50]
        filename = f"{safe_topic}_{timestamp}.md"

        path = subdir / filename
        path.write_text(content)
        logger.info(f"Report saved: {path}")
        return path

    async def run_task(self, context: ResearchContext) -> str:
        report = await self.generate_report(
            topic=context.topic,
            analysis=context.analysis_results,
            gathered_context=context.gathered_context,
            reviewer_feedback=context.reviewer_feedback,
        )
        context.report_path = self.save_report(context.topic, report)
        context.final_report = report
        return report


class ReviewerAgent(BaseAgent):
    """Validates quality and completeness of analysis."""

    def __init__(self):
        super().__init__("ReviewerAgent", "Validate analysis quality and completeness.")
        self.model_tier = "fast"

    async def review(
        self,
        topic: str,
        gathered_context: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Review gathered context and analysis for quality."""

        review_prompt = f"""Review this research for quality and completeness.

TOPIC: {topic}

GATHERED DATA:
- Symbols: {len(gathered_context.get('symbols', {}))}
- Routines: {len(gathered_context.get('routines', {}))}
- Relationships: {len(gathered_context.get('relationships', []))}

ANALYSIS SUMMARY:
{json.dumps(analysis, indent=2, default=str)[:3000]}

EVALUATE:
1. **Completeness**: Did we gather enough context? (1-10)
2. **Accuracy**: Are the findings consistent with ALTTP knowledge? (1-10)
3. **Depth**: Is the analysis sufficiently detailed? (1-10)
4. **Gaps**: What critical information is missing?
5. **Recommendations**: What additional research should be done?
6. **Overall Score**: (1-100)

Respond with structured JSON."""

        response = await self.generate_thought(review_prompt)

        # Parse response
        try:
            import re
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass

        return {
            "completeness": 5,
            "accuracy": 5,
            "depth": 5,
            "gaps": [],
            "recommendations": [],
            "overall_score": 50,
            "raw_review": response
        }

    async def run_task(self, context: ResearchContext) -> Dict[str, Any]:
        return await self.review(
            context.topic,
            context.gathered_context,
            context.analysis_results
        )


class ContextReportPipeline(BaseAgent):
    """Main orchestrator for embedding-aware context report generation.

    Coordinates multiple subagents through a 4-phase pipeline:
    1. Research - Parallel semantic search
    2. Analyze - Clustering and pattern detection
    3. Review - Quality validation
    4. Synthesize - Report generation
    """

    def __init__(self, project: str = "alttp"):
        super().__init__(
            "ContextReportPipeline",
            "Orchestrate multi-agent context building with embeddings."
        )
        self.project = project

        # Lazy-loaded components
        self._kb: Optional[Any] = None
        self._multi_kb: Optional[Any] = None
        self._analyzer: Optional[Any] = None

        # Pipeline agents
        self._researchers: List[EmbeddingResearchAgent] = []
        self._analysis_agent: Optional[AnalysisAgent] = None
        self._synthesis_agent: Optional[SynthesisAgent] = None
        self._reviewer_agent: Optional[ReviewerAgent] = None

    async def setup(self):
        """Initialize the pipeline with all required components."""
        await super().setup()

        # Initialize knowledge base based on project
        if self.project == "alttp":
            try:
                from hafs.agents.alttp_knowledge import ALTTPKnowledgeBase
                self._kb = ALTTPKnowledgeBase()
                await self._kb.setup()
                logger.info("ALTTPKnowledgeBase loaded")
            except ImportError:
                logger.warning("ALTTPKnowledgeBase not available")
            except Exception as e:
                logger.error(f"Failed to load ALTTPKnowledgeBase: {e}")

        # Initialize multi-KB manager if available
        try:
            from hafs.agents.alttp_multi_kb import ALTTPMultiKBManager
            self._multi_kb = ALTTPMultiKBManager()
            await self._multi_kb.setup()
        except Exception:
            pass

        # Initialize analyzer if available
        try:
            from hafs.agents.embedding_analysis import EmbeddingAnalyzer
            self._analyzer = EmbeddingAnalyzer()
            await self._analyzer.setup()
            if self._kb:
                await self._analyzer.load_from_kb(self.project)
        except Exception as e:
            logger.debug(f"EmbeddingAnalyzer not available: {e}")

        # Create pipeline agents
        kb_for_research = self._kb if self._kb else self
        self._researchers = [
            EmbeddingResearchAgent(kb_for_research, f"_{i}")
            for i in range(3)  # 3 parallel researchers
        ]
        for r in self._researchers:
            await r.setup()

        self._analysis_agent = AnalysisAgent(self._analyzer)
        await self._analysis_agent.setup()

        self._synthesis_agent = SynthesisAgent(self.project)
        await self._synthesis_agent.setup()

        self._reviewer_agent = ReviewerAgent()
        await self._reviewer_agent.setup()

        logger.info(f"ContextReportPipeline initialized for project: {self.project}")

    async def _step_research(self, context: ResearchContext) -> Dict[str, Any]:
        """Research phase: parallel semantic search."""
        logger.info(f"[Research] Starting for topic: {context.topic}")

        # Split queries among researchers
        queries = context.research_queries or [context.topic]

        # Run parallel research
        tasks = []
        for i, query in enumerate(queries):
            researcher = self._researchers[i % len(self._researchers)]
            tasks.append(researcher.search(query, limit=15))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        all_results = []
        for r in results:
            if isinstance(r, list):
                all_results.extend(r)

        # Deduplicate
        seen = set()
        unique_results = []
        for r in all_results:
            name = r.get("name", r.get("id", ""))
            if name not in seen:
                seen.add(name)
                unique_results.append(r)

        context.embedding_results = {"results": unique_results}

        # Gather expanded context
        seed_symbols = [r.get("name", r.get("id", "")) for r in unique_results[:10]]
        if seed_symbols and self._researchers:
            context.gathered_context = await self._researchers[0].gather_related_context(seed_symbols)
        else:
            context.gathered_context = {"symbols": {}, "routines": {}, "relationships": []}

        logger.info(f"[Research] Found {len(unique_results)} results, gathered {len(context.gathered_context.get('symbols', {}))} symbols")
        return context.embedding_results

    async def _step_analyze(self, context: ResearchContext) -> Dict[str, Any]:
        """Analysis phase: process gathered context."""
        logger.info("[Analysis] Processing gathered context")

        context.analysis_results = await self._analysis_agent.run_task(context.gathered_context)
        return context.analysis_results

    async def _step_review(self, context: ResearchContext) -> Dict[str, Any]:
        """Review phase: validate quality."""
        logger.info("[Review] Validating analysis quality")

        review = await self._reviewer_agent.run_task(context)
        context.reviewer_feedback = json.dumps(review, indent=2)
        return review

    async def _step_synthesize(self, context: ResearchContext) -> str:
        """Synthesis phase: generate final report."""
        logger.info("[Synthesis] Generating final report")

        report = await self._synthesis_agent.run_task(context)
        return report

    async def generate_report(
        self,
        topic: str,
        queries: Optional[List[str]] = None,
        report_type: str = "module"
    ) -> Dict[str, Any]:
        """Generate a complete context report.

        Args:
            topic: The main topic to research
            queries: Additional search queries (defaults to [topic])
            report_type: Type of report for organization (module, routine, symbol, analysis)

        Returns:
            Dict with report path, content, analysis, and pipeline status
        """

        context = ResearchContext(
            topic=topic,
            project=self.project,
            research_queries=queries or [topic],
        )

        pipeline = OrchestrationPipeline([
            PipelineStep(name="research", kind="collect", run=self._step_research),
            PipelineStep(name="analyze", kind="analyze", run=self._step_analyze),
            PipelineStep(name="review", kind="verify", run=self._step_review),
            PipelineStep(name="synthesize", kind="summarize", run=self._step_synthesize),
        ])

        result = await pipeline.run(context)

        return {
            "topic": topic,
            "project": self.project,
            "report_path": str(context.report_path) if context.report_path else None,
            "report": context.final_report,
            "analysis": context.analysis_results,
            "review": context.reviewer_feedback,
            "pipeline_status": [
                {"name": s.name, "status": s.status.value}
                for s in result.steps
            ]
        }

    async def run_task(self, task: str = "help") -> Dict[str, Any]:
        """Run pipeline task.

        Tasks:
            help - Show usage information
            report:TOPIC - Generate a report on a topic
            module:NAME - Analyze a specific game module
            routine:NAME - Analyze a specific routine
        """
        if task == "help":
            return {
                "usage": [
                    "report:TOPIC - Generate a report on a topic",
                    "module:MODULE_NAME - Analyze a specific game module",
                    "routine:ROUTINE_NAME - Analyze a specific routine",
                ]
            }

        if task.startswith("report:"):
            topic = task[7:].strip()
            return await self.generate_report(topic)

        if task.startswith("module:"):
            module = task[7:].strip()
            return await self.generate_report(module, report_type="modules")

        if task.startswith("routine:"):
            routine = task[8:].strip()
            return await self.generate_report(routine, report_type="routines")

        return {"error": f"Unknown task: {task}"}


# Convenience function for CLI usage
async def main():
    """CLI entry point for context report generation."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m hafs.agents.context_report_pipeline <topic>")
        print("       python -m hafs.agents.context_report_pipeline module:Underworld")
        return

    task = " ".join(sys.argv[1:])

    pipeline = ContextReportPipeline(project="alttp")
    await pipeline.setup()

    if task.startswith("module:") or task.startswith("routine:") or task.startswith("report:"):
        result = await pipeline.run_task(task)
    else:
        result = await pipeline.generate_report(task)

    print(f"\nReport generated: {result.get('report_path')}")
    print(f"Pipeline status: {result.get('pipeline_status')}")


if __name__ == "__main__":
    asyncio.run(main())
