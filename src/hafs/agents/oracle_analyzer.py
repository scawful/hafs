"""Oracle of Secrets Analyzer - ROM hack code analysis pipeline.

Provides detailed analysis of Oracle of Secrets ROM hack features,
comparing modifications against vanilla ALTTP and documenting
custom implementations.

Usage:
    analyzer = OracleOfSecretsAnalyzer()
    await analyzer.setup()
    result = await analyzer.analyze_feature("Custom Shop Items")
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from hafs.agents.base import BaseAgent
from hafs.agents.context_report_pipeline import (
    ContextReportPipeline,
    ResearchContext,
)

logger = logging.getLogger(__name__)

REPORTS_ROOT = Path.home() / ".context" / "reports"


@dataclass
class OracleAnalysisContext(ResearchContext):
    """Context for Oracle of Secrets analysis."""

    modifications: List[Dict[str, Any]] = field(default_factory=list)
    vanilla_comparison: Dict[str, Any] = field(default_factory=dict)
    feature_analysis: str = ""
    patch_coverage: Dict[str, Any] = field(default_factory=dict)


class VanillaComparisonAgent(BaseAgent):
    """Compares ROM hack code against vanilla ALTTP."""

    def __init__(self, vanilla_kb: Any):
        super().__init__("VanillaComparisonAgent", "Compare ROM hack changes against vanilla ALTTP.")
        self.vanilla_kb = vanilla_kb
        self.model_tier = "reasoning"

    async def compare_modification(self, modification: Dict[str, Any]) -> Dict[str, Any]:
        """Compare a modification against vanilla."""
        address = modification.get("address", "")
        hack_symbol = modification.get("hack_symbol", "")

        # Find vanilla code at this address
        vanilla_match = None
        vanilla_description = None

        if self.vanilla_kb and hasattr(self.vanilla_kb, '_routines'):
            for routine in self.vanilla_kb._routines.values():
                if hasattr(routine, 'address') and routine.address == address:
                    vanilla_match = routine.name
                    vanilla_description = routine.description if hasattr(routine, 'description') else None
                    break

        comparison = {
            "address": address,
            "hack_symbol": hack_symbol,
            "modification_type": modification.get("modification_type"),
            "vanilla_match": vanilla_match,
            "vanilla_description": vanilla_description,
            "file_path": modification.get("file_path", ""),
        }

        if vanilla_match:
            comparison["impact_analysis"] = await self._analyze_impact(modification, vanilla_match, vanilla_description)

        return comparison

    async def _analyze_impact(
        self,
        modification: Dict[str, Any],
        vanilla_name: str,
        vanilla_description: Optional[str]
    ) -> str:
        """Analyze the impact of a modification."""
        prompt = f"""Analyze this ROM hack modification:

MODIFICATION:
- Address: {modification.get('address')}
- Hack Symbol: {modification.get('hack_symbol')}
- Type: {modification.get('modification_type')}
- File: {modification.get('file_path')}

VANILLA CODE IT MODIFIES:
- Routine: {vanilla_name}
- Description: {vanilla_description or 'Unknown'}

Explain:
1. What the vanilla code likely does (based on the routine name)
2. What the modification likely changes
3. Potential side effects to watch for
4. Risk level (Low/Medium/High)

Keep the analysis concise but technical."""

        return await self.generate_thought(prompt)


class FeatureDocumenter(BaseAgent):
    """Documents ROM hack features."""

    def __init__(self):
        super().__init__("FeatureDocumenter", "Document ROM hack features and their implementations.")
        self.model_tier = "reasoning"

    async def document_feature(
        self,
        feature_name: str,
        related_symbols: List[Dict],
        related_routines: List[Dict],
        modifications: List[Dict]
    ) -> str:
        """Generate feature documentation."""

        symbols_str = json.dumps(related_symbols[:10], indent=2, default=str)
        routines_str = json.dumps(related_routines[:10], indent=2, default=str)
        mods_str = json.dumps(modifications[:10], indent=2, default=str)

        prompt = f"""Document this Oracle of Secrets feature:

FEATURE: {feature_name}

RELATED SYMBOLS:
{symbols_str}

RELATED ROUTINES:
{routines_str}

MODIFICATIONS INVOLVED:
{mods_str}

Generate comprehensive documentation including:
1. **Feature Overview**: What this feature does in the game
2. **Implementation**: How it's implemented technically
3. **Memory/Addresses**: Key memory locations used
4. **Hooks**: Where vanilla code is hooked/modified
5. **Dependencies**: Other features or systems this depends on
6. **Testing Notes**: How to test this feature in-game

Format as clean Markdown."""

        return await self.generate_thought(prompt)


class OracleOfSecretsAnalyzer(ContextReportPipeline):
    """Specialized pipeline for Oracle of Secrets ROM hack analysis."""

    def __init__(self):
        super().__init__(project="oracle-of-secrets")
        self.name = "OracleOfSecretsAnalyzer"

        self._oracle_kb = None
        self._vanilla_kb = None
        self._comparison_agent: Optional[VanillaComparisonAgent] = None
        self._feature_documenter: Optional[FeatureDocumenter] = None

    async def setup(self):
        """Initialize Oracle of Secrets analysis components."""
        await super().setup()

        # Load Oracle KB
        try:
            from hafs.agents.oracle_kb_builder import OracleKnowledgeBase
            self._oracle_kb = OracleKnowledgeBase()
            await self._oracle_kb.setup()
            logger.info(f"OracleKB loaded: {self._oracle_kb.get_statistics()}")
        except Exception as e:
            logger.warning(f"Could not load Oracle KB: {e}")

        # Load vanilla KB for comparison (already loaded by parent)
        self._vanilla_kb = self._kb

        # Create specialized agents
        self._comparison_agent = VanillaComparisonAgent(self._vanilla_kb)
        await self._comparison_agent.setup()

        self._feature_documenter = FeatureDocumenter()
        await self._feature_documenter.setup()

        logger.info("OracleOfSecretsAnalyzer initialized")

    async def analyze_feature(self, feature_name: str) -> Dict[str, Any]:
        """Analyze a specific ROM hack feature.

        Args:
            feature_name: Name or keyword for the feature to analyze

        Returns:
            Analysis results with report path and comparisons
        """

        context = OracleAnalysisContext(
            topic=f"Feature: {feature_name}",
            project="oracle-of-secrets",
            research_queries=[feature_name, feature_name.replace(" ", "_")],
        )

        # Search Oracle KB
        oracle_results = []
        if self._oracle_kb:
            oracle_results = await self._oracle_kb.search(feature_name, limit=20)

        # Search vanilla for comparison context
        vanilla_results = []
        if self._vanilla_kb and hasattr(self._vanilla_kb, 'search'):
            try:
                vanilla_results = await self._vanilla_kb.search(feature_name, limit=10)
            except Exception:
                pass

        # Find related modifications
        related_mods = []
        if self._oracle_kb:
            for m in self._oracle_kb._modifications:
                m_dict = m.to_dict() if hasattr(m, 'to_dict') else m
                if feature_name.lower() in str(m_dict).lower():
                    related_mods.append(m_dict)

        context.gathered_context = {
            "oracle_symbols": [r for r in oracle_results if r.get("type") == "symbol"],
            "oracle_routines": [r for r in oracle_results if r.get("type") == "routine"],
            "vanilla_matches": vanilla_results,
            "modifications": related_mods,
        }
        context.modifications = related_mods

        # Vanilla comparison
        comparisons = []
        if self._comparison_agent:
            for mod in related_mods[:10]:
                try:
                    comp = await self._comparison_agent.compare_modification(mod)
                    comparisons.append(comp)
                except Exception as e:
                    logger.debug(f"Comparison failed: {e}")
        context.vanilla_comparison = {"comparisons": comparisons}

        # Feature documentation
        if self._feature_documenter:
            context.feature_analysis = await self._feature_documenter.document_feature(
                feature_name,
                context.gathered_context["oracle_symbols"],
                context.gathered_context["oracle_routines"],
                related_mods
            )

        # Generate report
        report = await self._generate_feature_report(context, feature_name)

        return {
            "feature": feature_name,
            "report_path": str(context.report_path) if context.report_path else None,
            "report": report,
            "vanilla_comparisons": comparisons,
            "oracle_kb_stats": self._oracle_kb.get_statistics() if self._oracle_kb else {},
            "symbols_found": len(context.gathered_context.get("oracle_symbols", [])),
            "routines_found": len(context.gathered_context.get("oracle_routines", [])),
            "modifications_found": len(related_mods),
        }

    async def _generate_feature_report(
        self,
        context: OracleAnalysisContext,
        feature_name: str
    ) -> str:
        """Generate feature report."""

        # Create report content
        report_header = f"""# Oracle of Secrets Feature Analysis: {feature_name}

Generated: {datetime.now().isoformat()}
Project: Oracle of Secrets ROM Hack

---

"""

        report_body = f"""## Feature Documentation

{context.feature_analysis}

---

## Vanilla ALTTP Comparisons

"""
        for comp in context.vanilla_comparison.get("comparisons", []):
            report_body += f"""### {comp.get('hack_symbol', 'Unknown')}
- **Address**: {comp.get('address', 'N/A')}
- **Vanilla Match**: {comp.get('vanilla_match', 'None')}
- **Type**: {comp.get('modification_type', 'unknown')}

{comp.get('impact_analysis', '')}

"""

        report_body += f"""---

## Statistics

- Symbols Found: {len(context.gathered_context.get('oracle_symbols', []))}
- Routines Found: {len(context.gathered_context.get('oracle_routines', []))}
- Modifications: {len(context.modifications)}

"""

        report = report_header + report_body

        # Save to features subdirectory
        reports_dir = REPORTS_ROOT / "oracle-of-secrets" / "features"
        reports_dir.mkdir(parents=True, exist_ok=True)

        safe_name = feature_name.replace(" ", "_").replace("/", "_")[:50]
        filename = f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"

        path = reports_dir / filename
        path.write_text(report)
        context.report_path = path

        logger.info(f"Feature report saved: {path}")
        return report

    async def analyze_all_modifications(self) -> Dict[str, Any]:
        """Analyze all modifications in the ROM hack."""

        if not self._oracle_kb:
            return {"error": "Oracle KB not loaded"}

        stats = self._oracle_kb.get_statistics()

        # Group by modification type
        by_type = {}
        for mod_type in stats.get("modification_types", {}).keys():
            mods = self._oracle_kb.get_modifications_by_type(mod_type)
            by_type[mod_type] = {
                "count": len(mods),
                "samples": [m.to_dict() if hasattr(m, 'to_dict') else m for m in mods[:5]]
            }

        # Group by category
        by_category = {}
        for symbol in self._oracle_kb._symbols.values():
            cat = symbol.category or "uncategorized"
            if cat not in by_category:
                by_category[cat] = {"count": 0, "symbols": [], "routines": []}
            by_category[cat]["count"] += 1
            if len(by_category[cat]["symbols"]) < 5:
                by_category[cat]["symbols"].append(symbol.name)

        # Generate summary report
        reports_dir = REPORTS_ROOT / "oracle-of-secrets" / "analysis"
        reports_dir.mkdir(parents=True, exist_ok=True)

        report = f"""# Oracle of Secrets Modification Analysis

Generated: {datetime.now().isoformat()}

## Statistics

{json.dumps(stats, indent=2)}

## Modifications by Type

{json.dumps(by_type, indent=2, default=str)}

## Symbols by Category

{json.dumps(by_category, indent=2, default=str)}
"""

        path = reports_dir / f"modifications_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        path.write_text(report)

        return {
            "report_path": str(path),
            "statistics": stats,
            "by_type": by_type,
            "by_category": {k: v["count"] for k, v in by_category.items()},
        }

    async def run_task(self, task: str = "help") -> Dict[str, Any]:
        """Run Oracle analyzer task.

        Tasks:
            help - Show usage information
            feature:NAME - Analyze a specific feature
            modifications - Analyze all modifications
            stats - Get knowledge base statistics
            categories - List symbol categories
        """
        if task == "help":
            return {
                "usage": [
                    "feature:NAME - Analyze a specific feature (e.g., feature:Custom_Shop)",
                    "modifications - Analyze all modifications",
                    "stats - Get knowledge base statistics",
                    "categories - List symbol categories",
                ]
            }

        if task == "stats":
            if self._oracle_kb:
                return self._oracle_kb.get_statistics()
            return {"error": "Oracle KB not loaded"}

        if task == "categories":
            if self._oracle_kb:
                stats = self._oracle_kb.get_statistics()
                return {"categories": stats.get("categories", {})}
            return {"error": "Oracle KB not loaded"}

        if task == "modifications":
            return await self.analyze_all_modifications()

        if task.startswith("feature:"):
            feature = task[8:].strip()
            return await self.analyze_feature(feature)

        return await super().run_task(task)


# CLI entry point
async def main():
    """CLI entry point for Oracle analysis."""
    import sys

    analyzer = OracleOfSecretsAnalyzer()
    await analyzer.setup()

    if len(sys.argv) < 2:
        result = await analyzer.run_task("help")
    else:
        task = sys.argv[1]
        if len(sys.argv) > 2:
            task = f"{task}:{' '.join(sys.argv[2:])}"
        result = await analyzer.run_task(task)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
