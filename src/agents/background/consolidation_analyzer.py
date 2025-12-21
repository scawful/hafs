"""Consolidation Analyzer Agent - suggests filesystem organization strategies.

Analyzes filesystem inventories and suggests consolidation opportunities
based on file types, duplicates, access patterns, and organizational best practices.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.background.base import BackgroundAgent

logger = logging.getLogger(__name__)


class ConsolidationAnalyzerAgent(BackgroundAgent):
    """Consolidation analyzer for filesystem organization recommendations.

    Analyzes filesystem inventories and provides actionable recommendations
    for organizing, deduplicating, and consolidating files.
    """

    def __init__(self, config_path: str | Path | None = None, verbose: bool = False):
        """Initialize consolidation analyzer agent."""
        super().__init__(config_path, verbose)
        self.inventory_dir = Path(
            self.config.tasks.get(
                "inventory_dir", "D:/.context/scratchpad/filesystem_explorer"
            )
        )
        self.consolidation_rules = self.config.tasks.get("consolidation_rules", {})

    def run(self) -> dict[str, Any]:
        """Execute consolidation analysis.

        Returns:
            Dictionary with consolidation recommendations
        """
        results = {
            "analysis_timestamp": datetime.now().isoformat(),
            "recommendations": [],
            "priority_actions": [],
            "potential_savings_gb": 0.0,
            "organization_suggestions": [],
        }

        # Load latest inventory
        inventory = self._load_latest_inventory()
        if not inventory:
            logger.error("No filesystem inventory found")
            return {
                "status": "error",
                "message": "No inventory found. Run filesystem_explorer first.",
            }

        logger.info("Analyzing filesystem inventory...")

        # Analyze duplicates
        duplicate_recommendations = self._analyze_duplicates(inventory)
        results["recommendations"].extend(duplicate_recommendations)

        # Analyze file organization
        organization_recommendations = self._analyze_organization(inventory)
        results["organization_suggestions"].extend(organization_recommendations)

        # Analyze by extension patterns
        extension_recommendations = self._analyze_extensions(inventory)
        results["recommendations"].extend(extension_recommendations)

        # Calculate total potential savings
        results["potential_savings_gb"] = sum(
            r.get("savings_gb", 0) for r in results["recommendations"]
        )

        # Prioritize actions
        results["priority_actions"] = self._prioritize_actions(
            results["recommendations"]
        )

        # Save results
        self._save_output(results, "consolidation_analysis")

        # Generate actionable report
        report = self._generate_report(results)
        self._save_output(report, "consolidation_report", format="md")

        return results

    def _load_latest_inventory(self) -> dict[str, Any] | None:
        """Load the most recent filesystem inventory.

        Returns:
            Inventory data or None if not found
        """
        if not self.inventory_dir.exists():
            return None

        # Find latest inventory file
        inventory_files = sorted(
            self.inventory_dir.glob("filesystem_inventory_*.json"), reverse=True
        )

        if not inventory_files:
            return None

        latest = inventory_files[0]
        logger.info(f"Loading inventory from {latest}")

        with open(latest) as f:
            return json.load(f)

    def _analyze_duplicates(self, inventory: dict[str, Any]) -> list[dict[str, Any]]:
        """Analyze duplicate files and recommend cleanup.

        Args:
            inventory: Filesystem inventory

        Returns:
            List of duplicate-related recommendations
        """
        recommendations = []

        duplicates = inventory.get("duplicate_candidates", [])
        if not duplicates:
            return recommendations

        # Group duplicates by type
        by_extension: dict[str, list[dict]] = defaultdict(list)
        for dup_group in duplicates:
            if not dup_group["files"]:
                continue
            ext = Path(dup_group["files"][0]).suffix.lower() or "no_extension"
            by_extension[ext].append(dup_group)

        # Recommend cleanup by extension
        for ext, groups in by_extension.items():
            total_waste_gb = sum(g["waste_mb"] for g in groups) / 1024
            if total_waste_gb < 0.01:  # Skip < 10 MB
                continue

            recommendations.append(
                {
                    "type": "duplicate_cleanup",
                    "category": ext,
                    "severity": "high" if total_waste_gb > 1.0 else "medium",
                    "savings_gb": round(total_waste_gb, 2),
                    "description": f"Remove duplicate {ext} files",
                    "details": {
                        "duplicate_groups": len(groups),
                        "total_duplicates": sum(g["count"] for g in groups),
                        "examples": [
                            {
                                "files": g["files"],
                                "size_mb": round(g["size_mb"], 2),
                                "waste_mb": round(g["waste_mb"], 2),
                            }
                            for g in groups[:5]
                        ],
                    },
                    "action": "Review duplicate files and keep only one copy of each. "
                    f"Could free up {total_waste_gb:.2f} GB.",
                }
            )

        return recommendations

    def _analyze_organization(
        self, inventory: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Analyze file organization and suggest improvements.

        Args:
            inventory: Filesystem inventory

        Returns:
            List of organization suggestions
        """
        suggestions = []

        # Analyze extension distribution
        extension_summary = inventory.get("extension_summary", {})

        # Common file type categories
        categories = {
            "code": [".py", ".js", ".ts", ".cpp", ".h", ".java", ".go", ".rs"],
            "data": [".json", ".jsonl", ".csv", ".xml", ".yaml", ".toml"],
            "documents": [".md", ".txt", ".pdf", ".doc", ".docx"],
            "images": [".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"],
            "archives": [".zip", ".tar", ".gz", ".7z", ".rar"],
            "models": [".safetensors", ".bin", ".ckpt", ".pth"],
            "datasets": [".jsonl", ".parquet", ".arrow"],
        }

        # Count files by category across all directories
        category_counts: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "size_gb": 0.0, "extensions": set()}
        )

        for ext, data in extension_summary.items():
            for category, extensions in categories.items():
                if ext in extensions:
                    category_counts[category]["count"] += data["count"]
                    category_counts[category]["size_gb"] += data["size_gb"]
                    category_counts[category]["extensions"].add(ext)

        # Suggest consolidation for scattered categories
        for category, data in category_counts.items():
            if data["count"] < 10:  # Skip small categories
                continue

            suggestions.append(
                {
                    "category": category,
                    "file_count": data["count"],
                    "size_gb": round(data["size_gb"], 2),
                    "extensions": sorted(data["extensions"]),
                    "suggestion": f"Consolidate {category} files into dedicated directory",
                    "rationale": f"Found {data['count']} {category} files ({data['size_gb']:.1f} GB) "
                    f"scattered across multiple locations. "
                    f"Consider organizing into D:/projects/{category}/ or similar structure.",
                }
            )

        return suggestions

    def _analyze_extensions(self, inventory: dict[str, Any]) -> list[dict[str, Any]]:
        """Analyze file extensions and recommend actions.

        Args:
            inventory: Filesystem inventory

        Returns:
            List of extension-based recommendations
        """
        recommendations = []

        extension_summary = inventory.get("extension_summary", {})

        # Large extension categories
        for ext, data in extension_summary.items():
            size_gb = data["size_gb"]

            # Skip small categories
            if size_gb < 0.5:
                continue

            # Temporary files
            if ext in [".tmp", ".temp", ".cache", ".log"]:
                recommendations.append(
                    {
                        "type": "temporary_files",
                        "category": ext,
                        "severity": "medium",
                        "savings_gb": round(size_gb, 2),
                        "description": f"Clean up {ext} temporary files",
                        "details": {
                            "file_count": data["count"],
                            "size_gb": round(size_gb, 2),
                        },
                        "action": f"Review and delete {ext} files. These are typically safe to remove.",
                    }
                )

            # Compressed archives
            elif ext in [".zip", ".tar", ".gz", ".7z", ".rar"]:
                recommendations.append(
                    {
                        "type": "archive_cleanup",
                        "category": ext,
                        "severity": "low",
                        "savings_gb": round(size_gb, 2),
                        "description": f"Review {ext} archives for extraction/deletion",
                        "details": {
                            "file_count": data["count"],
                            "size_gb": round(size_gb, 2),
                        },
                        "action": f"Check if {ext} archives have been extracted. "
                        f"Delete archives after extraction to save {size_gb:.1f} GB.",
                    }
                )

            # Large datasets
            elif ext in [".jsonl", ".parquet", ".arrow", ".csv"]:
                if size_gb > 5.0:
                    recommendations.append(
                        {
                            "type": "dataset_archival",
                            "category": ext,
                            "severity": "low",
                            "savings_gb": 0.0,  # No deletion, just organization
                            "description": f"Archive large {ext} datasets",
                            "details": {
                                "file_count": data["count"],
                                "size_gb": round(size_gb, 2),
                            },
                            "action": f"Consider compressing or archiving {ext} files to save space. "
                            f"Current usage: {size_gb:.1f} GB.",
                        }
                    )

        return recommendations

    def _prioritize_actions(
        self, recommendations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Prioritize recommendations by impact.

        Args:
            recommendations: List of recommendations

        Returns:
            Sorted list of priority actions
        """
        # Sort by severity and savings
        severity_order = {"high": 3, "medium": 2, "low": 1}

        prioritized = sorted(
            recommendations,
            key=lambda r: (
                severity_order.get(r.get("severity", "low"), 0),
                r.get("savings_gb", 0),
            ),
            reverse=True,
        )

        return prioritized[:10]  # Top 10 actions

    def _generate_report(self, results: dict[str, Any]) -> str:
        """Generate markdown report.

        Args:
            results: Analysis results

        Returns:
            Markdown formatted report
        """
        lines = [
            "# Filesystem Consolidation Report",
            "",
            f"**Generated:** {results['analysis_timestamp']}",
            f"**Potential Savings:** {results['potential_savings_gb']:.2f} GB",
            "",
            "## Priority Actions",
            "",
        ]

        for i, action in enumerate(results["priority_actions"], 1):
            lines.extend(
                [
                    f"### {i}. {action['description']}",
                    "",
                    f"- **Type:** {action.get('type', 'N/A')}",
                    f"- **Severity:** {action.get('severity', 'N/A').upper()}",
                    f"- **Potential Savings:** {action.get('savings_gb', 0):.2f} GB",
                    "",
                    f"**Action:** {action.get('action', 'No action specified')}",
                    "",
                ]
            )

        # Organization suggestions
        if results["organization_suggestions"]:
            lines.extend(["## Organization Suggestions", ""])

            for i, suggestion in enumerate(results["organization_suggestions"], 1):
                lines.extend(
                    [
                        f"### {i}. {suggestion['suggestion']}",
                        "",
                        f"- **Category:** {suggestion['category']}",
                        f"- **Files:** {suggestion['file_count']:,}",
                        f"- **Size:** {suggestion['size_gb']:.2f} GB",
                        "",
                        f"**Rationale:** {suggestion['rationale']}",
                        "",
                    ]
                )

        return "\n".join(lines)


def main():
    """CLI entry point for consolidation analyzer agent."""
    parser = argparse.ArgumentParser(description="hafs Consolidation Analyzer Agent")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    agent = ConsolidationAnalyzerAgent(config_path=args.config, verbose=args.verbose)
    result = agent.execute()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
