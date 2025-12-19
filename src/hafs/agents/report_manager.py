"""Report Manager - Organize and retrieve generated reports.

Provides utilities for managing the report directory structure,
listing reports, searching content, and retrieving statistics.

Usage:
    manager = ReportManager()
    await manager.setup()
    reports = manager.list_reports(project="alttp", report_type="modules")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from hafs.agents.base import BaseAgent

logger = logging.getLogger(__name__)

REPORTS_ROOT = Path.home() / ".context" / "reports"


@dataclass
class ReportMetadata:
    """Metadata for a generated report."""

    path: Path
    project: str
    report_type: str
    topic: str
    created: datetime
    size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "path": str(self.path),
            "project": self.project,
            "report_type": self.report_type,
            "topic": self.topic,
            "created": self.created.isoformat(),
            "size_bytes": self.size_bytes,
        }


class ReportManager(BaseAgent):
    """Manages the organization and retrieval of generated reports."""

    REPORT_STRUCTURE = {
        "alttp": ["modules", "routines", "symbols", "analysis"],
        "oracle-of-secrets": ["features", "analysis", "modifications"],
        "gigaleak": ["translations", "cross_references"],
    }

    def __init__(self):
        super().__init__("ReportManager", "Organize and retrieve generated reports.")
        self.reports_root = REPORTS_ROOT

    async def setup(self):
        await super().setup()
        self._ensure_directory_structure()

    def _ensure_directory_structure(self):
        """Create the report directory structure."""
        self.reports_root.mkdir(parents=True, exist_ok=True)

        for project, subdirs in self.REPORT_STRUCTURE.items():
            project_dir = self.reports_root / project
            project_dir.mkdir(parents=True, exist_ok=True)

            for subdir in subdirs:
                (project_dir / subdir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Report directory structure ensured at {self.reports_root}")

    def list_reports(
        self,
        project: Optional[str] = None,
        report_type: Optional[str] = None,
        limit: int = 50
    ) -> List[ReportMetadata]:
        """List available reports.

        Args:
            project: Filter by project (e.g., "alttp", "oracle-of-secrets")
            report_type: Filter by report type (e.g., "modules", "features")
            limit: Maximum number of reports to return

        Returns:
            List of ReportMetadata sorted by creation date (newest first)
        """
        reports = []

        search_path = self.reports_root
        if project:
            search_path = search_path / project
            if report_type:
                search_path = search_path / report_type

        if not search_path.exists():
            return []

        for path in search_path.rglob("*.md"):
            try:
                parts = path.relative_to(self.reports_root).parts
                proj = parts[0] if len(parts) > 0 else "unknown"
                rtype = parts[1] if len(parts) > 1 else "general"

                stat = path.stat()
                reports.append(ReportMetadata(
                    path=path,
                    project=proj,
                    report_type=rtype,
                    topic=path.stem,
                    created=datetime.fromtimestamp(stat.st_ctime),
                    size_bytes=stat.st_size,
                ))
            except Exception as e:
                logger.debug(f"Failed to read report {path}: {e}")
                continue

        # Sort by creation date (newest first)
        reports.sort(key=lambda r: r.created, reverse=True)
        return reports[:limit]

    def get_report(self, path: str) -> Optional[str]:
        """Get report content by path.

        Args:
            path: Absolute or relative path to report

        Returns:
            Report content or None if not found
        """
        report_path = Path(path)
        if not report_path.is_absolute():
            report_path = self.reports_root / path

        if report_path.exists():
            return report_path.read_text()
        return None

    def search_reports(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search report contents.

        Args:
            query: Text to search for (case-insensitive)
            limit: Maximum number of results

        Returns:
            List of matching reports with context snippets
        """
        results = []
        query_lower = query.lower()

        for report in self.list_reports(limit=200):
            try:
                content = report.path.read_text()
                if query_lower in content.lower():
                    # Find context around match
                    idx = content.lower().find(query_lower)
                    start = max(0, idx - 100)
                    end = min(len(content), idx + 200)
                    context = content[start:end]

                    results.append({
                        "path": str(report.path),
                        "project": report.project,
                        "report_type": report.report_type,
                        "topic": report.topic,
                        "context": f"...{context}...",
                    })
            except Exception as e:
                logger.debug(f"Failed to search report {report.path}: {e}")
                continue

        return results[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Get report statistics.

        Returns:
            Dictionary with report counts, sizes, and organization
        """
        stats = {
            "total_reports": 0,
            "by_project": {},
            "total_size_mb": 0.0,
            "reports_root": str(self.reports_root),
        }

        for report in self.list_reports(limit=1000):
            stats["total_reports"] += 1
            stats["total_size_mb"] += report.size_bytes / (1024 * 1024)

            if report.project not in stats["by_project"]:
                stats["by_project"][report.project] = {"count": 0, "types": {}}

            stats["by_project"][report.project]["count"] += 1

            if report.report_type not in stats["by_project"][report.project]["types"]:
                stats["by_project"][report.project]["types"][report.report_type] = 0
            stats["by_project"][report.project]["types"][report.report_type] += 1

        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
        return stats

    def get_recent_reports(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent reports across all projects.

        Args:
            limit: Number of recent reports to return

        Returns:
            List of recent reports with metadata
        """
        reports = self.list_reports(limit=limit)
        return [r.to_dict() for r in reports]

    def delete_report(self, path: str) -> bool:
        """Delete a report.

        Args:
            path: Path to report to delete

        Returns:
            True if deleted, False otherwise
        """
        report_path = Path(path)
        if not report_path.is_absolute():
            report_path = self.reports_root / path

        if report_path.exists() and report_path.suffix == ".md":
            report_path.unlink()
            logger.info(f"Deleted report: {report_path}")
            return True
        return False

    async def run_task(self, task: str = "stats") -> Dict[str, Any]:
        """Run report manager task.

        Tasks:
            stats - Get report statistics
            list - List all reports
            list:PROJECT - List reports for a project
            list:PROJECT:TYPE - List reports for a project and type
            search:QUERY - Search report contents
            get:PATH - Get report content
            recent - Get recent reports
        """
        if task == "stats":
            return self.get_statistics()

        if task == "recent":
            return {"reports": self.get_recent_reports()}

        if task.startswith("list"):
            parts = task.split(":")
            project = parts[1] if len(parts) > 1 else None
            report_type = parts[2] if len(parts) > 2 else None

            reports = self.list_reports(project, report_type)
            return {
                "reports": [r.to_dict() for r in reports]
            }

        if task.startswith("search:"):
            query = task[7:].strip()
            return {"results": self.search_reports(query)}

        if task.startswith("get:"):
            path = task[4:].strip()
            content = self.get_report(path)
            return {"content": content} if content else {"error": "Report not found"}

        return {"error": f"Unknown task: {task}"}


# CLI entry point
async def main():
    """CLI entry point for report management."""
    import sys

    manager = ReportManager()
    await manager.setup()

    if len(sys.argv) < 2:
        task = "stats"
    else:
        task = " ".join(sys.argv[1:])

    result = await manager.run_task(task)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
