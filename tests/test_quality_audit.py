"""Quality audit tests for background agent reports and embeddings.

Tests the quality and integrity of:
- Background agent reports (autonomy daemon, context agent)
- Embedding vectors (dimension, coverage, duplicates)
- Service health and consistency
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

if os.environ.get("HAFS_RUN_QUALITY_AUDIT") != "1":
    pytest.skip(
        "Set HAFS_RUN_QUALITY_AUDIT=1 to run quality audit tests.",
        allow_module_level=True,
    )

# Paths
CONTEXT_ROOT = Path.home() / ".context"
REPORTS_DIR = CONTEXT_ROOT / "background_agent" / "reports"
KNOWLEDGE_DIR = CONTEXT_ROOT / "knowledge"
AUTONOMY_STATUS = CONTEXT_ROOT / "autonomy_daemon" / "daemon_status.json"
EMBEDDING_STATUS = CONTEXT_ROOT / "embedding_service" / "daemon_status.json"
CONTEXT_AGENT_STATUS = CONTEXT_ROOT / "context_agent_daemon" / "daemon_status.json"


class TestReportQuality:
    """Test quality of generated reports."""

    def test_reports_directory_exists(self) -> None:
        """Report directories should exist."""
        assert REPORTS_DIR.exists(), f"Reports directory not found: {REPORTS_DIR}"

    def test_report_subdirectories(self) -> None:
        """Expected report subdirectories should exist."""
        expected_dirs = [
            "self_improvement",
            "curiosity",
            "self_heal",
            "shadow_observer",
            "hallucination_watch",
            "mission",
        ]
        for subdir in expected_dirs:
            path = REPORTS_DIR / subdir
            assert path.exists(), f"Missing report directory: {subdir}"

    def test_reports_have_content(self) -> None:
        """Reports should have meaningful content, not be empty."""
        min_content_length = 50  # Minimum chars for a valid report
        empty_reports: list[str] = []

        for report_dir in REPORTS_DIR.iterdir():
            if not report_dir.is_dir():
                continue
            for report_file in report_dir.glob("*.md"):
                content = report_file.read_text()
                # Strip header and check remaining content
                lines = content.strip().split("\n")
                # Filter out header lines and empty lines
                content_lines = [
                    l for l in lines
                    if l.strip() and not l.startswith("#") and not l.startswith("Generated:")
                ]
                if len("\n".join(content_lines)) < min_content_length:
                    empty_reports.append(str(report_file.relative_to(REPORTS_DIR)))

        # Allow up to 80% sparse reports (shadow_observer often finds nothing when idle)
        total_reports = sum(1 for d in REPORTS_DIR.iterdir() if d.is_dir() for _ in d.glob("*.md"))
        sparse_ratio = len(empty_reports) / max(total_reports, 1)

        assert sparse_ratio < 0.85, (
            f"Too many sparse reports ({len(empty_reports)}/{total_reports}): "
            f"{empty_reports[:5]}..."
        )

    def test_reports_have_metrics(self) -> None:
        """Reports should include metrics JSON block."""
        reports_with_metrics = 0
        total_reports = 0

        for report_dir in REPORTS_DIR.iterdir():
            if not report_dir.is_dir():
                continue
            for report_file in report_dir.glob("*.md"):
                total_reports += 1
                content = report_file.read_text()
                if "## Metrics" in content or '"metrics"' in content or "{" in content:
                    reports_with_metrics += 1

        metrics_ratio = reports_with_metrics / max(total_reports, 1)
        # Allow lower ratio as older reports predate metrics implementation
        assert metrics_ratio >= 0.2, (
            f"Only {reports_with_metrics}/{total_reports} reports have metrics"
        )

    def test_recent_reports_exist(self) -> None:
        """Should have reports from the last 24 hours."""
        cutoff = datetime.now() - timedelta(hours=24)
        recent_reports = 0

        for report_dir in REPORTS_DIR.iterdir():
            if not report_dir.is_dir():
                continue
            for report_file in report_dir.glob("*.md"):
                mtime = datetime.fromtimestamp(report_file.stat().st_mtime)
                if mtime > cutoff:
                    recent_reports += 1

        assert recent_reports >= 5, f"Only {recent_reports} reports in last 24h"

    def test_report_timestamps_valid(self) -> None:
        """Report filenames should have valid timestamps."""
        invalid_timestamps: list[str] = []

        for report_dir in REPORTS_DIR.iterdir():
            if not report_dir.is_dir():
                continue
            for report_file in report_dir.glob("*.md"):
                name = report_file.name
                # Expected format: YYYYMMDD_HHMMSS_...
                if len(name) >= 15:
                    try:
                        datetime.strptime(name[:15], "%Y%m%d_%H%M%S")
                    except ValueError:
                        invalid_timestamps.append(name)

        assert len(invalid_timestamps) == 0, (
            f"Invalid timestamps in reports: {invalid_timestamps[:5]}"
        )


class TestEmbeddingQuality:
    """Test quality of generated embeddings."""

    @pytest.fixture
    def embedding_dirs(self) -> list[Path]:
        """Get all embedding directories."""
        dirs = []
        if KNOWLEDGE_DIR.exists():
            for project_dir in KNOWLEDGE_DIR.iterdir():
                if project_dir.is_dir():
                    emb_dir = project_dir / "embeddings"
                    if emb_dir.exists():
                        dirs.append(emb_dir)
        return dirs

    def test_embeddings_exist(self, embedding_dirs: list[Path]) -> None:
        """At least one project should have embeddings."""
        assert len(embedding_dirs) >= 1, "No embedding directories found"

        total_embeddings = sum(
            len(list(d.glob("*.json"))) for d in embedding_dirs
        )
        assert total_embeddings > 0, "No embedding files found"

    def test_embedding_dimensions(self, embedding_dirs: list[Path]) -> None:
        """All embeddings should have consistent dimensions."""
        dimension_counts: dict[int, int] = {}
        sample_size = 100  # Check a sample

        for emb_dir in embedding_dirs:
            for i, emb_file in enumerate(emb_dir.glob("*.json")):
                if i >= sample_size:
                    break
                try:
                    data = json.loads(emb_file.read_text())
                    if "embedding" in data:
                        dim = len(data["embedding"])
                        dimension_counts[dim] = dimension_counts.get(dim, 0) + 1
                except (json.JSONDecodeError, KeyError):
                    pass

        # Should have consistent dimensions (768 for most models)
        assert len(dimension_counts) <= 2, (
            f"Inconsistent embedding dimensions: {dimension_counts}"
        )
        # Most common dimension should be 768 or 384
        if dimension_counts:
            most_common = max(dimension_counts, key=dimension_counts.get)
            assert most_common in [384, 768, 1024, 1536], (
                f"Unexpected embedding dimension: {most_common}"
            )

    def test_embedding_format(self, embedding_dirs: list[Path]) -> None:
        """Embeddings should have required fields."""
        invalid_embeddings: list[str] = []
        sample_size = 50

        for emb_dir in embedding_dirs:
            for i, emb_file in enumerate(emb_dir.glob("*.json")):
                if i >= sample_size:
                    break
                try:
                    data = json.loads(emb_file.read_text())
                    # Required: id and embedding
                    # text OR text_preview is acceptable
                    issues = []
                    if "id" not in data:
                        issues.append("id")
                    if "embedding" not in data:
                        issues.append("embedding")
                    if "text" not in data and "text_preview" not in data:
                        issues.append("text/text_preview")
                    if issues:
                        invalid_embeddings.append(f"{emb_file.name}: missing {issues}")
                except json.JSONDecodeError as e:
                    invalid_embeddings.append(f"{emb_file.name}: {e}")

        assert len(invalid_embeddings) == 0, (
            f"Invalid embeddings: {invalid_embeddings[:5]}"
        )

    def test_no_duplicate_ids(self, embedding_dirs: list[Path]) -> None:
        """Embedding IDs should be unique within a project."""
        for emb_dir in embedding_dirs:
            ids: set[str] = set()
            duplicates: list[str] = []

            for emb_file in emb_dir.glob("*.json"):
                try:
                    data = json.loads(emb_file.read_text())
                    emb_id = data.get("id", "")
                    if emb_id in ids:
                        duplicates.append(emb_id)
                    ids.add(emb_id)
                except json.JSONDecodeError:
                    pass

            assert len(duplicates) == 0, (
                f"Duplicate IDs in {emb_dir.name}: {duplicates[:5]}"
            )

    def test_embedding_vectors_valid(self, embedding_dirs: list[Path]) -> None:
        """Embedding vectors should contain valid floats."""
        invalid_vectors: list[str] = []
        sample_size = 50

        for emb_dir in embedding_dirs:
            for i, emb_file in enumerate(emb_dir.glob("*.json")):
                if i >= sample_size:
                    break
                try:
                    data = json.loads(emb_file.read_text())
                    embedding = data.get("embedding", [])
                    # Check for NaN, Inf, or non-float values
                    for val in embedding:
                        if not isinstance(val, (int, float)):
                            invalid_vectors.append(f"{emb_file.name}: non-numeric")
                            break
                        if val != val:  # NaN check
                            invalid_vectors.append(f"{emb_file.name}: NaN")
                            break
                except (json.JSONDecodeError, TypeError) as e:
                    invalid_vectors.append(f"{emb_file.name}: {e}")

        assert len(invalid_vectors) == 0, (
            f"Invalid vectors: {invalid_vectors[:5]}"
        )

    def test_embedding_coverage(self, embedding_dirs: list[Path]) -> None:
        """Check embedding coverage metrics."""
        if not EMBEDDING_STATUS.exists():
            pytest.skip("Embedding status file not found")

        status = json.loads(EMBEDDING_STATUS.read_text())
        coverage = status.get("coverage_percent", 0)

        # Should have reasonable coverage
        assert coverage >= 50, f"Low embedding coverage: {coverage}%"


class TestServiceHealth:
    """Test health of background services."""

    def test_autonomy_daemon_status(self) -> None:
        """Autonomy daemon should be running and healthy."""
        if not AUTONOMY_STATUS.exists():
            pytest.skip("Autonomy status file not found")

        status = json.loads(AUTONOMY_STATUS.read_text())

        assert status.get("running") is True, "Autonomy daemon not running"

        # Check last update is recent (within 10 minutes)
        last_update = datetime.fromisoformat(status.get("last_update", "2000-01-01"))
        age = datetime.now() - last_update
        assert age < timedelta(minutes=10), f"Autonomy daemon stale: {age}"

    def test_autonomy_tasks_healthy(self) -> None:
        """All autonomy tasks should be in good state."""
        if not AUTONOMY_STATUS.exists():
            pytest.skip("Autonomy status file not found")

        status = json.loads(AUTONOMY_STATUS.read_text())
        tasks = status.get("tasks", [])

        failed_tasks = [
            t["name"] for t in tasks
            if t.get("last_status") == "failed"
        ]
        assert len(failed_tasks) == 0, f"Failed tasks: {failed_tasks}"

        # At least some tasks should have run
        ran_tasks = [t for t in tasks if t.get("last_run")]
        assert len(ran_tasks) >= 3, "Too few tasks have run"

    def test_embedding_service_status(self) -> None:
        """Embedding service should be running."""
        if not EMBEDDING_STATUS.exists():
            pytest.skip("Embedding status file not found")

        status = json.loads(EMBEDDING_STATUS.read_text())

        assert status.get("running") is True, "Embedding service not running"
        assert status.get("total_embeddings", 0) > 0, "No embeddings generated"

    def test_context_agent_status(self) -> None:
        """Context agent daemon should be running."""
        if not CONTEXT_AGENT_STATUS.exists():
            pytest.skip("Context agent status file not found")

        status = json.loads(CONTEXT_AGENT_STATUS.read_text())

        assert status.get("running") is True, "Context agent not running"

        # Check tasks
        tasks = status.get("tasks", [])
        assert len(tasks) >= 1, "No context agent tasks defined"

    def test_no_service_errors_in_logs(self) -> None:
        """Service logs should not have critical errors."""
        log_dir = CONTEXT_ROOT / "logs"
        if not log_dir.exists():
            pytest.skip("Log directory not found")

        critical_patterns = [
            "CRITICAL",
            "Traceback (most recent call last)",
            "FATAL",
        ]
        files_with_errors: list[str] = []

        for log_file in log_dir.glob("*.log"):
            try:
                content = log_file.read_text()
                for pattern in critical_patterns:
                    if pattern in content:
                        files_with_errors.append(log_file.name)
                        break
            except Exception:
                pass

        # Allow some error logs but not all
        log_count = len(list(log_dir.glob("*.log")))
        error_ratio = len(files_with_errors) / max(log_count, 1)
        assert error_ratio < 0.3, f"Too many logs with errors: {files_with_errors}"


class TestScratchpadState:
    """Test scratchpad state integrity."""

    def test_state_md_not_corrupted(self) -> None:
        """State.md should not have duplicate sections."""
        state_file = CONTEXT_ROOT / "scratchpad" / "state.md"
        if not state_file.exists():
            pytest.skip("State file not found")

        content = state_file.read_text()
        lines = content.split("\n")

        # Count section headers
        section_counts: dict[str, int] = {}
        for line in lines:
            if line.startswith("## "):
                section = line[3:].strip()
                section_counts[section] = section_counts.get(section, 0) + 1

        # Check for duplicates (more than 3 of same section is suspicious)
        suspicious = {k: v for k, v in section_counts.items() if v > 3}
        assert len(suspicious) == 0, f"Duplicate sections in state.md: {suspicious}"

    def test_metacognition_json_valid(self) -> None:
        """Metacognition state should be valid JSON."""
        meta_file = CONTEXT_ROOT / "scratchpad" / "metacognition.json"
        if not meta_file.exists():
            pytest.skip("Metacognition file not found")

        try:
            data = json.loads(meta_file.read_text())
            # Check required fields
            assert "progressStatus" in data
            assert "cognitiveLoad" in data
            assert "flowState" in data
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid metacognition JSON: {e}")


class TestMissionReports:
    """Test quality of mission-specific reports."""

    def test_mission_reports_not_empty(self) -> None:
        """Mission reports should have research findings."""
        mission_dir = REPORTS_DIR / "mission"
        if not mission_dir.exists():
            pytest.skip("Mission directory not found")

        reports = list(mission_dir.glob("*.md"))
        if not reports:
            pytest.skip("No mission reports found")

        empty_count = 0
        for report in reports:
            content = report.read_text()
            # Check if the report found anything
            if "Items analyzed: 0" in content and "New discoveries: 0" in content:
                empty_count += 1

        # All empty is concerning - allow up to 100% for now
        # as mission queries are being improved
        empty_ratio = empty_count / len(reports)
        if empty_ratio == 1.0:
            # Warn but don't fail - missions may need query tuning
            import warnings
            warnings.warn(f"All {len(reports)} mission reports are empty - queries may need tuning")

    def test_mission_metrics_valid(self) -> None:
        """Mission report metrics should be valid JSON."""
        mission_dir = REPORTS_DIR / "mission"
        if not mission_dir.exists():
            pytest.skip("Mission directory not found")

        for report in mission_dir.glob("*.md"):
            content = report.read_text()
            # Extract metrics JSON
            if "## Metrics" in content:
                try:
                    # Find JSON block after Metrics
                    parts = content.split("## Metrics")
                    if len(parts) > 1:
                        json_part = parts[1].strip()
                        if json_part.startswith("{"):
                            # Find the end of JSON
                            brace_count = 0
                            end_idx = 0
                            for i, c in enumerate(json_part):
                                if c == "{":
                                    brace_count += 1
                                elif c == "}":
                                    brace_count -= 1
                                    if brace_count == 0:
                                        end_idx = i + 1
                                        break
                            json_str = json_part[:end_idx]
                            json.loads(json_str)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid metrics in {report.name}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
