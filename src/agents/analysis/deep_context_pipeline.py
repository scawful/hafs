"""Deep context analysis and smart ML pipeline helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from agents.core.base import BaseAgent
from hafs.core.orchestration import OrchestrationPipeline, PipelineContext, PipelineStep

logger = logging.getLogger(__name__)


DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".context",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".cache",
    "node_modules",
    "dist",
    "build",
}

DEFAULT_TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".rs",
    ".go",
    ".java",
    ".sh",
    ".zsh",
}

DOC_EXTENSIONS = {
    ".md",
    ".txt",
    ".rst",
    ".adoc",
}

TODO_PATTERN = re.compile(r"\b(TODO|FIXME|HACK|XXX)\b", re.IGNORECASE)
DOC_HEADING_PATTERN = re.compile(r"^(#{1,4})\s+(.*)")


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _count_json_entries(data: Any) -> int:
    if isinstance(data, dict):
        return len(data)
    if isinstance(data, list):
        return len(data)
    return 0


def _iter_files(
    root: Path,
    *,
    exclude_dirs: set[str],
    max_files: int,
) -> Iterable[Path]:
    count = 0
    for path in root.rglob("*"):
        if count >= max_files:
            break
        if not path.is_file():
            continue
        if exclude_dirs.intersection(path.parts):
            continue
        count += 1
        yield path


def _is_text_candidate(path: Path) -> bool:
    return path.suffix.lower() in DEFAULT_TEXT_EXTENSIONS


def _is_doc_candidate(path: Path) -> bool:
    return path.suffix.lower() in DOC_EXTENSIONS


def _extract_headings(text: str, max_headings: int = 6) -> list[str]:
    headings: list[str] = []
    for line in text.splitlines():
        match = DOC_HEADING_PATTERN.match(line.strip())
        if not match:
            continue
        heading = match.group(2).strip()
        if heading:
            headings.append(heading)
        if len(headings) >= max_headings:
            break
    return headings


def _safe_run_git(root: Path, args: list[str], timeout: float = 2.0) -> str:
    if shutil.which("git") is None:
        return ""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return ""


class RepoSnapshotAgent(BaseAgent):
    """Collect repository-level context signals."""

    def __init__(
        self,
        repo_root: Path,
        *,
        exclude_dirs: Optional[set[str]] = None,
    ) -> None:
        super().__init__("RepoSnapshotAgent", "Summarize repository structure and activity.")
        self.repo_root = repo_root
        self.exclude_dirs = exclude_dirs or set(DEFAULT_EXCLUDE_DIRS)

    def snapshot(self, max_files: int = 5000) -> dict[str, Any]:
        total_files = 0
        total_bytes = 0
        ext_counts: dict[str, int] = {}
        top_dirs: dict[str, int] = {}
        largest_files: list[dict[str, Any]] = []
        recent_files: list[dict[str, Any]] = []

        for path in _iter_files(self.repo_root, exclude_dirs=self.exclude_dirs, max_files=max_files):
            total_files += 1
            try:
                stat = path.stat()
            except OSError:
                continue

            total_bytes += stat.st_size
            ext = path.suffix.lower() or "<none>"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

            rel = self._rel_path(path)
            top_dir = rel.split("/", 1)[0] if "/" in rel else rel
            if top_dir:
                top_dirs[top_dir] = top_dirs.get(top_dir, 0) + 1

            largest_files.append({"path": rel, "size": stat.st_size})
            recent_files.append({"path": rel, "mtime": stat.st_mtime})

        largest_files.sort(key=lambda item: item["size"], reverse=True)
        recent_files.sort(key=lambda item: item["mtime"], reverse=True)

        return {
            "repo_root": str(self.repo_root),
            "total_files": total_files,
            "total_bytes": total_bytes,
            "extensions": dict(sorted(ext_counts.items(), key=lambda item: item[1], reverse=True)[:12]),
            "top_dirs": dict(sorted(top_dirs.items(), key=lambda item: item[1], reverse=True)[:12]),
            "largest_files": largest_files[:8],
            "recent_files": recent_files[:8],
        }

    def _rel_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.repo_root).as_posix()
        except ValueError:
            return str(path)


class ContextSignalAgent(BaseAgent):
    """Collect context signals like TODO hotspots and git activity."""

    def __init__(
        self,
        repo_root: Path,
        *,
        exclude_dirs: Optional[set[str]] = None,
    ) -> None:
        super().__init__("ContextSignalAgent", "Scan repo for actionable context signals.")
        self.repo_root = repo_root
        self.exclude_dirs = exclude_dirs or set(DEFAULT_EXCLUDE_DIRS)

    def collect_signals(
        self,
        *,
        max_files: int = 2500,
        max_hits: int = 120,
        max_file_kb: int = 256,
    ) -> dict[str, Any]:
        todos: list[dict[str, Any]] = []

        for path in _iter_files(self.repo_root, exclude_dirs=self.exclude_dirs, max_files=max_files):
            if not _is_text_candidate(path):
                continue
            try:
                if path.stat().st_size > max_file_kb * 1024:
                    continue
            except OSError:
                continue

            try:
                content = path.read_text(errors="ignore")
            except OSError:
                continue

            for idx, line in enumerate(content.splitlines(), start=1):
                if len(todos) >= max_hits:
                    break
                match = TODO_PATTERN.search(line)
                if match:
                    todos.append(
                        {
                            "path": self._rel_path(path),
                            "line": idx,
                            "tag": match.group(1).upper(),
                            "text": line.strip()[:160],
                        }
                    )

            if len(todos) >= max_hits:
                break

        git_info = self._git_summary()

        return {
            "todo_count": len(todos),
            "todo_hits": todos,
            "git": git_info,
        }

    def _git_summary(self) -> dict[str, Any]:
        top_level = _safe_run_git(self.repo_root, ["rev-parse", "--show-toplevel"])
        if not top_level:
            return {"available": False}

        branch = _safe_run_git(self.repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
        status = _safe_run_git(self.repo_root, ["status", "--porcelain"])
        recent = _safe_run_git(self.repo_root, ["log", "-n", "5", "--oneline"])

        dirty_files = [line[3:] for line in status.splitlines() if line.strip()]

        return {
            "available": True,
            "branch": branch or "unknown",
            "dirty_count": len(dirty_files),
            "dirty_files": dirty_files[:15],
            "recent_commits": recent.splitlines() if recent else [],
        }

    def _rel_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.repo_root).as_posix()
        except ValueError:
            return str(path)


class KnowledgeCoverageAgent(BaseAgent):
    """Summarize knowledge base coverage and embedding profiles."""

    def __init__(
        self,
        context_root: Path,
        *,
        kb_names: Optional[list[str]] = None,
    ) -> None:
        super().__init__(
            "KnowledgeCoverageAgent",
            "Measure knowledge base coverage and embedding completeness.",
        )
        self.context_root = context_root
        self.kb_names = kb_names

    def collect(self) -> dict[str, Any]:
        knowledge_root = self.context_root / "knowledge"
        if not knowledge_root.exists():
            return {"knowledge_root": str(knowledge_root), "bases": [], "summary": {}}

        project_profiles = self._load_project_profiles()
        names = self.kb_names or sorted(
            path.name for path in knowledge_root.iterdir() if path.is_dir()
        )

        bases: list[dict[str, Any]] = []
        for name in names:
            kb_dir = knowledge_root / name
            if not kb_dir.exists():
                continue

            symbols = _count_json_entries(_safe_read_json(kb_dir / "symbols.json"))
            routines = _count_json_entries(_safe_read_json(kb_dir / "routines.json"))
            total_items = symbols + routines

            profiles = self._load_embedding_profiles(kb_dir, total_items)
            expected = project_profiles.get(name)
            if expected and expected.get("storage_id"):
                profiles = self._ensure_expected_profile(
                    profiles,
                    expected["storage_id"],
                    total_items,
                )

            best_coverage = max(
                (profile["coverage_ratio"] for profile in profiles),
                default=0.0,
            )

            bases.append(
                {
                    "name": name,
                    "symbols": symbols,
                    "routines": routines,
                    "total_items": total_items,
                    "embedding_profiles": profiles,
                    "expected_profile": expected or None,
                    "best_coverage": round(best_coverage, 4),
                }
            )

        summary = self._summarize(bases)
        return {
            "knowledge_root": str(knowledge_root),
            "bases": bases,
            "summary": summary,
        }

    def _summarize(self, bases: list[dict[str, Any]]) -> dict[str, Any]:
        total_items = sum(base.get("total_items", 0) for base in bases)
        total_profiles = sum(len(base.get("embedding_profiles", [])) for base in bases)
        total_embeddings = sum(
            profile.get("embeddings", 0)
            for base in bases
            for profile in base.get("embedding_profiles", [])
        )
        missing = [
            base["name"]
            for base in bases
            if base.get("total_items", 0) > 0 and not base.get("embedding_profiles")
        ]
        return {
            "bases": len(bases),
            "total_items": total_items,
            "total_embeddings": total_embeddings,
            "profiles": total_profiles,
            "bases_missing_embeddings": missing,
        }

    def _load_project_profiles(self) -> dict[str, dict[str, Any]]:
        projects_file = self.context_root / "embedding_service" / "projects.json"
        data = _safe_read_json(projects_file)
        if not isinstance(data, dict):
            return {}

        try:
            from hafs.core.embeddings import BatchEmbeddingManager
        except Exception:
            BatchEmbeddingManager = None  # type: ignore[assignment]

        profiles: dict[str, dict[str, Any]] = {}
        for name, payload in data.items():
            provider = payload.get("embedding_provider")
            model = payload.get("embedding_model")
            storage_id = None
            if BatchEmbeddingManager is not None:
                storage_id = BatchEmbeddingManager.resolve_storage_id(provider, model)
            profiles[name] = {
                "provider": provider,
                "model": model,
                "storage_id": storage_id,
                "project_type": payload.get("project_type"),
            }
        return profiles

    def _ensure_expected_profile(
        self,
        profiles: list[dict[str, Any]],
        storage_id: str,
        total_items: int,
    ) -> list[dict[str, Any]]:
        for profile in profiles:
            if profile.get("name") == storage_id:
                return profiles

        profiles.append(self._profile_summary(storage_id, 0, total_items))
        return profiles

    def _profile_summary(
        self,
        name: str,
        embeddings: int,
        total_items: int,
    ) -> dict[str, Any]:
        backlog = max(total_items - embeddings, 0)
        coverage_ratio = embeddings / max(total_items, 1)
        return {
            "name": name,
            "embeddings": embeddings,
            "coverage_ratio": round(coverage_ratio, 4),
            "backlog": backlog,
        }

    def _load_embedding_profiles(
        self,
        kb_dir: Path,
        total_items: int,
    ) -> list[dict[str, Any]]:
        profiles: list[dict[str, Any]] = []

        index_files = sorted(kb_dir.glob("embedding_index*.json"))
        for index_file in index_files:
            if index_file.name == "embedding_index.json":
                profile_name = "default"
            else:
                profile_name = index_file.stem.replace("embedding_index_", "")
            count = _count_json_entries(_safe_read_json(index_file))
            profiles.append(self._profile_summary(profile_name, count, total_items))

        if profiles:
            return profiles

        embeddings_dir = kb_dir / "embeddings"
        if not embeddings_dir.exists():
            return profiles

        root_count = sum(
            1
            for path in embeddings_dir.glob("*.json")
            if path.is_file()
        )
        if root_count:
            profiles.append(self._profile_summary("default", root_count, total_items))

        for subdir in sorted(path for path in embeddings_dir.iterdir() if path.is_dir()):
            count = sum(
                1
                for path in subdir.glob("*.json")
                if path.is_file()
            )
            profiles.append(self._profile_summary(subdir.name, count, total_items))

        return profiles


class DocIndexAgent(BaseAgent):
    """Index documentation files for context building."""

    def __init__(
        self,
        repo_root: Path,
        *,
        exclude_dirs: Optional[set[str]] = None,
    ) -> None:
        super().__init__(
            "DocIndexAgent",
            "Scan documentation for headings and coverage.",
        )
        self.repo_root = repo_root
        self.exclude_dirs = exclude_dirs or set(DEFAULT_EXCLUDE_DIRS)

    def collect(
        self,
        *,
        max_files: int = 1500,
        max_file_kb: int = 512,
        sample_limit: int = 12,
    ) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        ext_counts: dict[str, int] = {}
        top_dirs: dict[str, int] = {}

        for path in _iter_files(self.repo_root, exclude_dirs=self.exclude_dirs, max_files=max_files):
            if not _is_doc_candidate(path):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size > max_file_kb * 1024:
                continue

            rel = self._rel_path(path)
            ext = path.suffix.lower() or "<none>"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
            top_dir = rel.split("/", 1)[0] if "/" in rel else rel
            if top_dir:
                top_dirs[top_dir] = top_dirs.get(top_dir, 0) + 1

            headings: list[str] = []
            title = path.stem
            try:
                content = path.read_text(errors="ignore")[:12000]
                headings = _extract_headings(content)
                if headings:
                    title = headings[0]
            except OSError:
                pass

            entries.append(
                {
                    "path": rel,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "title": title,
                    "headings": headings[:6],
                }
            )

        largest_docs = sorted(entries, key=lambda item: item["size"], reverse=True)[:8]
        recent_docs = sorted(entries, key=lambda item: item["mtime"], reverse=True)[:8]
        samples = [entry for entry in entries if entry.get("headings")][:sample_limit]

        return {
            "total_docs": len(entries),
            "extensions": dict(sorted(ext_counts.items(), key=lambda item: item[1], reverse=True)[:12]),
            "top_dirs": dict(sorted(top_dirs.items(), key=lambda item: item[1], reverse=True)[:12]),
            "largest_docs": largest_docs,
            "recent_docs": recent_docs,
            "samples": samples,
        }

    def _rel_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.repo_root).as_posix()
        except ValueError:
            return str(path)


class MLSignalAgent(BaseAgent):
    """Collect ML pipeline signals from context state files."""

    def __init__(self, context_root: Path) -> None:
        super().__init__("MLSignalAgent", "Capture embedding/training status for ML pipelines.")
        self.context_root = context_root

    def collect_signals(self) -> dict[str, Any]:
        embedding_root = self.context_root / "embedding_service"
        training_root = self.context_root / "training"

        daemon_status = _safe_read_json(embedding_root / "daemon_status.json")
        post_status = _safe_read_json(embedding_root / "post_completion.json")
        service_projects = _safe_read_json(embedding_root / "projects.json")
        service_status = _safe_read_json(embedding_root / "status.json")
        service_queue_raw = _safe_read_json(embedding_root / "queue.json")
        service_queue = service_queue_raw if isinstance(service_queue_raw, list) else []

        project_summaries: list[dict[str, Any]] = []
        if isinstance(service_projects, dict):
            for name, config in service_projects.items():
                progress = {}
                if isinstance(service_status, dict):
                    progress = service_status.get(name, {}) or {}
                total_items = int(progress.get("total_items") or 0)
                processed = int(progress.get("processed_items") or 0)
                backlog = max(total_items - processed, 0)
                project_summaries.append(
                    {
                        "name": name,
                        "status": progress.get("status", "unknown"),
                        "total_items": total_items,
                        "processed_items": processed,
                        "failed_items": int(progress.get("failed_items") or 0),
                        "provider": config.get("embedding_provider"),
                        "model": config.get("embedding_model"),
                        "project_type": config.get("project_type"),
                        "backlog": backlog,
                    }
                )

        training_statuses: list[dict[str, Any]] = []
        if training_root.exists():
            for path in sorted(training_root.glob("*.json")):
                payload = _safe_read_json(path)
                if payload:
                    training_statuses.append(
                        {"file": path.name, "data": payload}
                    )

        return {
            "embedding_daemon": daemon_status,
            "embedding_post": post_status,
            "embedding_service": {
                "projects": project_summaries,
                "queue": service_queue,
            },
            "training_status": training_statuses,
        }


class NodeHealthAgent(BaseAgent):
    """Collect distributed node health data."""

    def __init__(self) -> None:
        super().__init__("NodeHealthAgent", "Assess compute node availability.")

    async def collect(self) -> dict[str, Any]:
        try:
            from hafs.core.nodes import NodeStatus, NodeManager
        except Exception as exc:  # pragma: no cover - depends on optional deps
            return {"status": "error", "error": str(exc)}

        manager = NodeManager()
        try:
            await manager.load_config()
            await manager.health_check_all()
            summary = manager.summary()
            offline = [
                node.name
                for node in manager.nodes
                if node.status in {NodeStatus.OFFLINE, NodeStatus.ERROR}
            ]
            online = [node.name for node in manager.nodes if node.name not in offline]
            return {
                "status": "ok",
                "summary": summary,
                "offline": offline,
                "online": online,
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
        finally:
            await manager.close()


class MLPipelinePlannerAgent(BaseAgent):
    """Plan smart ML actions based on observed signals."""

    def __init__(self) -> None:
        super().__init__("MLPipelinePlannerAgent", "Recommend ML pipeline actions.")

    def plan(
        self,
        ml_signals: dict[str, Any],
        context_signals: Optional[dict[str, Any]] = None,
        node_health: Optional[dict[str, Any]] = None,
        kb_coverage: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        actions: list[str] = []
        seen: set[str] = set()

        def _add_action(text: str) -> None:
            if text in seen:
                return
            if len(actions) >= 8:
                return
            actions.append(text)
            seen.add(text)

        embedding = ml_signals.get("embedding_daemon", {}) if ml_signals else {}
        total_symbols = embedding.get("total_symbols") or embedding.get("total_embeddings_target")
        total_embeddings = embedding.get("total_embeddings")
        running = embedding.get("running")

        if total_symbols is not None and total_embeddings is not None:
            backlog = max(0, int(total_symbols) - int(total_embeddings))
            if backlog > 0 and not running:
                _add_action("Start embedding daemon to reduce backlog.")
            elif backlog > 0:
                _add_action("Increase embedding throughput to clear backlog.")
        elif not embedding:
            _add_action("Configure embedding daemon status reporting.")

        service = ml_signals.get("embedding_service", {}) if ml_signals else {}
        for project in service.get("projects", []):
            backlog = int(project.get("backlog") or 0)
            if backlog <= 0:
                continue
            status = str(project.get("status", "")).lower()
            provider = project.get("provider") or "default"
            model = project.get("model")
            detail = f"{provider}/{model}" if model else provider
            if status != "running":
                _add_action(
                    f"Queue embedding run for {project.get('name', 'project')} ({detail}) to clear {backlog} items."
                )
            else:
                _add_action(
                    f"Monitor embedding progress for {project.get('name', 'project')} ({detail})."
                )

        training_status = ml_signals.get("training_status", []) if ml_signals else []
        if not training_status:
            _add_action("Capture training status snapshots for pipeline planning.")
        else:
            _add_action("Review training metrics and schedule quality audits.")

        if context_signals:
            todo_count = context_signals.get("todo_count", 0)
            if todo_count > 25:
                _add_action("Prioritize TODO hotspots for context enrichment.")

        if kb_coverage:
            for base in kb_coverage.get("bases", []):
                total_items = int(base.get("total_items") or 0)
                if total_items <= 0:
                    continue
                profiles = base.get("embedding_profiles", [])
                best_coverage = float(base.get("best_coverage") or 0.0)
                expected = base.get("expected_profile") or {}
                profile_note = ""
                if expected.get("storage_id"):
                    profile_note = f" profile {expected['storage_id']}"
                if not profiles:
                    _add_action(f"Generate embeddings for KB {base.get('name', 'unknown')} (no profiles found).")
                elif best_coverage < 0.5:
                    _add_action(
                        f"Increase embeddings for KB {base.get('name', 'unknown')}{profile_note} "
                        f"(coverage {best_coverage:.0%})."
                    )

        if node_health and node_health.get("offline"):
            _add_action("Investigate offline nodes and restore connectivity.")

        return actions


class DeepSynthesisAgent(BaseAgent):
    """Synthesize deep context and ML signals into a report."""

    def __init__(self, reports_root: Path, *, llm_enabled: bool) -> None:
        super().__init__("DeepSynthesisAgent", "Summarize deep context signals.")
        self.reports_root = reports_root
        self.llm_enabled = llm_enabled
        self.model_tier = "reasoning"

    async def synthesize(
        self,
        topic: str,
        *,
        repo_snapshot: dict[str, Any],
        context_signals: dict[str, Any],
        ml_signals: dict[str, Any],
        node_health: dict[str, Any],
        kb_coverage: Optional[dict[str, Any]] = None,
        doc_index: Optional[dict[str, Any]] = None,
        recommendations: list[str],
    ) -> str:
        base_report = self._render_markdown(
            topic,
            repo_snapshot,
            context_signals,
            ml_signals,
            node_health,
            kb_coverage or {},
            doc_index or {},
            recommendations,
        )

        if not self.llm_enabled:
            return base_report

        prompt = (
            f"Produce a deep analysis report in markdown.\n\n"
            f"TOPIC: {topic}\n\n"
            f"SNAPSHOT:\n{json.dumps(repo_snapshot, indent=2)[:4000]}\n\n"
            f"CONTEXT SIGNALS:\n{json.dumps(context_signals, indent=2)[:4000]}\n\n"
            f"ML SIGNALS:\n{json.dumps(ml_signals, indent=2)[:4000]}\n\n"
            f"NODE HEALTH:\n{json.dumps(node_health, indent=2)[:2000]}\n\n"
            f"KB COVERAGE:\n{json.dumps(kb_coverage or {}, indent=2)[:3000]}\n\n"
            f"DOC INDEX:\n{json.dumps(doc_index or {}, indent=2)[:3000]}\n\n"
            f"RECOMMENDATIONS:\n{json.dumps(recommendations, indent=2)}\n\n"
            "Deliver sections: Executive Summary, Risks, Opportunities, Next Actions."
        )

        try:
            analysis = await self.generate_thought(prompt)
        except Exception as exc:
            logger.warning("Deep synthesis LLM unavailable: %s", exc)
            return base_report

        return f"{base_report}\n\n## LLM Deep Analysis\n\n{analysis}"

    def _render_markdown(
        self,
        topic: str,
        repo_snapshot: dict[str, Any],
        context_signals: dict[str, Any],
        ml_signals: dict[str, Any],
        node_health: dict[str, Any],
        kb_coverage: dict[str, Any],
        doc_index: dict[str, Any],
        recommendations: list[str],
    ) -> str:
        timestamp = datetime.now().isoformat(timespec="seconds")
        todos = context_signals.get("todo_count", 0)
        git_info = context_signals.get("git", {})
        backlog = "-"
        embedding = ml_signals.get("embedding_daemon", {}) if ml_signals else {}
        if embedding and embedding.get("total_symbols") is not None and embedding.get("total_embeddings") is not None:
            backlog = max(0, int(embedding["total_symbols"]) - int(embedding["total_embeddings"]))

        lines = [
            f"# Deep Context Report",
            f"",
            f"Topic: {topic}",
            f"Generated: {timestamp}",
            f"",
            "## Repository Snapshot",
            f"- Root: {repo_snapshot.get('repo_root', '-')}",
            f"- Total files: {repo_snapshot.get('total_files', 0)}",
            f"- Total size: {repo_snapshot.get('total_bytes', 0)} bytes",
            f"- Extensions: {', '.join(repo_snapshot.get('extensions', {}).keys()) or '-'}",
            f"",
            "## Context Signals",
            f"- TODO/FIXME hits: {todos}",
            f"- Git branch: {git_info.get('branch', 'n/a')}",
            f"- Dirty files: {git_info.get('dirty_count', 0)}",
            f"",
            "## ML Pipeline Signals",
            f"- Embedding backlog: {backlog}",
            f"- Training status files: {len(ml_signals.get('training_status', []))}",
            f"",
            "## Knowledge Coverage",
            f"- KB bases: {kb_coverage.get('summary', {}).get('bases', 0)}",
            f"- Total KB items: {kb_coverage.get('summary', {}).get('total_items', 0)}",
            f"- Embedding profiles: {kb_coverage.get('summary', {}).get('profiles', 0)}",
            f"",
            "## Documentation Index",
            f"- Total docs: {doc_index.get('total_docs', 0)}",
            f"- Top dirs: {', '.join(doc_index.get('top_dirs', {}).keys()) or '-'}",
            f"",
            "## Knowledge Coverage Detail",
        ]
        bases = kb_coverage.get("bases", [])
        if bases:
            for base in bases[:5]:
                best = base.get("best_coverage", 0)
                lines.append(
                    f"- KB {base.get('name', 'unknown')}: {base.get('total_items', 0)} items, "
                    f"best coverage {best:.0%}"
                )
        else:
            lines.append("- No KB coverage data available.")
        lines.extend(
            [
                "",
                "## Node Health",
                f"- Status: {node_health.get('status', 'unknown')}",
                f"- Offline nodes: {len(node_health.get('offline', []))}",
                "",
                "## Recommended Actions",
            ]
        )
        if recommendations:
            lines.extend([f"- {item}" for item in recommendations])
        else:
            lines.append("- No immediate actions recommended.")
        lines.append("")
        return "\n".join(lines)


@dataclass
class DeepContext(PipelineContext):
    """Pipeline context for deep analysis."""

    repo_root: Path = field(default_factory=Path.cwd)
    repo_snapshot: dict[str, Any] = field(default_factory=dict)
    context_signals: dict[str, Any] = field(default_factory=dict)
    ml_signals: dict[str, Any] = field(default_factory=dict)
    kb_coverage: dict[str, Any] = field(default_factory=dict)
    doc_index: dict[str, Any] = field(default_factory=dict)
    node_health: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    report: str = ""
    report_path: Optional[Path] = None


class DeepContextPipeline(BaseAgent):
    """Orchestrates deep context analysis for a repo."""

    def __init__(
        self,
        *,
        repo_root: Optional[Path] = None,
        context_root: Optional[Path] = None,
        reports_root: Optional[Path] = None,
        check_nodes: bool = True,
        llm_summary: bool = True,
    ) -> None:
        super().__init__("DeepContextPipeline", "Run deep context analysis pipelines.")
        self.repo_root = (repo_root or Path.cwd()).expanduser().resolve()
        self.context_root = (context_root or Path.home() / ".context").expanduser()
        self.reports_root = (reports_root or (self.context_root / "reports")).expanduser()
        self.check_nodes = check_nodes
        self.llm_summary = llm_summary

        self._snapshot_agent = RepoSnapshotAgent(self.repo_root)
        self._signal_agent = ContextSignalAgent(self.repo_root)
        self._ml_agent = MLSignalAgent(self.context_root)
        self._kb_agent = KnowledgeCoverageAgent(self.context_root)
        self._doc_agent = DocIndexAgent(self.repo_root)
        self._planner = MLPipelinePlannerAgent()
        self._synthesizer = DeepSynthesisAgent(self.reports_root, llm_enabled=self._llm_enabled())

    def _llm_enabled(self) -> bool:
        if not self.llm_summary:
            return False
        return bool(os.environ.get("AISTUDIO_API_KEY") or os.environ.get("GEMINI_API_KEY"))

    async def setup(self) -> None:
        if self._llm_enabled():
            await super().setup()
            await self._synthesizer.setup()

    async def _step_snapshot(self, context: DeepContext) -> dict[str, Any]:
        context.repo_snapshot = self._snapshot_agent.snapshot()
        return context.repo_snapshot

    async def _step_signals(self, context: DeepContext) -> dict[str, Any]:
        context.context_signals = self._signal_agent.collect_signals()
        return context.context_signals

    async def _step_ml(self, context: DeepContext) -> dict[str, Any]:
        context.ml_signals = self._ml_agent.collect_signals()
        return context.ml_signals

    async def _step_kb(self, context: DeepContext) -> dict[str, Any]:
        context.kb_coverage = self._kb_agent.collect()
        return context.kb_coverage

    async def _step_docs(self, context: DeepContext) -> dict[str, Any]:
        context.doc_index = self._doc_agent.collect()
        return context.doc_index

    async def _step_nodes(self, context: DeepContext) -> dict[str, Any]:
        if not self.check_nodes:
            context.node_health = {"status": "skipped"}
            return context.node_health
        agent = NodeHealthAgent()
        context.node_health = await agent.collect()
        return context.node_health

    async def _step_plan(self, context: DeepContext) -> list[str]:
        context.recommendations = self._planner.plan(
            context.ml_signals,
            context_signals=context.context_signals,
            node_health=context.node_health,
            kb_coverage=context.kb_coverage,
        )
        return context.recommendations

    async def _step_summarize(self, context: DeepContext) -> str:
        context.report = await self._synthesizer.synthesize(
            context.topic,
            repo_snapshot=context.repo_snapshot,
            context_signals=context.context_signals,
            ml_signals=context.ml_signals,
            node_health=context.node_health,
            kb_coverage=context.kb_coverage,
            doc_index=context.doc_index,
            recommendations=context.recommendations,
        )
        context.report_path = self._write_report(context.topic, context.report)
        return context.report

    def _write_report(self, topic: str, report: str) -> Path:
        safe_topic = re.sub(r"[^a-zA-Z0-9_-]+", "_", topic).strip("_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = self.reports_root / "deep_analysis"
        reports_dir.mkdir(parents=True, exist_ok=True)
        filename = f"deep_context_{safe_topic or 'report'}_{timestamp}.md"
        path = reports_dir / filename
        path.write_text(report)
        return path

    async def generate_report(self, topic: str) -> dict[str, Any]:
        await self.setup()

        context = DeepContext(topic=topic, repo_root=self.repo_root)
        pipeline = OrchestrationPipeline(
            [
                PipelineStep(name="snapshot", kind="collect", run=self._step_snapshot),
                PipelineStep(name="signals", kind="collect", run=self._step_signals),
                PipelineStep(name="ml_signals", kind="collect", run=self._step_ml),
                PipelineStep(name="kb_coverage", kind="collect", run=self._step_kb),
                PipelineStep(name="doc_index", kind="collect", run=self._step_docs),
                PipelineStep(name="nodes", kind="observe", run=self._step_nodes, required=False),
                PipelineStep(name="plan", kind="plan", run=self._step_plan),
                PipelineStep(name="summarize", kind="summarize", run=self._step_summarize),
            ]
        )

        result = await pipeline.run(context)
        return {
            "topic": topic,
            "repo_root": str(self.repo_root),
            "snapshot": context.repo_snapshot,
            "signals": context.context_signals,
            "ml_signals": context.ml_signals,
            "kb_coverage": context.kb_coverage,
            "doc_index": context.doc_index,
            "node_health": context.node_health,
            "recommendations": context.recommendations,
            "report_path": str(context.report_path) if context.report_path else None,
            "report": context.report,
            "pipeline_status": [
                {"name": step.name, "status": step.status.value} for step in result.steps
            ],
        }

    async def run_task(self, task: str = "help") -> dict[str, Any]:
        if task == "help":
            return {"usage": ["report:TOPIC - Run deep context analysis"]}
        if task.startswith("report:"):
            topic = task[7:].strip()
            return await self.generate_report(topic)
        return {"error": f"Unknown task: {task}"}


@dataclass
class MLPlanContext(PipelineContext):
    """Pipeline context for ML pipeline planning."""

    ml_signals: dict[str, Any] = field(default_factory=dict)
    kb_coverage: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    report: str = ""
    report_path: Optional[Path] = None


class SmartMLPipeline(BaseAgent):
    """Generate ML pipeline recommendations from context signals."""

    def __init__(
        self,
        *,
        context_root: Optional[Path] = None,
        reports_root: Optional[Path] = None,
        llm_summary: bool = True,
    ) -> None:
        super().__init__("SmartMLPipeline", "Plan ML pipeline improvements.")
        self.context_root = (context_root or Path.home() / ".context").expanduser()
        self.reports_root = (reports_root or (self.context_root / "reports")).expanduser()
        self.llm_summary = llm_summary

        self._ml_agent = MLSignalAgent(self.context_root)
        self._kb_agent = KnowledgeCoverageAgent(self.context_root)
        self._planner = MLPipelinePlannerAgent()
        self._synthesizer = DeepSynthesisAgent(self.reports_root, llm_enabled=self._llm_enabled())

    def _llm_enabled(self) -> bool:
        if not self.llm_summary:
            return False
        return bool(os.environ.get("AISTUDIO_API_KEY") or os.environ.get("GEMINI_API_KEY"))

    async def setup(self) -> None:
        if self._llm_enabled():
            await super().setup()
            await self._synthesizer.setup()

    async def _step_signals(self, context: MLPlanContext) -> dict[str, Any]:
        context.ml_signals = self._ml_agent.collect_signals()
        return context.ml_signals

    async def _step_kb(self, context: MLPlanContext) -> dict[str, Any]:
        context.kb_coverage = self._kb_agent.collect()
        return context.kb_coverage

    async def _step_plan(self, context: MLPlanContext) -> list[str]:
        context.recommendations = self._planner.plan(
            context.ml_signals,
            kb_coverage=context.kb_coverage,
        )
        return context.recommendations

    async def _step_summarize(self, context: MLPlanContext) -> str:
        context.report = await self._synthesizer.synthesize(
            context.topic,
            repo_snapshot={},
            context_signals={},
            ml_signals=context.ml_signals,
            node_health={},
            kb_coverage=context.kb_coverage,
            recommendations=context.recommendations,
        )
        context.report_path = self._write_report(context.topic, context.report)
        return context.report

    def _write_report(self, topic: str, report: str) -> Path:
        safe_topic = re.sub(r"[^a-zA-Z0-9_-]+", "_", topic).strip("_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = self.reports_root / "ml_pipeline"
        reports_dir.mkdir(parents=True, exist_ok=True)
        filename = f"ml_pipeline_{safe_topic or 'report'}_{timestamp}.md"
        path = reports_dir / filename
        path.write_text(report)
        return path

    async def generate_report(self, topic: str) -> dict[str, Any]:
        await self.setup()

        context = MLPlanContext(topic=topic)
        pipeline = OrchestrationPipeline(
            [
                PipelineStep(name="signals", kind="collect", run=self._step_signals),
                PipelineStep(name="kb_coverage", kind="collect", run=self._step_kb),
                PipelineStep(name="plan", kind="plan", run=self._step_plan),
                PipelineStep(name="summarize", kind="summarize", run=self._step_summarize),
            ]
        )

        result = await pipeline.run(context)
        return {
            "topic": topic,
            "ml_signals": context.ml_signals,
            "kb_coverage": context.kb_coverage,
            "recommendations": context.recommendations,
            "report_path": str(context.report_path) if context.report_path else None,
            "report": context.report,
            "pipeline_status": [
                {"name": step.name, "status": step.status.value} for step in result.steps
            ],
        }

    async def run_task(self, task: str = "help") -> dict[str, Any]:
        if task == "help":
            return {"usage": ["report:TOPIC - Run ML pipeline planning"]}
        if task.startswith("report:"):
            topic = task[7:].strip()
            return await self.generate_report(topic)
        return {"error": f"Unknown task: {task}"}


async def main() -> None:
    """CLI entry point for deep context analysis."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m agents.analysis.deep_context_pipeline <topic>")
        return

    topic = " ".join(sys.argv[1:])
    pipeline = DeepContextPipeline()
    result = await pipeline.generate_report(topic)
    print(f"Report: {result.get('report_path')}")


if __name__ == "__main__":
    asyncio.run(main())
