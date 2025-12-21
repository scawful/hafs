"""Background Embedding Service.

Provides:
- Background embedding generation with checkpointing
- Progress tracking via status files
- Project indexing and management
- Cross-reference capabilities between knowledge bases

Usage:
    # Start background embedding
    service = EmbeddingService()
    await service.start()

    # Add a project to index
    await service.add_project("alttp", "/path/to/usdasm")

    # Check progress
    status = await service.get_status()

    # Cross-reference between projects
    results = await service.cross_reference("alttp", "oracle-of-secrets", "SprY")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import json

from hafs.core.embeddings import BatchEmbeddingManager
from hafs.core.projects import ProjectRegistry

logger = logging.getLogger(__name__)


class ProjectType(Enum):
    """Types of projects that can be indexed."""

    ASM_DISASSEMBLY = "asm_disassembly"  # 65816 ASM disassembly
    ROM_HACK = "rom_hack"                 # ROM hack source
    CODEBASE = "codebase"                 # Generic codebase
    DOCUMENTATION = "documentation"       # Markdown/text docs
    CUSTOM = "custom"                     # Custom extraction


@dataclass
class ProjectConfig:
    """Configuration for an indexable project."""

    name: str
    path: str
    project_type: ProjectType
    description: str = ""
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    enabled: bool = True
    priority: int = 50  # Lower = higher priority

    # Extraction settings
    include_patterns: list[str] = field(default_factory=lambda: ["*.asm", "*.md"])
    exclude_patterns: list[str] = field(default_factory=lambda: ["*.bak", ".git"])
    max_files: int = 10000

    # Cross-reference settings
    cross_ref_projects: list[str] = field(default_factory=list)
    knowledge_roots: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["project_type"] = self.project_type.value
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectConfig":
        data = data.copy()
        data["project_type"] = ProjectType(data.get("project_type", "custom"))
        return cls(**data)


@dataclass
class EmbeddingProgress:
    """Progress tracking for embedding generation."""

    project_name: str
    total_items: int
    processed_items: int
    failed_items: int
    start_time: str
    last_update: str
    status: str  # pending, running, paused, completed, failed
    current_item: str = ""
    rate_items_per_min: float = 0.0
    estimated_remaining_mins: float = 0.0
    checkpoint_file: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddingProgress":
        return cls(**data)


@dataclass
class CrossReference:
    """Cross-reference between two projects."""

    source_project: str
    target_project: str
    source_symbol: str
    target_symbol: str
    match_type: str  # name_exact, name_similar, address, semantic
    confidence: float = 1.0
    notes: str = ""


class EmbeddingService:
    """Background service for embedding generation and project indexing.

    Features:
    - Runs in background with checkpointing
    - Progress tracking via status files
    - Multiple project support
    - Cross-reference capabilities

    Example:
        service = EmbeddingService()
        await service.start()

        # Add projects
        await service.add_project(ProjectConfig(
            name="alttp",
            path="~/Code/usdasm",
            project_type=ProjectType.ASM_DISASSEMBLY,
        ))

        # Start indexing
        await service.index_project("alttp")

        # Check progress
        status = await service.get_status("alttp")
        print(f"Progress: {status.processed_items}/{status.total_items}")
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path.home() / ".context" / "embedding_service"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Storage files
        self.projects_file = self.data_dir / "projects.json"
        self.status_file = self.data_dir / "status.json"
        self.queue_file = self.data_dir / "queue.json"

        # In-memory state
        self._projects: dict[str, ProjectConfig] = {}
        self._progress: dict[str, EmbeddingProgress] = {}
        self._cross_refs: dict[str, list[CrossReference]] = {}

        # Background task
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._queue: list[str] = []

        # Orchestrator (lazy loaded)
        self._orchestrator = None

        self._load_state()
        try:
            self.sync_projects_from_registry()
        except Exception as exc:
            logger.debug("Project registry sync skipped: %s", exc)

    def _load_state(self):
        """Load persisted state from disk."""
        if self.projects_file.exists():
            try:
                data = json.loads(self.projects_file.read_text())
                for name, config in data.items():
                    self._projects[name] = ProjectConfig.from_dict(config)
            except Exception as e:
                logger.warning(f"Failed to load projects: {e}")

        if self.status_file.exists():
            try:
                data = json.loads(self.status_file.read_text())
                for name, progress in data.items():
                    self._progress[name] = EmbeddingProgress.from_dict(progress)
            except Exception as e:
                logger.warning(f"Failed to load status: {e}")

        if self.queue_file.exists():
            try:
                self._queue = json.loads(self.queue_file.read_text())
            except:
                pass

    def _save_state(self):
        """Save state to disk."""
        try:
            self.projects_file.write_text(json.dumps(
                {name: config.to_dict() for name, config in self._projects.items()},
                indent=2
            ))

            self.status_file.write_text(json.dumps(
                {name: progress.to_dict() for name, progress in self._progress.items()},
                indent=2
            ))

            self.queue_file.write_text(json.dumps(self._queue))
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    async def _ensure_orchestrator(self):
        """Lazy load orchestrator."""
        if self._orchestrator is None:
            from hafs.core.orchestrator_v2 import UnifiedOrchestrator
            self._orchestrator = UnifiedOrchestrator()
            await self._orchestrator.initialize()

    async def start(self):
        """Start the background service."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._background_loop())
        logger.info("Embedding service started")

    async def stop(self):
        """Stop the background service."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Embedding service stopped")

    async def _background_loop(self):
        """Background processing loop."""
        while self._running:
            try:
                # Process queue
                if self._queue:
                    project_name = self._queue[0]
                    await self._process_project(project_name)
                    self._queue.pop(0)
                    self._save_state()
                else:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background loop error: {e}")
                await asyncio.sleep(5)

    async def add_project(self, config: ProjectConfig) -> bool:
        """Add a project configuration.

        Args:
            config: Project configuration.

        Returns:
            True if added successfully.
        """
        self._projects[config.name] = config
        self._save_state()
        logger.info(f"Added project: {config.name}")
        return True

    async def remove_project(self, name: str) -> bool:
        """Remove a project.

        Args:
            name: Project name.

        Returns:
            True if removed.
        """
        if name in self._projects:
            del self._projects[name]
            self._save_state()
            return True
        return False

    def get_projects(self) -> list[ProjectConfig]:
        """Get all registered projects."""
        return list(self._projects.values())

    def sync_projects_from_registry(self, force: bool = False) -> dict[str, ProjectConfig]:
        """Sync projects from the main ProjectRegistry."""
        registry = ProjectRegistry.load()
        added: dict[str, ProjectConfig] = {}

        for project in registry.list():
            config = self._project_config_from_registry(project)
            if not force and config.name in self._projects:
                continue
            self._projects[config.name] = config
            added[config.name] = config

        if added:
            self._save_state()
        return added

    def _project_config_from_registry(self, project: Any) -> ProjectConfig:
        project_type = self._infer_project_type(project)
        include_patterns = self._default_include_patterns(project_type)
        exclude_patterns = self._default_exclude_patterns()
        knowledge_roots = [str(p) for p in getattr(project, "knowledge_roots", [])]

        return ProjectConfig(
            name=project.name,
            path=str(project.path),
            project_type=project_type,
            description=project.description or "",
            embedding_provider=getattr(project, "embedding_provider", None),
            embedding_model=getattr(project, "embedding_model", None),
            enabled=project.enabled,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            knowledge_roots=knowledge_roots,
        )

    @staticmethod
    def _infer_project_type(project: Any) -> ProjectType:
        kind = (getattr(project, "kind", "") or "").lower()
        tags = {t.lower() for t in getattr(project, "tags", [])}

        if kind in {"asm", "disassembly"} or "disassembly" in tags:
            return ProjectType.ASM_DISASSEMBLY
        if kind in {"snes", "romhack"} or "romhack" in tags or "snes" in tags:
            return ProjectType.ROM_HACK
        if kind in {"docs", "documentation"} or "docs" in tags:
            return ProjectType.DOCUMENTATION
        return ProjectType.CODEBASE

    @staticmethod
    def _default_include_patterns(project_type: ProjectType) -> list[str]:
        if project_type in {ProjectType.ASM_DISASSEMBLY, ProjectType.ROM_HACK}:
            return ["*.asm", "*.inc", "*.s", "*.tbl", "*.md", "*.txt"]
        if project_type == ProjectType.DOCUMENTATION:
            return ["*.md", "*.txt", "*.rst", "*.adoc"]
        return [
            "*.py",
            "*.ts",
            "*.tsx",
            "*.js",
            "*.jsx",
            "*.go",
            "*.rs",
            "*.c",
            "*.cpp",
            "*.h",
            "*.hpp",
            "*.java",
            "*.kt",
            "*.swift",
            "*.rb",
            "*.md",
            "*.toml",
            "*.yaml",
            "*.yml",
            "*.json",
        ]

    @staticmethod
    def _default_exclude_patterns() -> list[str]:
        return ["*.bak", "*.lock", ".git", ".context", "**/node_modules/**", "**/.venv/**"]

    def _resolve_embedding_settings(
        self,
        config: ProjectConfig,
        embedding_provider: Optional[str],
        embedding_model: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        provider = embedding_provider or config.embedding_provider
        model = embedding_model or config.embedding_model
        return provider, model

    def _resolve_embedding_root(self, config: ProjectConfig) -> Path:
        if config.project_type == ProjectType.ASM_DISASSEMBLY:
            return Path.home() / ".context" / "knowledge" / "alttp"
        if config.project_type == ProjectType.ROM_HACK:
            return Path.home() / ".context" / "knowledge" / "oracle-of-secrets"
        return self.data_dir / "projects" / config.name

    def get_embedding_dir(
        self,
        project_name: str,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> Optional[Path]:
        """Resolve the embeddings directory for a project."""
        config = self.resolve_project(project_name)
        if not config:
            return None
        provider, model = self._resolve_embedding_settings(
            config,
            embedding_provider,
            embedding_model,
        )
        storage_id = BatchEmbeddingManager.resolve_storage_id(provider, model)
        base_dir = self._resolve_embedding_root(config)
        return BatchEmbeddingManager.resolve_embeddings_dir(base_dir, storage_id)

    def get_embedding_root(self, project_name: str) -> Optional[Path]:
        """Resolve the embedding root for a project."""
        config = self.resolve_project(project_name)
        if not config:
            return None
        return self._resolve_embedding_root(config)

    async def queue_indexing(self, project_name: str) -> bool:
        """Queue a project for indexing.

        Args:
            project_name: Name of project to index.

        Returns:
            True if queued.
        """
        if project_name not in self._projects:
            return False

        if project_name not in self._queue:
            self._queue.append(project_name)

            # Initialize progress
            self._progress[project_name] = EmbeddingProgress(
                project_name=project_name,
                total_items=0,
                processed_items=0,
                failed_items=0,
                start_time=datetime.now().isoformat(),
                last_update=datetime.now().isoformat(),
                status="pending",
            )

            self._save_state()

        return True

    def resolve_project(self, name: str) -> Optional[ProjectConfig]:
        """Resolve a project by name."""
        for project_name, config in self._projects.items():
            if project_name.lower() == name.lower():
                return config
        return None

    async def run_indexing(
        self,
        project_names: list[str],
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        """Run indexing for specific projects immediately."""
        for project_name in project_names:
            await self._process_project(
                project_name,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
            )

    async def get_status(self, project_name: Optional[str] = None) -> dict[str, Any]:
        """Get indexing status.

        Args:
            project_name: Optional specific project.

        Returns:
            Status information.
        """
        if project_name:
            if project_name in self._progress:
                return self._progress[project_name].to_dict()
            return {"error": "Project not found"}

        return {
            "service_running": self._running,
            "queue": self._queue,
            "projects": {
                name: progress.to_dict()
                for name, progress in self._progress.items()
            },
        }

    async def get_progress_async(self, project_name: str) -> Optional[EmbeddingProgress]:
        """Get progress for async viewing.

        This reads from the status file which is updated during processing.
        """
        # Re-load from disk for latest
        if self.status_file.exists():
            try:
                data = json.loads(self.status_file.read_text())
                if project_name in data:
                    return EmbeddingProgress.from_dict(data[project_name])
            except:
                pass

        return self._progress.get(project_name)

    async def _process_project(
        self,
        project_name: str,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """Process a project for embedding generation."""
        if project_name not in self._projects:
            return

        config = self._projects[project_name]
        progress = self._progress.get(project_name)

        if not progress:
            return

        progress.status = "running"
        progress.start_time = datetime.now().isoformat()
        self._save_state()

        try:
            await self._ensure_orchestrator()

            # Extract items based on project type
            if config.project_type == ProjectType.ASM_DISASSEMBLY:
                await self._process_asm_project(
                    config,
                    progress,
                    embedding_provider=embedding_provider,
                    embedding_model=embedding_model,
                )
            elif config.project_type == ProjectType.ROM_HACK:
                await self._process_rom_hack_project(
                    config,
                    progress,
                    embedding_provider=embedding_provider,
                    embedding_model=embedding_model,
                )
            else:
                await self._process_generic_project(
                    config,
                    progress,
                    embedding_provider=embedding_provider,
                    embedding_model=embedding_model,
                )

            progress.status = "completed"
        except Exception as e:
            logger.error(f"Failed to process {project_name}: {e}")
            progress.status = "failed"
        finally:
            progress.last_update = datetime.now().isoformat()
            self._save_state()

    async def _process_asm_project(
        self,
        config: ProjectConfig,
        progress: EmbeddingProgress,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """Process an ASM disassembly project."""
        from hafs.agents.alttp_knowledge import ALTTPKnowledgeBase

        source_path = Path(config.path).expanduser()
        kb = ALTTPKnowledgeBase(source_path)
        await kb.setup()
        if not self._orchestrator:
            return

        provider, model = self._resolve_embedding_settings(
            config,
            embedding_provider,
            embedding_model,
        )
        manager = BatchEmbeddingManager(
            kb_dir=kb.kb_dir,
            orchestrator=self._orchestrator,
            embedding_provider=provider,
            embedding_model=model,
        )
        progress.checkpoint_file = str(manager.checkpoint_file)

        items = []
        for symbol in kb._symbols.values():
            text = f"{symbol.name}: {symbol.description}" if symbol.description else symbol.name
            items.append((symbol.id, text))

        progress.total_items = len(items)
        start_time = time.time()

        def _update_progress(processed: int, total: int) -> None:
            elapsed = time.time() - start_time
            rate = processed / (elapsed / 60) if elapsed > 0 else 0
            remaining = (total - processed) / rate if rate > 0 else 0

            progress.processed_items = processed
            progress.total_items = total
            progress.rate_items_per_min = round(rate, 1)
            progress.estimated_remaining_mins = round(remaining, 1)
            progress.last_update = datetime.now().isoformat()
            self._save_state()

        stats = await manager.generate_embeddings(
            items,
            kb_name="alttp_symbols",
            progress_callback=_update_progress,
        )

        progress.failed_items = stats.get("errors", 0)
        progress.last_update = datetime.now().isoformat()
        self._save_state()

    async def _process_rom_hack_project(
        self,
        config: ProjectConfig,
        progress: EmbeddingProgress,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """Process a ROM hack project."""
        from hafs.agents.alttp_unified_kb import OracleOfSecretsKB

        provider, model = self._resolve_embedding_settings(
            config,
            embedding_provider,
            embedding_model,
        )
        kb = OracleOfSecretsKB(
            embedding_provider=provider,
            embedding_model=model,
        )
        kb.SOURCE_PATH = Path(config.path).expanduser()
        await kb.setup()

        # Build KB
        await kb.build(generate_embeddings=True)

        progress.processed_items = len(kb._symbols) + len(kb._routines)
        progress.total_items = progress.processed_items

    def _resolve_roots(self, config: ProjectConfig) -> list[Path]:
        """Resolve project roots for indexing."""
        source_path = Path(config.path).expanduser()
        if config.knowledge_roots:
            roots: list[Path] = []
            for root in config.knowledge_roots:
                root_path = Path(root).expanduser()
                if not root_path.is_absolute():
                    root_path = source_path / root_path
                if root_path.exists():
                    roots.append(root_path)
            if roots:
                return roots
        return [source_path]

    def _collect_files(self, config: ProjectConfig) -> list[Path]:
        """Collect files for indexing."""
        roots = self._resolve_roots(config)
        files: list[Path] = []

        for root in roots:
            for pattern in config.include_patterns:
                files.extend(root.rglob(pattern))

        exclude_set: set[Path] = set()
        for root in roots:
            for pattern in config.exclude_patterns:
                exclude_set.update(root.rglob(pattern))

        filtered = [
            f for f in files
            if f.is_file() and f not in exclude_set
        ]

        # Drop common heavy directories even if patterns miss them
        excluded_dirs = {
            ".git",
            ".context",
            "node_modules",
            ".venv",
            "__pycache__",
            "dist",
            "build",
        }
        filtered = [
            f for f in filtered
            if not any(part in excluded_dirs for part in f.parts)
        ]

        return sorted(set(filtered))[:config.max_files]

    async def _process_generic_project(
        self,
        config: ProjectConfig,
        progress: EmbeddingProgress,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """Process a generic project."""
        source_path = Path(config.path).expanduser()
        files = self._collect_files(config)
        progress.total_items = len(files)

        kb_dir = self.data_dir / "projects" / config.name
        kb_dir.mkdir(parents=True, exist_ok=True)

        provider, model = self._resolve_embedding_settings(
            config,
            embedding_provider,
            embedding_model,
        )
        manager = BatchEmbeddingManager(
            kb_dir=kb_dir,
            orchestrator=self._orchestrator,
            embedding_provider=provider,
            embedding_model=model,
        )
        progress.checkpoint_file = str(manager.checkpoint_file)

        items: list[tuple[str, str]] = []
        for file_path in files:
            try:
                try:
                    relative = file_path.relative_to(source_path)
                except ValueError:
                    relative = Path(file_path.name)
                file_id = f"file:{relative.as_posix()}"

                if manager.has_embedding(file_id):
                    items.append((file_id, ""))
                    continue

                content = file_path.read_text(errors="ignore")[:2000]
                text = f"{relative.as_posix()}: {content[:500]}"
                items.append((file_id, text))
            except Exception:
                progress.failed_items += 1

        start_time = time.time()

        def _update_progress(processed: int, total: int) -> None:
            elapsed = time.time() - start_time
            rate = processed / (elapsed / 60) if elapsed > 0 else 0
            remaining = (total - processed) / rate if rate > 0 else 0

            progress.processed_items = processed
            progress.total_items = total
            progress.rate_items_per_min = round(rate, 1)
            progress.estimated_remaining_mins = round(remaining, 1)
            progress.last_update = datetime.now().isoformat()
            self._save_state()

        stats = await manager.generate_embeddings(
            items,
            kb_name=f"{config.name}_files",
            progress_callback=_update_progress,
        )

        progress.failed_items = stats.get("errors", 0)
        progress.last_update = datetime.now().isoformat()
        self._save_state()

    async def cross_reference(
        self,
        source_project: str,
        target_project: str,
        query: Optional[str] = None,
    ) -> list[CrossReference]:
        """Find cross-references between projects.

        Args:
            source_project: Source project name.
            target_project: Target project name.
            query: Optional query to filter results.

        Returns:
            List of cross-references.
        """
        if source_project not in self._projects or target_project not in self._projects:
            return []

        refs = []

        # Load both KBs
        source_kb = await self._load_project_kb(source_project)
        target_kb = await self._load_project_kb(target_project)

        if not source_kb or not target_kb:
            return []

        # Find matches
        source_symbols = source_kb.get("symbols", {})
        target_symbols = target_kb.get("symbols", {})

        for source_name, source_data in source_symbols.items():
            # Exact name match
            if source_name in target_symbols:
                refs.append(CrossReference(
                    source_project=source_project,
                    target_project=target_project,
                    source_symbol=source_name,
                    target_symbol=source_name,
                    match_type="name_exact",
                    confidence=1.0,
                ))

            # Address match
            source_addr = source_data.get("address", "")
            if source_addr:
                for target_name, target_data in target_symbols.items():
                    if target_data.get("address") == source_addr and source_name != target_name:
                        refs.append(CrossReference(
                            source_project=source_project,
                            target_project=target_project,
                            source_symbol=source_name,
                            target_symbol=target_name,
                            match_type="address",
                            confidence=0.9,
                        ))

        # Filter by query if provided
        if query:
            query_lower = query.lower()
            refs = [r for r in refs if
                    query_lower in r.source_symbol.lower() or
                    query_lower in r.target_symbol.lower()]

        return refs

    async def _load_project_kb(self, project_name: str) -> Optional[dict]:
        """Load a project's knowledge base."""
        if project_name not in self._projects:
            return None

        config = self._projects[project_name]

        if config.project_type == ProjectType.ASM_DISASSEMBLY:
            kb_dir = Path.home() / ".context" / "knowledge" / "alttp"
        elif config.project_type == ProjectType.ROM_HACK:
            kb_dir = Path.home() / ".context" / "knowledge" / "oracle-of-secrets"
        else:
            kb_dir = self.data_dir / "projects" / project_name

        result = {"symbols": {}, "routines": {}}

        symbols_file = kb_dir / "symbols.json"
        if symbols_file.exists():
            try:
                data = json.loads(symbols_file.read_text())
                if isinstance(data, list):
                    for item in data:
                        name = item.get("name", "")
                        if name:
                            result["symbols"][name] = item
                else:
                    result["symbols"] = data
            except:
                pass

        routines_file = kb_dir / "routines.json"
        if routines_file.exists():
            try:
                data = json.loads(routines_file.read_text())
                if isinstance(data, list):
                    for item in data:
                        name = item.get("name", "")
                        if name:
                            result["routines"][name] = item
                else:
                    result["routines"] = data
            except:
                pass

        return result

    async def semantic_cross_reference(
        self,
        source_project: str,
        target_project: str,
        threshold: float = 0.7,
        source_provider: Optional[str] = None,
        source_model: Optional[str] = None,
        target_provider: Optional[str] = None,
        target_model: Optional[str] = None,
    ) -> list[CrossReference]:
        """Find semantic cross-references using embeddings.

        Args:
            source_project: Source project name.
            target_project: Target project name.
            threshold: Minimum similarity threshold.

        Returns:
            List of semantic cross-references.
        """
        from hafs.core.similarity import cosine_similarity

        # Load embeddings for both projects
        source_embs = await self._load_embeddings(
            source_project,
            embedding_provider=source_provider,
            embedding_model=source_model,
        )
        target_embs = await self._load_embeddings(
            target_project,
            embedding_provider=target_provider,
            embedding_model=target_model,
        )

        refs = []

        for source_id, source_emb in source_embs.items():
            best_match = None
            best_score = threshold

            for target_id, target_emb in target_embs.items():
                score = cosine_similarity(source_emb, target_emb)
                if score > best_score:
                    best_score = score
                    best_match = target_id

            if best_match:
                refs.append(CrossReference(
                    source_project=source_project,
                    target_project=target_project,
                    source_symbol=source_id.split(":")[-1],
                    target_symbol=best_match.split(":")[-1],
                    match_type="semantic",
                    confidence=best_score,
                ))

        return sorted(refs, key=lambda r: r.confidence, reverse=True)

    async def _load_embeddings(
        self,
        project_name: str,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> dict[str, list[float]]:
        """Load embeddings for a project."""
        config = self.resolve_project(project_name)
        if not config:
            return {}

        provider, model = self._resolve_embedding_settings(
            config,
            embedding_provider,
            embedding_model,
        )
        storage_id = BatchEmbeddingManager.resolve_storage_id(provider, model)
        base_dir = self._resolve_embedding_root(config)
        emb_dir = BatchEmbeddingManager.resolve_embeddings_dir(base_dir, storage_id)

        embeddings = {}

        if emb_dir.exists():
            for emb_file in emb_dir.glob("*.json"):
                try:
                    data = json.loads(emb_file.read_text())
                    emb_id = data.get("id")
                    embedding = data.get("embedding")
                    if emb_id and embedding:
                        embeddings[emb_id] = embedding
                except:
                    pass

        return embeddings


# CLI for managing the service
async def cli_main():
    """CLI interface for the embedding service."""
    import argparse

    parser = argparse.ArgumentParser(description="Embedding Service CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Status command
    subparsers.add_parser("status", help="Get service status")

    # Add project
    add_parser = subparsers.add_parser("add", help="Add a project")
    add_parser.add_argument("name", help="Project name")
    add_parser.add_argument("path", help="Project path")
    add_parser.add_argument("--type", default="asm_disassembly",
                          choices=["asm_disassembly", "rom_hack", "codebase", "documentation"])

    # Queue indexing
    queue_parser = subparsers.add_parser("index", help="Queue project for indexing")
    queue_parser.add_argument("name", help="Project name")

    # Cross-reference
    xref_parser = subparsers.add_parser("xref", help="Find cross-references")
    xref_parser.add_argument("source", help="Source project")
    xref_parser.add_argument("target", help="Target project")
    xref_parser.add_argument("--query", help="Filter query")

    args = parser.parse_args()

    service = EmbeddingService()

    if args.command == "status":
        status = await service.get_status()
        print(json.dumps(status, indent=2))

    elif args.command == "add":
        config = ProjectConfig(
            name=args.name,
            path=args.path,
            project_type=ProjectType(args.type),
        )
        await service.add_project(config)
        print(f"Added project: {args.name}")

    elif args.command == "index":
        await service.queue_indexing(args.name)
        print(f"Queued {args.name} for indexing")

    elif args.command == "xref":
        refs = await service.cross_reference(args.source, args.target, args.query)
        for ref in refs:
            print(f"{ref.source_symbol} <-> {ref.target_symbol} ({ref.match_type}, {ref.confidence:.2f})")


if __name__ == "__main__":
    asyncio.run(cli_main())
