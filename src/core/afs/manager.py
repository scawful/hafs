"""AFS (Agentic File System) manager for context directories."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.schema import HafsConfig
from core.config.loader import CognitiveProtocolConfig, get_config
from core.protocol.io_manager import get_io_manager
from core.afs.mapping import resolve_directory_map, resolve_directory_name
from models.afs import ContextRoot, MountPoint, MountType, ProjectMetadata


class AFSManager:
    """Manages AFS (.context) directories.

    Provides operations for initializing, mounting, listing, and cleaning
    AFS context directories.
    """

    CONTEXT_ROOT = ".context"
    METADATA_FILE = "metadata.json"
    STATE_FILE = "state.md"
    DEFERRED_FILE = "deferred.md"
    STATE_JSON_FILE = "state.json"
    METACOGNITION_FILE = "metacognition.json"
    GOALS_FILE = "goals.json"
    FEARS_FILE = "fears.json"
    EMOTIONS_FILE = "emotions.json"
    EPISTEMIC_FILE = "epistemic.json"
    ANALYSIS_TRIGGERS_FILE = "analysis-triggers.json"

    DEFAULT_STATE_TEMPLATE = """# Agent State

## 1. Current Context
- **Last User Input:** [Copy of the latest user prompt]
- **Relevant History:** [Brief summary of relevant past interactions from `history`]
- **Applicable Rules:** [Key constraints or facts from `memory` or `knowledge`]

## 2. Theory of Mind
- **User's Goal:** [Inferred intent of the user]
- **User's Likely Knowledge:** [What can I assume the user knows or sees?]
- **Predicted User Reaction:** [How might the user react to my proposed action?]

## 3. Deliberation & Intent
- **Options Considered:**
  1. [Option A: Pros/Cons]
  2. [Option B: Pros/Cons]
- **Chosen Action:** [Description of the action to be taken]
- **Justification:** [Why this action was chosen over others]
- **Intended Outcome:** [What this action is expected to achieve]

## 4. Action Outcome
- **Result:** [To be filled in after the action is executed. Was it successful? What was the output?]
- **Next Steps:** [Immediate follow-up actions, if any]

## 5. Emotional State & Risk Assessment
- **Identified Concerns:** [List of potential negative outcomes, e.g., "This change might break API compatibility."]
- **Confidence Score (0-1):** [e.g., 0.75]
- **Mitigation Strategy:** [How to address the concerns, e.g., "I will add a new test case to verify compatibility."]

## 6. Metacognitive Assessment
- **Current Strategy:** [incremental | divide_and_conquer | depth_first | breadth_first | research_first | prototype]
- **Strategy Effectiveness (0-1):** [How well the current strategy is working]
- **Progress Status:** [making_progress | spinning | blocked]
- **Cognitive Load:** [Percentage of working memory capacity in use]
- **Items in Focus:** [Number of items currently being tracked]
- **Spinning Warning:** [Yes/No - Are we repeating similar actions without progress?]
- **Help Needed:** [Yes/No - Should we ask the user for clarification?]
- **Flow State:** [Yes/No - Are conditions optimal for autonomous action?]
"""

    def __init__(
        self,
        config: HafsConfig,
        cognitive_config: CognitiveProtocolConfig | None = None,
    ):
        """Initialize manager with configuration.

        Args:
            config: HAFS configuration object.
            cognitive_config: Cognitive protocol configuration. If None, uses default config.
        """
        self.config = config
        self._cognitive_config = cognitive_config or get_config()
        self._directories = {d.name: d for d in config.afs_directories}
        self._directory_map = resolve_directory_map(afs_directories=config.afs_directories)

    def _directory_for_mount_type(self, mount_type: MountType) -> str:
        return resolve_directory_name(mount_type, afs_directories=self.config.afs_directories)

    def ensure(self, path: Path = Path(".")) -> ContextRoot:
        """Ensure AFS exists and includes cognitive protocol scaffolding.

        This is a non-destructive operation: it creates missing directories/files
        but will not overwrite existing user content.

        Args:
            path: Directory to ensure AFS in.

        Returns:
            ContextRoot for the ensured AFS.
        """
        context_path = path.resolve() / self.CONTEXT_ROOT
        if not context_path.exists():
            root = self.init(path, force=False)
            # init() already scaffolds
            return root

        # Ensure subdirectories and .keep files exist (config-driven)
        context_path.mkdir(exist_ok=True)
        for dir_config in self.config.afs_directories:
            subdir = context_path / dir_config.name
            subdir.mkdir(exist_ok=True)
            keep = subdir / ".keep"
            if not keep.exists():
                keep.touch()

        # Ensure metadata exists
        metadata_path = context_path / self.METADATA_FILE
        directory_map = {role.value: name for role, name in self._directory_map.items()}
        if not metadata_path.exists():
            metadata = ProjectMetadata(
                created_at=datetime.now(),
                agents=[],
                description=f"AFS for {path.resolve().name}",
                directories=directory_map,
            )
            metadata_path.write_text(
                json.dumps(metadata.model_dump(mode="json"), indent=2, default=str),
                encoding="utf-8",
            )
        else:
            try:
                metadata = ProjectMetadata(**json.loads(metadata_path.read_text()))
            except Exception:
                metadata = None
            if metadata and not metadata.directories and directory_map:
                metadata = metadata.model_copy(update={"directories": directory_map})
                metadata_path.write_text(
                    json.dumps(metadata.model_dump(mode="json"), indent=2, default=str),
                    encoding="utf-8",
                )

        self._ensure_protocol_scaffold(context_path)
        return self.list_afs_structure(context_path=context_path)

    def init(self, path: Path = Path("."), force: bool = False) -> ContextRoot:
        """Initialize AFS in the given directory.

        Creates the .context directory structure with all subdirectories,
        .keep files for Git compatibility, and metadata.json.

        Args:
            path: Directory to initialize AFS in.
            force: If True, overwrite existing AFS.

        Returns:
            ContextRoot object representing the initialized AFS.

        Raises:
            FileExistsError: If AFS already exists and force is False.
        """
        context_path = path.resolve() / self.CONTEXT_ROOT

        if context_path.exists() and not force:
            raise FileExistsError(f"AFS already exists at {context_path}")

        # Create root
        context_path.mkdir(exist_ok=True)

        # Create subdirectories
        for dir_config in self.config.afs_directories:
            subdir = context_path / dir_config.name
            subdir.mkdir(exist_ok=True)
            (subdir / ".keep").touch()

        # Create metadata
        metadata = ProjectMetadata(
            created_at=datetime.now(),
            agents=[],
            description=f"AFS for {path.resolve().name}",
            directories={role.value: name for role, name in self._directory_map.items()},
        )

        metadata_path = context_path / self.METADATA_FILE
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                metadata.model_dump(mode="json"),
                f,
                indent=2,
                default=str,
            )

        self._ensure_protocol_scaffold(context_path)

        return ContextRoot(
            path=context_path,
            project_name=path.resolve().name,
            metadata=metadata,
        )

    def _ensure_protocol_scaffold(self, context_path: Path) -> None:
        """Create missing cognitive protocol files in-place.

        This is intentionally non-destructive and will not overwrite existing
        files. It assumes the AFS directory structure already exists.
        """
        scratchpad_dir = context_path / self._directory_for_mount_type(MountType.SCRATCHPAD)
        memory_dir = context_path / self._directory_for_mount_type(MountType.MEMORY)

        try:
            scratchpad_dir.mkdir(parents=True, exist_ok=True)
            memory_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return

        # scratchpad/state.md (rendered view; canonical is scratchpad/state.json)
        state_file = scratchpad_dir / self.STATE_FILE
        if not state_file.exists():
            state_file.write_text(self.DEFAULT_STATE_TEMPLATE, encoding="utf-8")

        # scratchpad/deferred.md
        deferred_file = scratchpad_dir / self.DEFERRED_FILE
        if not deferred_file.exists():
            deferred_file.write_text("# Deferred\n\n", encoding="utf-8")

        # scratchpad/state.json (canonical v0.3)
        state_json = scratchpad_dir / self.STATE_JSON_FILE
        if not state_json.exists():
            io_manager = get_io_manager()
            io_manager.write_json(
                state_json,
                {
                    "schema_version": "0.3",
                    "producer": {"name": "hafs", "version": "unknown"},
                    "last_updated": datetime.now().isoformat(),
                    "entries": [],
                },
                immediate=True,
            )

        # scratchpad/metacognition.json
        meta_file = scratchpad_dir / self.METACOGNITION_FILE
        if not meta_file.exists():
            io_manager = get_io_manager()
            try:
                from models.metacognition import MetacognitiveState

                meta_state = MetacognitiveState()
                io_manager.write_json(
                    meta_file,
                    meta_state.model_dump(mode="json"),
                    immediate=True,
                )
            except Exception:
                # Fall back to a minimal schema the UI can read
                io_manager.write_json(
                    meta_file,
                    {
                        "current_strategy": "incremental",
                        "strategy_effectiveness": 0.5,
                        "progress_status": "making_progress",
                        "spin_detection": {
                            "recent_actions": [],
                            "similar_action_count": 0,
                            "spinning_threshold": 4,
                        },
                        "cognitive_load": {"current": 0.0, "items_in_focus": 0},
                        "help_seeking": {
                            "current_uncertainty": 0.0,
                            "consecutive_failures": 0,
                            "should_ask_user": False,
                        },
                        "flow_state": False,
                        "self_corrections": [],
                        "schema_version": "0.3",
                        "producer": {"name": "hafs", "version": "unknown"},
                        "last_updated": datetime.now().isoformat(),
                    },
                    immediate=True,
                )

        # scratchpad/goals.json
        goals_file = scratchpad_dir / self.GOALS_FILE
        if not goals_file.exists():
            io_manager = get_io_manager()
            try:
                from models.goals import GoalHierarchy

                hierarchy = GoalHierarchy()
                io_manager.write_json(
                    goals_file,
                    hierarchy.model_dump(mode="json"),
                    immediate=True,
                )
            except Exception:
                io_manager.write_json(
                    goals_file,
                    {
                        "primary_goal": None,
                        "subgoals": [],
                        "instrumental_goals": [],
                        "goal_stack": [],
                        "conflicts": [],
                        "schema_version": "0.3",
                        "producer": {"name": "hafs", "version": "unknown"},
                        "last_updated": datetime.now().isoformat(),
                    },
                    immediate=True,
                )

        # memory/fears.json
        fears_file = memory_dir / self.FEARS_FILE
        if not fears_file.exists():
            io_manager = get_io_manager()
            io_manager.write_json(
                fears_file,
                {
                    "version": 1,
                    "fears": [
                        {
                            "id": "fear-edit-without-reading",
                            "trigger": {
                                "keywords": ["edit", "patch", "change"],
                                "pattern": "making changes without reviewing existing context",
                            },
                            "concern": "Accidental breakage from acting without context",
                            "mitigation": "Read relevant files first, then make minimal changes with tests.",
                            "learned_from": [],
                        }
                    ],
                    "schema_version": "0.3",
                    "producer": {"name": "hafs", "version": "unknown"},
                    "last_updated": datetime.now().isoformat(),
                },
                immediate=True,
            )

        # scratchpad/emotions.json (aligned with oracle-code schema)
        emotions_file = scratchpad_dir / self.EMOTIONS_FILE
        if not emotions_file.exists():
            now = datetime.now().isoformat()
            io_manager = get_io_manager()
            io_manager.write_json(
                emotions_file,
                {
                    "schema_version": "0.3",
                    "producer": {"name": "hafs", "version": "unknown"},
                    "last_updated": now,
                    "session": {
                        "mood": "neutral",
                        "anxietyLevel": 0,
                        "confidenceLevel": 50,
                        "recentEmotions": [],
                        "moodHistory": [],
                        "mode": "general",
                        "sessionStart": now,
                    },
                    "fears": {},
                    "curiosities": {},
                    "satisfactions": {},
                    "frustrations": {},
                    "excitements": {},
                    "determinations": {},
                    "cautions": {},
                    "reliefs": {},
                    "settings": {
                        "decay": {
                            "fear": self._cognitive_config.emotions.decay_rates.fear,
                            "curiosity": self._cognitive_config.emotions.decay_rates.curiosity,
                            "satisfaction": self._cognitive_config.emotions.decay_rates.satisfaction,
                            "frustration": self._cognitive_config.emotions.decay_rates.frustration,
                            "excitement": self._cognitive_config.emotions.decay_rates.excitement,
                            "determination": self._cognitive_config.emotions.decay_rates.determination,
                            "caution": self._cognitive_config.emotions.decay_rates.caution,
                            "relief": self._cognitive_config.emotions.decay_rates.relief,
                        }
                    },
                },
                immediate=True,
            )

        # scratchpad/epistemic.json
        epistemic_file = scratchpad_dir / self.EPISTEMIC_FILE
        if not epistemic_file.exists():
            now = datetime.now().isoformat()
            io_manager = get_io_manager()
            io_manager.write_json(
                epistemic_file,
                {
                    "schema_version": "0.3",
                    "producer": {"name": "hafs", "version": "unknown"},
                    "last_updated": now,
                    "last_decay_check": now,
                    "golden_facts": {},
                    "working_facts": {},
                    "assumptions": {},
                    "unknowns": [],
                    "contradictions": [],
                    "settings": {
                        "auto_record_from_tools": True,
                        "auto_detect_contradictions": True,
                        "min_confidence_for_auto_record": self._cognitive_config.epistemic.auto_record_confidence,
                        "decay_rate_per_hour": self._cognitive_config.epistemic.decay_rate_per_hour,
                        "prune_threshold": self._cognitive_config.epistemic.prune_threshold,
                        "max_golden_facts": self._cognitive_config.epistemic.max_golden_facts,
                        "max_working_facts": self._cognitive_config.epistemic.max_working_facts,
                    },
                },
                immediate=True,
            )

        # scratchpad/analysis-triggers.json
        triggers_file = scratchpad_dir / self.ANALYSIS_TRIGGERS_FILE
        if not triggers_file.exists():
            io_manager = get_io_manager()
            io_manager.write_json(
                triggers_file,
                {
                    "schema_version": "0.3",
                    "producer": {"name": "hafs", "version": "unknown"},
                    "last_updated": datetime.now().isoformat(),
                    "pending": [],
                },
                immediate=True,
            )

    def mount(
        self,
        source: Path,
        mount_type: MountType,
        alias: Optional[str] = None,
        context_path: Optional[Path] = None,
    ) -> MountPoint:
        """Mount a resource into the AFS.

        Creates a symlink from the AFS directory to the source.

        Args:
            source: Path to the file or directory to mount.
            mount_type: Type of mount (memory, knowledge, tools, etc.).
            alias: Optional custom name for the mount point.
            context_path: Optional custom .context path.

        Returns:
            MountPoint object representing the mount.

        Raises:
            FileNotFoundError: If source doesn't exist.
            FileExistsError: If mount point already exists.
        """
        if context_path is None:
            context_path = Path(".") / self.CONTEXT_ROOT

        source = source.resolve()
        if not source.exists():
            raise FileNotFoundError(f"Source {source} does not exist")

        alias = alias or source.name
        dest = context_path / self._directory_for_mount_type(mount_type) / alias

        if dest.exists():
            raise FileExistsError(
                f"Mount point '{alias}' already exists in {mount_type.value}"
            )

        # Create symlink
        dest.symlink_to(source)

        return MountPoint(
            name=alias,
            source=source,
            mount_type=mount_type,
            is_symlink=True,
        )

    def unmount(
        self,
        alias: str,
        mount_type: MountType,
        context_path: Optional[Path] = None,
    ) -> bool:
        """Remove a mount point.

        Args:
            alias: Name of the mount point to remove.
            mount_type: Type of mount.
            context_path: Optional custom .context path.

        Returns:
            True if mount was removed, False if it didn't exist.
        """
        if context_path is None:
            context_path = Path(".") / self.CONTEXT_ROOT

        mount_path = context_path / self._directory_for_mount_type(mount_type) / alias
        if mount_path.exists() or mount_path.is_symlink():
            mount_path.unlink()
            return True
        return False

    def list_afs_structure(self, context_path: Optional[Path] = None) -> ContextRoot:
        """List current AFS structure.

        Args:
            context_path: Optional custom .context path.

        Returns:
            ContextRoot object with all mounts.

        Raises:
            FileNotFoundError: If no AFS is initialized.
        """
        if context_path is None:
            context_path = Path(".") / self.CONTEXT_ROOT

        context_path = context_path.resolve()
        if not context_path.exists():
            raise FileNotFoundError("No AFS initialized")

        # Load metadata
        metadata_path = context_path / self.METADATA_FILE
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                metadata = ProjectMetadata(**json.load(f))
        else:
            metadata = ProjectMetadata()

        # Scan mounts
        mounts: dict[MountType, list[MountPoint]] = {}
        directory_map = resolve_directory_map(
            afs_directories=self.config.afs_directories,
            metadata=metadata,
        )

        for mt in MountType:
            subdir = context_path / directory_map.get(mt, mt.value)
            if not subdir.exists():
                continue

            mount_list: list[MountPoint] = []
            for item in subdir.iterdir():
                if item.name == ".keep":
                    continue

                source = item.resolve() if item.is_symlink() else item
                mount_list.append(
                    MountPoint(
                        name=item.name,
                        source=source,
                        mount_type=mt,
                        is_symlink=item.is_symlink(),
                    )
                )

            mounts[mt] = mount_list

        return ContextRoot(
            path=context_path,
            project_name=context_path.parent.name,
            metadata=metadata,
            mounts=mounts,
        )

    def clean(self, context_path: Optional[Path] = None) -> None:
        """Remove the AFS directory.

        This only removes symlinks, not the actual files they point to.

        Args:
            context_path: Optional custom .context path.
        """
        if context_path is None:
            context_path = Path(".") / self.CONTEXT_ROOT

        if context_path.exists():
            shutil.rmtree(context_path)

    def update_metadata(
        self,
        context_path: Optional[Path] = None,
        description: Optional[str] = None,
        agents: Optional[list[str]] = None,
    ) -> ProjectMetadata:
        """Update AFS metadata.

        Args:
            context_path: Optional custom .context path.
            description: New description (if provided).
            agents: New agents list (if provided).

        Returns:
            Updated ProjectMetadata object.

        Raises:
            FileNotFoundError: If no AFS is initialized.
        """
        if context_path is None:
            context_path = Path(".") / self.CONTEXT_ROOT

        metadata_path = context_path / self.METADATA_FILE
        if not metadata_path.exists():
            raise FileNotFoundError("No AFS initialized")

        with open(metadata_path, encoding="utf-8") as f:
            metadata = ProjectMetadata(**json.load(f))

        if description is not None:
            metadata = metadata.model_copy(update={"description": description})
        if agents is not None:
            metadata = metadata.model_copy(update={"agents": agents})

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata.model_dump(mode="json"), f, indent=2, default=str)

        return metadata
