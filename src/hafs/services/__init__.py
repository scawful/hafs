"""HAFS Background Services.

Available services:
- EmbeddingService: Project indexing and cross-referencing
- EmbeddingDaemon: Continuous embedding generation daemon
- AFSSyncService: Cross-node AFS sync
"""

from hafs.services.embedding_service import (
    EmbeddingService,
    ProjectConfig,
    ProjectType,
    EmbeddingProgress,
    CrossReference,
)

from hafs.services.embedding_daemon import (
    EmbeddingDaemon,
    get_status as get_daemon_status,
    install_launchd,
    uninstall_launchd,
)

try:
    from hafs.services.autonomy_daemon import (
        AutonomyDaemon,
        get_status as get_autonomy_status,
        install_launchd as install_autonomy_launchd,
        uninstall_launchd as uninstall_autonomy_launchd,
    )
except Exception as exc:
    AutonomyDaemon = None

    def get_autonomy_status(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError(f"Autonomy daemon unavailable: {exc}")

    def install_autonomy_launchd(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError(f"Autonomy daemon unavailable: {exc}")

    def uninstall_autonomy_launchd(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError(f"Autonomy daemon unavailable: {exc}")

try:
    from hafs.services.observability_daemon import (
        ObservabilityDaemon,
        get_status as get_observability_status,
    )
except Exception as exc:
    ObservabilityDaemon = None

    def get_observability_status(*args, **kwargs):  # type: ignore[override]
        raise RuntimeError(f"Observability daemon unavailable: {exc}")

from hafs.services.afs_sync import (
    AFSSyncService,
    SyncProfile,
    SyncRegistry,
    SyncTarget,
    SyncResult,
)

from hafs.services.synergy_service import (
    SynergyService,
    SynergyStatus,
    SynergySummary,
)

__all__ = [
    "EmbeddingService",
    "ProjectConfig",
    "ProjectType",
    "EmbeddingProgress",
    "CrossReference",
    "EmbeddingDaemon",
    "get_daemon_status",
    "install_launchd",
    "uninstall_launchd",
    "AutonomyDaemon",
    "get_autonomy_status",
    "install_autonomy_launchd",
    "uninstall_autonomy_launchd",
    "ObservabilityDaemon",
    "get_observability_status",
    "AFSSyncService",
    "SyncProfile",
    "SyncRegistry",
    "SyncTarget",
    "SyncResult",
    "SynergyService",
    "SynergyStatus",
    "SynergySummary",
]
