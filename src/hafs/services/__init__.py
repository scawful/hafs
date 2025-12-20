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

from hafs.services.autonomy_daemon import (
    AutonomyDaemon,
    get_status as get_autonomy_status,
    install_launchd as install_autonomy_launchd,
    uninstall_launchd as uninstall_autonomy_launchd,
)

from hafs.services.afs_sync import (
    AFSSyncService,
    SyncProfile,
    SyncRegistry,
    SyncTarget,
    SyncResult,
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
    "AFSSyncService",
    "SyncProfile",
    "SyncRegistry",
    "SyncTarget",
    "SyncResult",
]
