import importlib
import warnings

_DEPRECATION_MESSAGE = "services is deprecated. Import from 'services' instead."

warnings.warn(
    _DEPRECATION_MESSAGE,
    DeprecationWarning,
    stacklevel=2,
)

_EXPORTS = [
    ("EmbeddingService", "services.embedding_service", "EmbeddingService"),
    ("ProjectConfig", "services.embedding_service", "ProjectConfig"),
    ("ProjectType", "services.embedding_service", "ProjectType"),
    ("EmbeddingProgress", "services.embedding_service", "EmbeddingProgress"),
    ("CrossReference", "services.embedding_service", "CrossReference"),
    ("EmbeddingDaemon", "services.daemons.embedding_daemon", "EmbeddingDaemon"),
    ("get_daemon_status", "services.daemons.embedding_daemon", "get_status"),
    ("install_launchd", "services.daemons.embedding_daemon", "install_launchd"),
    ("uninstall_launchd", "services.daemons.embedding_daemon", "uninstall_launchd"),
    ("AutonomyDaemon", "services.daemons.autonomy_daemon", "AutonomyDaemon"),
    ("get_autonomy_status", "services.daemons.autonomy_daemon", "get_status"),
    ("install_autonomy_launchd", "services.daemons.autonomy_daemon", "install_launchd"),
    ("uninstall_autonomy_launchd", "services.daemons.autonomy_daemon", "uninstall_launchd"),
    ("ObservabilityDaemon", "services.daemons.observability_daemon", "ObservabilityDaemon"),
    ("get_observability_status", "services.daemons.observability_daemon", "get_status"),
    ("AFSSyncService", "services.afs_sync", "AFSSyncService"),
    ("SyncProfile", "services.afs_sync", "SyncProfile"),
    ("SyncRegistry", "services.afs_sync", "SyncRegistry"),
    ("SyncTarget", "services.afs_sync", "SyncTarget"),
    ("SyncResult", "services.afs_sync", "SyncResult"),
    ("SynergyService", "services.synergy_service", "SynergyService"),
    ("SynergyStatus", "services.synergy_service", "SynergyStatus"),
    ("SynergySummary", "services.synergy_service", "SynergySummary"),
]

_EXPORT_MAP = {name: (module_path, attr) for name, module_path, attr in _EXPORTS}


def __getattr__(name: str):
    if name in _EXPORT_MAP:
        warnings.warn(
            _DEPRECATION_MESSAGE,
            DeprecationWarning,
            stacklevel=2,
        )
        module_path, attr = _EXPORT_MAP[name]
        try:
            module = importlib.import_module(module_path)
            return getattr(module, attr)
        except (ImportError, AttributeError):
            # Special handling for legacy try/except blocks in services
            if name in ["AutonomyDaemon", "ObservabilityDaemon"]:
                return None
            raise
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_EXPORT_MAP.keys()))


__all__ = [name for name, _, _ in _EXPORTS]
