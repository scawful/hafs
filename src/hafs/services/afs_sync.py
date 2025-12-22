import importlib
import warnings

_DEPRECATION_MESSAGE = "services.afs_sync is deprecated. Import from 'services.afs_sync' instead."

warnings.warn(
    _DEPRECATION_MESSAGE,
    DeprecationWarning,
    stacklevel=2,
)

def __getattr__(name: str):
    module = importlib.import_module("services.afs_sync")
    return getattr(module, name)
