import importlib
import warnings

_DEPRECATION_MESSAGE = "services.observability_daemon is deprecated. Import from 'services.daemons.observability_daemon' instead."

warnings.warn(
    _DEPRECATION_MESSAGE,
    DeprecationWarning,
    stacklevel=2,
)

def __getattr__(name: str):
    module = importlib.import_module("services.daemons.observability_daemon")
    return getattr(module, name)
