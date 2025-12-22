import importlib
import warnings

_DEPRECATION_MESSAGE = "services.autonomy_daemon is deprecated. Import from 'services.daemons.autonomy_daemon' instead."

warnings.warn(
    _DEPRECATION_MESSAGE,
    DeprecationWarning,
    stacklevel=2,
)

def __getattr__(name: str):
    module = importlib.import_module("services.daemons.autonomy_daemon")
    return getattr(module, name)
