import importlib
import warnings

_DEPRECATION_MESSAGE = "services.context_agent_daemon is deprecated. Import from 'services.daemons.context_agent_daemon' instead."

warnings.warn(
    _DEPRECATION_MESSAGE,
    DeprecationWarning,
    stacklevel=2,
)

def __getattr__(name: str):
    module = importlib.import_module("services.daemons.context_agent_daemon")
    return getattr(module, name)
