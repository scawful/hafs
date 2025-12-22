import importlib
import warnings

_DEPRECATION_MESSAGE = "services.local_ai_orchestrator is deprecated. Import from 'services.local_ai_orchestrator' instead."

warnings.warn(
    _DEPRECATION_MESSAGE,
    DeprecationWarning,
    stacklevel=2,
)

def __getattr__(name: str):
    module = importlib.import_module("services.local_ai_orchestrator")
    return getattr(module, name)
