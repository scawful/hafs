import importlib
import warnings

_DEPRECATION_MESSAGE = "services.synergy_service is deprecated. Import from 'services.synergy_service' instead."

warnings.warn(
    _DEPRECATION_MESSAGE,
    DeprecationWarning,
    stacklevel=2,
)

def __getattr__(name: str):
    module = importlib.import_module("services.synergy_service")
    return getattr(module, name)
