import importlib
import warnings

_DEPRECATION_MESSAGE = "services.embedding_service is deprecated. Import from 'services.embedding_service' instead."

warnings.warn(
    _DEPRECATION_MESSAGE,
    DeprecationWarning,
    stacklevel=2,
)

def __getattr__(name: str):
    module = importlib.import_module("services.embedding_service")
    return getattr(module, name)
