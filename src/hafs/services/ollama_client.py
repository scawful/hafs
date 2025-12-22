import importlib
import warnings

_DEPRECATION_MESSAGE = "services.ollama_client is deprecated. Import from 'services.ollama_client' instead."

warnings.warn(
    _DEPRECATION_MESSAGE,
    DeprecationWarning,
    stacklevel=2,
)

def __getattr__(name: str):
    module = importlib.import_module("services.ollama_client")
    return getattr(module, name)
