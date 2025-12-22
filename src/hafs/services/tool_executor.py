import importlib
import warnings

_DEPRECATION_MESSAGE = "services.tool_executor is deprecated. Import from 'services.tool_executor' instead."

warnings.warn(
    _DEPRECATION_MESSAGE,
    DeprecationWarning,
    stacklevel=2,
)

def __getattr__(name: str):
    module = importlib.import_module("services.tool_executor")
    return getattr(module, name)
