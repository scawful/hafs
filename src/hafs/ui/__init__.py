import importlib
import warnings

_DEPRECATION_MESSAGE = "tui is deprecated. Import from 'tui' instead."

warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)

def __getattr__(name: str):
    module = importlib.import_module("tui")
    return getattr(module, name)

def __dir__():
    module = importlib.import_module("tui")
    return dir(module)
