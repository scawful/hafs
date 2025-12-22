import importlib
import warnings

_DEPRECATION_MESSAGE = "adapters is deprecated. Import from 'adapters' instead."

warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)

def __getattr__(name: str):
    module = importlib.import_module("adapters")
    return getattr(module, name)

def __dir__():
    module = importlib.import_module("adapters")
    return dir(module)
