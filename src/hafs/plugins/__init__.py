import importlib
import warnings

_DEPRECATION_MESSAGE = "plugins is deprecated. Import from 'plugins' instead."

warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)

def __getattr__(name: str):
    module = importlib.import_module("plugins")
    return getattr(module, name)

def __dir__():
    module = importlib.import_module("plugins")
    return dir(module)
