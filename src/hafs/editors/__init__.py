import importlib
import warnings

_DEPRECATION_MESSAGE = "editors is deprecated. Import from 'editors' instead."

warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)

def __getattr__(name: str):
    module = importlib.import_module("editors")
    return getattr(module, name)

def __dir__():
    module = importlib.import_module("editors")
    return dir(module)
