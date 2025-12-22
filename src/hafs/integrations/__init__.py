import importlib
import warnings

_DEPRECATION_MESSAGE = "integrations is deprecated. Import from 'integrations' instead."

warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)

def __getattr__(name: str):
    module = importlib.import_module("integrations")
    return getattr(module, name)

def __dir__():
    module = importlib.import_module("integrations")
    return dir(module)
