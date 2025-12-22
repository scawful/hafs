import importlib
import warnings

_DEPRECATION_MESSAGE = "llm is deprecated. Import from 'llm' instead."

warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)

def __getattr__(name: str):
    module = importlib.import_module("llm")
    return getattr(module, name)

def __dir__():
    module = importlib.import_module("llm")
    return dir(module)
