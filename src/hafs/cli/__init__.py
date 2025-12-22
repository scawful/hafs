import importlib
import os
import warnings

_DEPRECATION_MESSAGE = "cli is deprecated. Import from 'cli' instead."

if os.environ.get("HAFS_CLI_WARN_DEPRECATED") == "1":
    warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)

def __getattr__(name: str):
    module = importlib.import_module("cli")
    return getattr(module, name)

def __dir__():
    module = importlib.import_module("cli")
    return dir(module)
