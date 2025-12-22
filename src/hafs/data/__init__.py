import warnings
from pathlib import Path

_DEPRECATION_MESSAGE = "data is deprecated. Access via 'data' top-level directory."

warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)

# Data is usually accessed via Path, so we provide an absolute path helper if needed.
# But for now, we'll just keep the __init__.py minimal.
