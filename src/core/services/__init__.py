"""Services for HAFS.

DEPRECATED: This module re-exports from the new 'services' package.
Please import directly from 'services' instead.

Example:
    # Old (deprecated):
    from services import ServiceManager

    # New (preferred):
    from services import ServiceManager
"""

from __future__ import annotations

import warnings

# Emit deprecation warning on import
warnings.warn(
    "services is deprecated. Import from 'services' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the new canonical location
from services import (
    ServiceManager,
    ServiceDefinition,
    ServiceState,
    ServiceStatus,
    ServiceType,
)

__all__ = [
    "ServiceManager",
    "ServiceDefinition",
    "ServiceState",
    "ServiceStatus",
    "ServiceType",
]
