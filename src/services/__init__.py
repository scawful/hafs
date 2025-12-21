"""Services for HAFS.

This is the canonical location for service management. For backward compatibility,
hafs.core.services re-exports from this module.

This module provides cross-platform service management for running
HAFS components as background daemons.
"""

from services.manager import ServiceManager
from services.models import (
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
