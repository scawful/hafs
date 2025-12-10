"""AFS (Agentic File System) management."""

from hafs.core.afs.manager import AFSManager
from hafs.core.afs.policy import PolicyEnforcer
from hafs.core.afs.discovery import discover_projects, find_context_root

__all__ = [
    "AFSManager",
    "PolicyEnforcer",
    "discover_projects",
    "find_context_root",
]
