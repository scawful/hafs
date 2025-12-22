"""AFS (Agentic File System) management."""

from core.afs.discovery import discover_projects, find_context_root
from core.afs.manager import AFSManager
from core.afs.policy import PolicyEnforcer

__all__ = [
    "AFSManager",
    "PolicyEnforcer",
    "discover_projects",
    "find_context_root",
]
