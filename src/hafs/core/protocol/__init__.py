"""Cognitive protocol helpers (agent-agnostic).

These functions implement lightweight, deterministic interactions with the
`.context/` protocol artifacts so any agent (oracle-code or not) can
participate via file operations and a small set of UI commands.
"""

from hafs.core.protocol.actions import (
    ProtocolFiles,
    append_deferred,
    ensure_protocol,
    open_protocol_file,
    set_primary_goal,
    snapshot_state,
)

__all__ = [
    "ProtocolFiles",
    "append_deferred",
    "ensure_protocol",
    "open_protocol_file",
    "set_primary_goal",
    "snapshot_state",
]

