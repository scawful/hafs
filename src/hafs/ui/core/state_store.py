"""State Store - Centralized reactive state container.

This module provides a centralized state management system for the TUI,
inspired by Redux/Zustand patterns but adapted for Textual's reactive system.

State Slices:
- navigation: Current screen, history, params
- agents: Active agents, statuses, lanes
- context: Open files, active project, AFS state
- chat: Messages, streaming state, input buffer
- metrics: Token usage, latency, costs
- analysis: Research analysis results
- settings: User preferences, keybindings

Usage:
    store = StateStore()

    # Get state
    agents = store.get("agents")

    # Update state
    store.set("agents.active", ["planner", "coder"])

    # Subscribe to changes
    store.subscribe("agents.*", lambda path, old, new: print(f"{path}: {old} -> {new}"))

    # Batch updates
    with store.batch():
        store.set("agents.active", [...])
        store.set("agents.statuses.planner", "thinking")
"""

from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class StateSlice(Generic[T]):
    """A typed slice of state with optional default value.

    Provides type-safe access to a portion of the state tree.
    """
    path: str
    default: T = field(default=None)
    validator: Optional[Callable[[T], bool]] = None

    def get(self, store: "StateStore") -> T:
        """Get the value from the store."""
        return store.get(self.path, self.default)

    def set(self, store: "StateStore", value: T) -> None:
        """Set the value in the store."""
        if self.validator and not self.validator(value):
            raise ValueError(f"Invalid value for {self.path}: {value}")
        store.set(self.path, value)


StateChangeHandler = Callable[[str, Any, Any], None]


@dataclass
class StateSubscription:
    """A subscription to state changes."""
    id: str
    pattern: str
    handler: StateChangeHandler
    _store: Optional["StateStore"] = field(default=None, repr=False)

    def unsubscribe(self) -> None:
        """Unsubscribe from state changes."""
        if self._store:
            self._store.unsubscribe(self.id)


class StateStore:
    """Centralized reactive state container.

    Stores application state in a nested dictionary structure.
    Supports dot-notation paths for accessing nested values.
    Notifies subscribers when state changes.
    """

    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        self._state: Dict[str, Any] = initial_state or self._default_state()
        self._subscriptions: Dict[str, StateSubscription] = {}
        self._batch_depth: int = 0
        self._pending_notifications: List[tuple[str, Any, Any]] = []
        self._history: List[tuple[datetime, str, Any, Any]] = []
        self._max_history: int = 100
        self._frozen_paths: set[str] = set()

    def _default_state(self) -> Dict[str, Any]:
        """Create default initial state."""
        return {
            "navigation": {
                "current_path": "/dashboard",
                "history": [],
                "params": {},
            },
            "agents": {
                "active": [],
                "statuses": {},
                "lanes": {},
                "thoughts": [],
            },
            "context": {
                "open_files": [],
                "active_project": None,
                "active_file": None,
                "afs_state": {},
            },
            "chat": {
                "messages": [],
                "streaming": False,
                "streaming_message_id": None,
                "input_buffer": "",
            },
            "metrics": {
                "tokens_used": 0,
                "tokens_by_agent": {},
                "latency_history": [],
                "cost_total": 0.0,
            },
            "analysis": {
                "synergy_tom": {},
                "scaling_metrics": {},
                "review_quality": {},
                "doc_quality": {},
                "active_modes": [],
            },
            "settings": {
                "vim_mode": True,
                "theme": "hafs-halext",
                # Theme configuration
                "theme_preset": "halext",
                "theme_variant": "dark",
                # Panel visibility
                "sidebar_visible": True,
                "context_panel_visible": True,
                "synergy_panel_visible": True,
                # Panel sizes (persisted widths)
                "sidebar_width": 32,
                "context_panel_width": 30,
                "synergy_panel_width": 18,
                # Layout preset
                "layout_preset": "default",
                # Custom layout presets
                "custom_layout_presets": {},
                # Keybindings
                "keybindings": {},
            },
        }

    def get(self, path: str, default: Any = None) -> Any:
        """Get a value from the state tree.

        Args:
            path: Dot-notation path (e.g., "agents.active", "chat.messages")
            default: Default value if path doesn't exist

        Returns:
            The value at the path, or default if not found
        """
        parts = path.split(".")
        current = self._state

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default

        return current

    def set(self, path: str, value: Any) -> None:
        """Set a value in the state tree.

        Args:
            path: Dot-notation path (e.g., "agents.active", "chat.streaming")
            value: The value to set
        """
        # Check if path is frozen
        for frozen in self._frozen_paths:
            if path.startswith(frozen) or frozen.startswith(path):
                raise ValueError(f"Cannot modify frozen path: {path}")

        parts = path.split(".")
        old_value = self.get(path)

        # Navigate to parent and set value
        current = self._state
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

        # Track history
        self._history.append((datetime.now(), path, old_value, value))
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Notify subscribers
        if self._batch_depth > 0:
            self._pending_notifications.append((path, old_value, value))
        else:
            self._notify(path, old_value, value)

    def update(self, path: str, updater: Callable[[Any], Any]) -> None:
        """Update a value using a function.

        Args:
            path: Dot-notation path
            updater: Function that takes old value and returns new value
        """
        old_value = self.get(path)
        new_value = updater(old_value)
        self.set(path, new_value)

    def merge(self, path: str, values: Dict[str, Any]) -> None:
        """Merge a dictionary into the state at path.

        Args:
            path: Dot-notation path to a dictionary
            values: Dictionary of values to merge
        """
        current = self.get(path, {})
        if not isinstance(current, dict):
            raise TypeError(f"Cannot merge into non-dict at {path}")

        merged = {**current, **values}
        self.set(path, merged)

    def append(self, path: str, value: Any) -> None:
        """Append a value to a list in the state.

        Args:
            path: Dot-notation path to a list
            value: Value to append
        """
        current = self.get(path, [])
        if not isinstance(current, list):
            raise TypeError(f"Cannot append to non-list at {path}")

        self.set(path, current + [value])

    def remove(self, path: str, value: Any) -> bool:
        """Remove a value from a list in the state.

        Args:
            path: Dot-notation path to a list
            value: Value to remove

        Returns:
            True if value was found and removed
        """
        current = self.get(path, [])
        if not isinstance(current, list):
            raise TypeError(f"Cannot remove from non-list at {path}")

        if value in current:
            new_list = [v for v in current if v != value]
            self.set(path, new_list)
            return True
        return False

    def delete(self, path: str) -> bool:
        """Delete a key from the state.

        Args:
            path: Dot-notation path to delete

        Returns:
            True if path existed and was deleted
        """
        parts = path.split(".")
        current = self._state

        for part in parts[:-1]:
            if part not in current:
                return False
            current = current[part]

        if parts[-1] in current:
            old_value = current[parts[-1]]
            del current[parts[-1]]
            self._notify(path, old_value, None)
            return True
        return False

    def subscribe(
        self,
        pattern: str,
        handler: StateChangeHandler,
    ) -> StateSubscription:
        """Subscribe to state changes.

        Args:
            pattern: Path pattern to match (supports wildcards)
            handler: Callback function(path, old_value, new_value)

        Returns:
            Subscription object for unsubscribing
        """
        import uuid
        sub_id = str(uuid.uuid4())[:8]
        subscription = StateSubscription(
            id=sub_id,
            pattern=pattern,
            handler=handler,
            _store=self,
        )
        self._subscriptions[sub_id] = subscription
        logger.debug(f"StateStore: Subscribed {sub_id} to pattern '{pattern}'")
        return subscription

    def unsubscribe(self, subscription_id: str) -> bool:
        """Remove a subscription.

        Args:
            subscription_id: The subscription ID to remove

        Returns:
            True if subscription was found and removed
        """
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            logger.debug(f"StateStore: Unsubscribed {subscription_id}")
            return True
        return False

    def _notify(self, path: str, old_value: Any, new_value: Any) -> None:
        """Notify matching subscribers of a state change."""
        import fnmatch

        for sub in self._subscriptions.values():
            if fnmatch.fnmatch(path, sub.pattern):
                try:
                    sub.handler(path, old_value, new_value)
                except Exception as e:
                    logger.error(f"StateStore: Handler error for {sub.id}: {e}")

    @contextmanager
    def batch(self) -> Iterator[None]:
        """Batch multiple state updates into a single notification cycle.

        Usage:
            with store.batch():
                store.set("a", 1)
                store.set("b", 2)
            # Notifications sent here
        """
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0:
                # Flush pending notifications
                notifications = self._pending_notifications
                self._pending_notifications = []
                for path, old_value, new_value in notifications:
                    self._notify(path, old_value, new_value)

    def freeze(self, path: str) -> None:
        """Freeze a path to prevent modifications.

        Args:
            path: The path to freeze
        """
        self._frozen_paths.add(path)

    def unfreeze(self, path: str) -> None:
        """Unfreeze a previously frozen path.

        Args:
            path: The path to unfreeze
        """
        self._frozen_paths.discard(path)

    def snapshot(self) -> Dict[str, Any]:
        """Create a deep copy of the current state.

        Returns:
            Deep copy of the state tree
        """
        return copy.deepcopy(self._state)

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore state from a snapshot.

        Args:
            snapshot: Previously saved state snapshot
        """
        old_state = self._state
        self._state = copy.deepcopy(snapshot)

        # Notify all subscribers of potential changes
        if self._batch_depth == 0:
            for sub in self._subscriptions.values():
                try:
                    sub.handler("*", old_state, self._state)
                except Exception as e:
                    logger.error(f"StateStore: Restore handler error for {sub.id}: {e}")

    def get_history(self, path: Optional[str] = None, limit: int = 10) -> List[tuple]:
        """Get recent state change history.

        Args:
            path: Optional path to filter history
            limit: Maximum entries to return

        Returns:
            List of (timestamp, path, old_value, new_value) tuples
        """
        history = self._history
        if path:
            history = [h for h in history if h[1].startswith(path)]
        return history[-limit:]

    def reset(self, path: Optional[str] = None) -> None:
        """Reset state to defaults.

        Args:
            path: Optional path to reset (resets entire state if None)
        """
        if path is None:
            old_state = self._state
            self._state = self._default_state()
            for sub in self._subscriptions.values():
                try:
                    sub.handler("*", old_state, self._state)
                except Exception as e:
                    logger.error(f"StateStore: Reset handler error: {e}")
        else:
            default = self._get_default_for_path(path)
            self.set(path, default)

    def _get_default_for_path(self, path: str) -> Any:
        """Get the default value for a path."""
        defaults = self._default_state()
        parts = path.split(".")
        current = defaults

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current


# Global state store instance
_global_store: Optional[StateStore] = None


def get_state_store() -> StateStore:
    """Get the global state store instance."""
    global _global_store
    if _global_store is None:
        _global_store = StateStore()
    return _global_store


def reset_state_store() -> None:
    """Reset the global state store (for testing)."""
    global _global_store
    _global_store = None


# Pre-defined state slices for type safety
NAVIGATION_PATH = StateSlice[str]("navigation.current_path", "/dashboard")
NAVIGATION_HISTORY = StateSlice[List[str]]("navigation.history", [])
ACTIVE_AGENTS = StateSlice[List[str]]("agents.active", [])
AGENT_STATUSES = StateSlice[Dict[str, str]]("agents.statuses", {})
OPEN_FILES = StateSlice[List[str]]("context.open_files", [])
ACTIVE_PROJECT = StateSlice[Optional[str]]("context.active_project", None)
CHAT_MESSAGES = StateSlice[List[Dict]]("chat.messages", [])
CHAT_STREAMING = StateSlice[bool]("chat.streaming", False)
VIM_MODE = StateSlice[bool]("settings.vim_mode", True)
SIDEBAR_VISIBLE = StateSlice[bool]("settings.sidebar_visible", True)
