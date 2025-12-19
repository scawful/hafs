"""Binding Registry - Centralized keybinding management.

This module provides centralized management of keyboard bindings across
the TUI, supporting:

- Context-aware bindings (global, screen, widget)
- Conflict detection
- Vim mode bindings
- Which-key style leader sequences
- Runtime rebinding
- Export to cheatsheet

Contexts:
- global: Available everywhere
- screen:<name>: Only in specific screen
- widget:<name>: Only in specific widget
- mode:vim: Only in vim mode
- mode:normal: Only outside vim mode

Usage:
    registry = BindingRegistry()

    # Register bindings
    registry.register("ctrl+s", "file.save", context="global")
    registry.register("j", "nav.down", context="mode:vim")

    # Check for conflicts
    conflicts = registry.check_conflicts()

    # Get bindings for a context
    bindings = registry.get_for_context(["global", "screen:main", "mode:vim"])
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class BindingContext(str, Enum):
    """Standard binding contexts."""
    GLOBAL = "global"
    VIM_NORMAL = "mode:vim:normal"
    VIM_INSERT = "mode:vim:insert"
    VIM_VISUAL = "mode:vim:visual"
    VIM_COMMAND = "mode:vim:command"
    NORMAL = "mode:normal"


@dataclass
class Binding:
    """A keyboard binding configuration.

    Represents a single key or key sequence bound to a command.
    """
    key: str
    command_id: str
    context: str = "global"
    priority: int = 0  # Higher = takes precedence in conflicts
    description: Optional[str] = None
    source: str = "default"  # default, user, plugin:<name>
    enabled: bool = True

    def __post_init__(self):
        # Normalize key format
        self.key = self._normalize_key(self.key)

    @staticmethod
    def _normalize_key(key: str) -> str:
        """Normalize key notation to consistent format."""
        # Convert common variations
        key = key.lower()
        key = key.replace("control+", "ctrl+")
        key = key.replace("command+", "cmd+")
        key = key.replace("option+", "alt+")
        key = key.replace("meta+", "cmd+")

        # Sort modifiers consistently
        parts = key.split("+")
        if len(parts) > 1:
            modifiers = sorted(parts[:-1])
            final_key = parts[-1]
            key = "+".join(modifiers + [final_key])

        return key

    @property
    def is_sequence(self) -> bool:
        """Check if this is a key sequence (e.g., 'space f a')."""
        return " " in self.key

    @property
    def sequence_parts(self) -> List[str]:
        """Get parts of a key sequence."""
        if self.is_sequence:
            return self.key.split(" ")
        return [self.key]

    @property
    def modifiers(self) -> Set[str]:
        """Get modifier keys."""
        if "+" not in self.key:
            return set()
        parts = self.key.split("+")
        return set(parts[:-1])


@dataclass
class BindingConflict:
    """A conflict between two bindings."""
    key: str
    bindings: List[Binding]
    contexts: List[str]
    resolution: Optional[str] = None  # Which binding wins

    def __str__(self) -> str:
        binding_strs = [f"{b.command_id}({b.context})" for b in self.bindings]
        return f"Conflict on '{self.key}': {', '.join(binding_strs)}"


@dataclass
class WhichKeyGroup:
    """A group in the which-key hierarchy."""
    prefix: str
    name: str
    icon: Optional[str] = None
    bindings: Dict[str, "WhichKeyEntry"] = field(default_factory=dict)


@dataclass
class WhichKeyEntry:
    """An entry in which-key (either a command or nested group)."""
    key: str
    label: str
    command_id: Optional[str] = None
    group: Optional[WhichKeyGroup] = None
    icon: Optional[str] = None


class BindingRegistry:
    """Centralized registry for all keyboard bindings.

    Manages bindings across different contexts and supports:
    - Context-aware binding lookup
    - Conflict detection and resolution
    - Vim mode integration
    - Which-key style sequences
    - Runtime modifications
    """

    def __init__(self):
        self._bindings: Dict[str, List[Binding]] = {}  # key -> bindings
        self._contexts: Dict[str, Set[str]] = {}  # context -> keys
        self._which_key: Dict[str, WhichKeyGroup] = {}  # prefix -> group
        self._leader: str = "space"

    def register(
        self,
        key: str,
        command_id: str,
        context: str = "global",
        priority: int = 0,
        description: Optional[str] = None,
        source: str = "default",
    ) -> Binding:
        """Register a keybinding.

        Args:
            key: The key or key sequence (e.g., "ctrl+s", "space f a")
            command_id: The command to execute
            context: The context where this binding is active
            priority: Higher priority wins conflicts
            description: Human-readable description
            source: Origin of the binding (default, user, plugin)

        Returns:
            The created Binding object
        """
        binding = Binding(
            key=key,
            command_id=command_id,
            context=context,
            priority=priority,
            description=description,
            source=source,
        )

        # Store by key
        if binding.key not in self._bindings:
            self._bindings[binding.key] = []
        self._bindings[binding.key].append(binding)

        # Index by context
        if context not in self._contexts:
            self._contexts[context] = set()
        self._contexts[context].add(binding.key)

        logger.debug(f"BindingRegistry: Registered '{key}' -> '{command_id}' in {context}")
        return binding

    def unregister(self, key: str, context: Optional[str] = None) -> int:
        """Remove bindings for a key.

        Args:
            key: The key to unbind
            context: Optional context to limit removal

        Returns:
            Number of bindings removed
        """
        key = Binding._normalize_key(key)
        if key not in self._bindings:
            return 0

        if context is None:
            count = len(self._bindings[key])
            del self._bindings[key]
            return count

        original = self._bindings[key]
        self._bindings[key] = [b for b in original if b.context != context]
        removed = len(original) - len(self._bindings[key])

        if not self._bindings[key]:
            del self._bindings[key]

        if context in self._contexts:
            self._contexts[context].discard(key)

        return removed

    def get(self, key: str) -> List[Binding]:
        """Get all bindings for a key.

        Args:
            key: The key to look up

        Returns:
            List of bindings (may be empty)
        """
        key = Binding._normalize_key(key)
        return self._bindings.get(key, [])

    def get_for_context(
        self,
        active_contexts: List[str],
    ) -> Dict[str, Binding]:
        """Get effective bindings for a set of active contexts.

        When multiple bindings exist for the same key, the one with
        highest priority in the most specific context wins.

        Args:
            active_contexts: List of currently active contexts

        Returns:
            Dict mapping key to winning binding
        """
        effective: Dict[str, Binding] = {}

        for key, bindings in self._bindings.items():
            best_binding: Optional[Binding] = None
            best_score = -1

            for binding in bindings:
                if not binding.enabled:
                    continue

                if binding.context not in active_contexts and binding.context != "global":
                    continue

                # Score based on context specificity and priority
                score = binding.priority
                if binding.context != "global":
                    score += 100  # Prefer specific contexts
                if binding.context in active_contexts:
                    score += active_contexts.index(binding.context) * 10

                if score > best_score:
                    best_score = score
                    best_binding = binding

            if best_binding:
                effective[key] = best_binding

        return effective

    def get_by_command(self, command_id: str) -> List[Binding]:
        """Get all bindings for a command.

        Args:
            command_id: The command to look up

        Returns:
            List of bindings for the command
        """
        result = []
        for bindings in self._bindings.values():
            for binding in bindings:
                if binding.command_id == command_id:
                    result.append(binding)
        return result

    def check_conflicts(
        self,
        contexts: Optional[List[str]] = None,
    ) -> List[BindingConflict]:
        """Check for binding conflicts.

        Args:
            contexts: Optional list of contexts to check

        Returns:
            List of conflicts found
        """
        conflicts = []

        for key, bindings in self._bindings.items():
            if len(bindings) <= 1:
                continue

            # Group by context overlap
            context_groups: Dict[str, List[Binding]] = {}
            for binding in bindings:
                ctx = binding.context
                if contexts and ctx not in contexts and ctx != "global":
                    continue

                if ctx not in context_groups:
                    context_groups[ctx] = []
                context_groups[ctx].append(binding)

            # Check for same-context conflicts
            for ctx, ctx_bindings in context_groups.items():
                if len(ctx_bindings) > 1:
                    # Determine winner by priority
                    sorted_bindings = sorted(ctx_bindings, key=lambda b: b.priority, reverse=True)
                    winner = sorted_bindings[0]

                    conflicts.append(BindingConflict(
                        key=key,
                        bindings=ctx_bindings,
                        contexts=[ctx],
                        resolution=winner.command_id,
                    ))

        return conflicts

    def set_leader(self, leader: str) -> None:
        """Set the leader key for which-key sequences.

        Args:
            leader: The leader key (default: "space")
        """
        self._leader = Binding._normalize_key(leader)

    def get_leader(self) -> str:
        """Get the current leader key."""
        return self._leader

    def register_which_key_group(
        self,
        prefix: str,
        name: str,
        icon: Optional[str] = None,
    ) -> WhichKeyGroup:
        """Register a which-key group.

        Args:
            prefix: Key sequence prefix (e.g., "space f" for file operations)
            name: Display name for the group
            icon: Optional icon

        Returns:
            The created group
        """
        group = WhichKeyGroup(prefix=prefix, name=name, icon=icon)
        self._which_key[prefix] = group
        return group

    def register_which_key_binding(
        self,
        prefix: str,
        key: str,
        label: str,
        command_id: str,
        icon: Optional[str] = None,
    ) -> None:
        """Register a binding within a which-key group.

        Args:
            prefix: The group prefix
            key: The final key in the sequence
            label: Display label
            command_id: The command to execute
            icon: Optional icon
        """
        if prefix not in self._which_key:
            self.register_which_key_group(prefix, prefix)

        entry = WhichKeyEntry(
            key=key,
            label=label,
            command_id=command_id,
            icon=icon,
        )
        self._which_key[prefix].bindings[key] = entry

        # Also register as a regular binding
        full_key = f"{prefix} {key}"
        self.register(
            full_key,
            command_id,
            context="global",
            description=label,
        )

    def get_which_key_group(self, prefix: str) -> Optional[WhichKeyGroup]:
        """Get a which-key group by prefix.

        Args:
            prefix: The prefix to look up

        Returns:
            The group if found
        """
        return self._which_key.get(prefix)

    def get_which_key_hints(self, prefix: str) -> List[Tuple[str, str, Optional[str]]]:
        """Get hints for which-key display.

        Args:
            prefix: Current key sequence prefix

        Returns:
            List of (key, label, icon) tuples
        """
        hints = []

        # Check for exact group match
        if prefix in self._which_key:
            group = self._which_key[prefix]
            for key, entry in sorted(group.bindings.items()):
                if entry.group:
                    hints.append((key, f"+{entry.label}", entry.icon))
                else:
                    hints.append((key, entry.label, entry.icon))

        # Check for subgroups
        for group_prefix, group in self._which_key.items():
            if group_prefix.startswith(prefix + " "):
                next_key = group_prefix[len(prefix) + 1:].split(" ")[0]
                if not any(h[0] == next_key for h in hints):
                    hints.append((next_key, f"+{group.name}", group.icon))

        return sorted(hints, key=lambda h: h[0])

    def is_sequence_prefix(self, key: str) -> bool:
        """Check if a key is a prefix for any sequence.

        Args:
            key: The key to check

        Returns:
            True if this key starts a sequence
        """
        key = Binding._normalize_key(key)

        for bound_key in self._bindings:
            if bound_key.startswith(key + " "):
                return True

        for prefix in self._which_key:
            if prefix.startswith(key):
                return True

        return False

    def export_cheatsheet(self, contexts: Optional[List[str]] = None) -> str:
        """Export bindings as a markdown cheatsheet.

        Args:
            contexts: Optional contexts to include

        Returns:
            Markdown-formatted binding reference
        """
        lines = ["# Keyboard Shortcuts\n"]

        # Group by context
        context_bindings: Dict[str, List[Binding]] = {}
        for bindings in self._bindings.values():
            for binding in bindings:
                if contexts and binding.context not in contexts:
                    continue
                if binding.context not in context_bindings:
                    context_bindings[binding.context] = []
                context_bindings[binding.context].append(binding)

        for context in sorted(context_bindings.keys()):
            lines.append(f"\n## {context.title()}\n")
            lines.append("| Key | Command | Description |")
            lines.append("|-----|---------|-------------|")

            for binding in sorted(context_bindings[context], key=lambda b: b.key):
                desc = binding.description or ""
                lines.append(f"| `{binding.key}` | {binding.command_id} | {desc} |")

        return "\n".join(lines)

    def load_user_bindings(self, bindings: Dict[str, Any]) -> int:
        """Load user-defined bindings from config.

        Args:
            bindings: Dict of binding definitions

        Returns:
            Number of bindings loaded
        """
        count = 0
        for key, config in bindings.items():
            if isinstance(config, str):
                # Simple format: key: command_id
                self.register(key, config, source="user")
                count += 1
            elif isinstance(config, dict):
                # Full format with options
                self.register(
                    key,
                    config.get("command", config.get("command_id", "")),
                    context=config.get("context", "global"),
                    priority=config.get("priority", 50),
                    description=config.get("description"),
                    source="user",
                )
                count += 1

        logger.info(f"BindingRegistry: Loaded {count} user bindings")
        return count


# Global binding registry instance
_global_registry: Optional[BindingRegistry] = None


def get_binding_registry() -> BindingRegistry:
    """Get the global binding registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = BindingRegistry()
        _register_default_bindings(_global_registry)
    return _global_registry


def reset_binding_registry() -> None:
    """Reset the global binding registry (for testing)."""
    global _global_registry
    _global_registry = None


def _register_default_bindings(registry: BindingRegistry) -> None:
    """Register default bindings."""

    # Global bindings
    registry.register("q", "system.quit", context="global", description="Quit")
    registry.register("?", "help.show", context="global", description="Show help")
    registry.register("r", "view.refresh", context="global", description="Refresh")
    registry.register("ctrl+shift+p", "system.command_palette", context="global", description="Command palette")

    # Navigation bindings
    registry.register("1", "nav.dashboard", context="global", description="Dashboard")
    registry.register("2", "nav.logs", context="global", description="Logs")
    registry.register("3", "nav.settings", context="global", description="Settings")
    registry.register("4", "nav.chat", context="global", description="Chat")
    registry.register("5", "nav.services", context="global", description="Services")

    # Vim mode bindings
    registry.register("j", "nav.down", context=BindingContext.VIM_NORMAL, description="Move down")
    registry.register("k", "nav.up", context=BindingContext.VIM_NORMAL, description="Move up")
    registry.register("h", "nav.left", context=BindingContext.VIM_NORMAL, description="Move left")
    registry.register("l", "nav.right", context=BindingContext.VIM_NORMAL, description="Move right")
    registry.register("gg", "nav.top", context=BindingContext.VIM_NORMAL, description="Go to top")
    registry.register("shift+g", "nav.bottom", context=BindingContext.VIM_NORMAL, description="Go to bottom")
    registry.register("/", "search.start", context=BindingContext.VIM_NORMAL, description="Search")
    registry.register("n", "search.next", context=BindingContext.VIM_NORMAL, description="Next match")
    registry.register("shift+n", "search.prev", context=BindingContext.VIM_NORMAL, description="Previous match")
    registry.register("escape", "mode.normal", context=BindingContext.VIM_INSERT, description="Exit insert mode")
    registry.register("i", "mode.insert", context=BindingContext.VIM_NORMAL, description="Enter insert mode")

    # Which-key groups (space leader)
    registry.register_which_key_group("space", "Leader", icon="")
    registry.register_which_key_group("space f", "File", icon="")
    registry.register_which_key_group("space p", "Project", icon="")
    registry.register_which_key_group("space c", "Context", icon="")
    registry.register_which_key_group("space a", "Agent", icon="")
    registry.register_which_key_group("space s", "Search", icon="")
    registry.register_which_key_group("space g", "Git", icon="")

    # File operations
    registry.register_which_key_binding("space f", "s", "Save", "file.save", icon="")
    registry.register_which_key_binding("space f", "o", "Open", "file.open", icon="")
    registry.register_which_key_binding("space f", "n", "New", "file.new", icon="")
    registry.register_which_key_binding("space f", "d", "Delete", "file.delete", icon="")
    registry.register_which_key_binding("space f", "r", "Rename", "file.rename", icon="")

    # Project operations
    registry.register_which_key_binding("space p", "s", "Switch Project", "project.switch", icon="")
    registry.register_which_key_binding("space p", "r", "Refresh", "project.refresh", icon="")

    # Context operations
    registry.register_which_key_binding("space c", "m", "Mount", "context.mount", icon="")
    registry.register_which_key_binding("space c", "s", "Sync", "context.sync", icon="")
    registry.register_which_key_binding("space c", "r", "Refresh", "context.refresh", icon="")

    # Agent operations
    registry.register_which_key_binding("space a", "a", "Add Agent", "agent.add", icon="")
    registry.register_which_key_binding("space a", "r", "Remove Agent", "agent.remove", icon="")
    registry.register_which_key_binding("space a", "l", "List Agents", "agent.list", icon="")

    # Search operations
    registry.register_which_key_binding("space s", "f", "Find File", "search.file", icon="")
    registry.register_which_key_binding("space s", "g", "Grep", "search.grep", icon="")
    registry.register_which_key_binding("space s", "s", "Semantic Search", "search.semantic", icon="")

    # View toggles
    registry.register("ctrl+b", "view.toggle_sidebar", context="global", description="Toggle sidebar")
