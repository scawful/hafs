"""Command Registry - Extensible command registration system.

This module provides a central registry for all TUI commands, enabling:
- Command discovery and fuzzy search
- Plugin-registered commands
- Keybinding association
- Command categorization
- Command history and recent commands

Categories:
- file: File operations (open, save, close, new)
- agent: Agent management (add, remove, restart)
- context: Context operations (mount, sync, refresh)
- view: View/display commands (toggle sidebar, zoom)
- navigation: Navigation (go to screen, back, forward)
- tools: Tool execution (run, test, build)
- search: Search commands (find, grep, replace)
- help: Help and documentation

Usage:
    registry = CommandRegistry()

    # Register a command
    registry.register(Command(
        id="file.save",
        name="Save File",
        description="Save the current file",
        handler=save_handler,
        keybinding="ctrl+s",
        category="file",
    ))

    # Search commands
    matches = registry.search("save")

    # Execute by ID
    registry.execute("file.save", path="/path/to/file")
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class CommandCategory(str, Enum):
    """Command categories for organization and filtering."""

    FILE = "file"
    AGENT = "agent"
    CONTEXT = "context"
    VIEW = "view"
    NAVIGATION = "navigation"
    TOOLS = "tools"
    SEARCH = "search"
    HELP = "help"
    CHAT = "chat"
    ANALYSIS = "analysis"
    SYSTEM = "system"


CommandHandler = Callable[..., Any]
AsyncCommandHandler = Callable[..., Awaitable[Any]]


@dataclass
class Command:
    """A registered command in the TUI.

    Commands are the primary way users interact with the TUI.
    They can be invoked via keybindings, command palette, or programmatically.
    """

    id: str
    name: str
    description: str
    handler: Union[CommandHandler, AsyncCommandHandler]
    category: Union[str, CommandCategory] = CommandCategory.SYSTEM
    keybinding: Optional[str] = None
    icon: Optional[str] = None
    is_async: bool = False
    enabled: bool = True
    visible: bool = True
    requires_confirmation: bool = False
    confirmation_message: Optional[str] = None
    parameters: List["CommandParameter"] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    source: str = "core"  # core, plugin:name, user

    def __post_init__(self):
        if isinstance(self.category, str):
            try:
                self.category = CommandCategory(self.category)
            except ValueError:
                pass  # Keep as string for custom categories


@dataclass
class CommandParameter:
    """A parameter for a command."""

    name: str
    description: str
    type: str = "string"  # string, int, float, bool, path, choice
    required: bool = False
    default: Any = None
    choices: Optional[List[str]] = None


@dataclass
class CommandResult:
    """Result of executing a command."""

    success: bool
    command_id: str
    output: Any = None
    error: Optional[str] = None
    duration_ms: int = 0


@dataclass
class CommandHistoryEntry:
    """An entry in command history."""

    command_id: str
    timestamp: datetime
    args: Dict[str, Any]
    result: CommandResult


class CommandRegistry:
    """Central registry for all TUI commands.

    Provides:
    - Command registration and lookup
    - Fuzzy search across commands
    - Category-based filtering
    - Command execution with history
    - Recent commands tracking
    """

    # Category icons for UI display
    CATEGORY_ICONS = {
        CommandCategory.FILE: "",
        CommandCategory.AGENT: "",
        CommandCategory.CONTEXT: "",
        CommandCategory.VIEW: "",
        CommandCategory.NAVIGATION: "",
        CommandCategory.TOOLS: "",
        CommandCategory.SEARCH: "",
        CommandCategory.HELP: "",
        CommandCategory.CHAT: "",
        CommandCategory.ANALYSIS: "",
        CommandCategory.SYSTEM: "",
    }

    def __init__(self, max_history: int = 100, max_recent: int = 10):
        self._commands: Dict[str, Command] = {}
        self._aliases: Dict[str, str] = {}  # alias -> command_id
        self._history: deque[CommandHistoryEntry] = deque(maxlen=max_history)
        self._recent: deque[str] = deque(maxlen=max_recent)
        self._disabled_commands: set[str] = set()

    def register(self, command: Command) -> None:
        """Register a command.

        Args:
            command: The command to register

        Raises:
            ValueError: If command ID already exists
        """
        if command.id in self._commands:
            raise ValueError(f"Command already registered: {command.id}")

        self._commands[command.id] = command

        # Register aliases
        for alias in command.aliases:
            self._aliases[alias] = command.id

        logger.debug(f"CommandRegistry: Registered '{command.id}' ({command.name})")

    def unregister(self, command_id: str) -> bool:
        """Unregister a command.

        Args:
            command_id: The command ID to remove

        Returns:
            True if command was found and removed
        """
        if command_id in self._commands:
            command = self._commands[command_id]

            # Remove aliases
            for alias in command.aliases:
                self._aliases.pop(alias, None)

            del self._commands[command_id]
            logger.debug(f"CommandRegistry: Unregistered '{command_id}'")
            return True
        return False

    def get(self, command_id: str) -> Optional[Command]:
        """Get a command by ID or alias.

        Args:
            command_id: The command ID or alias

        Returns:
            The command if found, None otherwise
        """
        # Check direct ID first
        if command_id in self._commands:
            return self._commands[command_id]

        # Check aliases
        if command_id in self._aliases:
            return self._commands.get(self._aliases[command_id])

        return None

    def get_by_category(
        self,
        category: Union[str, CommandCategory],
        include_disabled: bool = False,
    ) -> List[Command]:
        """Get all commands in a category.

        Args:
            category: The category to filter by
            include_disabled: Include disabled commands

        Returns:
            List of commands in the category
        """
        if isinstance(category, str):
            try:
                category = CommandCategory(category)
            except ValueError:
                pass

        commands = []
        for cmd in self._commands.values():
            if cmd.category == category:
                if include_disabled or (cmd.enabled and cmd.id not in self._disabled_commands):
                    commands.append(cmd)

        return sorted(commands, key=lambda c: c.name)

    def get_all(self, include_disabled: bool = False) -> List[Command]:
        """Get all registered commands.

        Args:
            include_disabled: Include disabled commands

        Returns:
            List of all commands
        """
        commands = []
        for cmd in self._commands.values():
            if include_disabled or (cmd.enabled and cmd.id not in self._disabled_commands):
                commands.append(cmd)
        return sorted(
            commands,
            key=lambda c: (
                c.category.value if isinstance(c.category, CommandCategory) else str(c.category),
                c.name,
            ),
        )

    def search(
        self,
        query: str,
        limit: int = 10,
        category: Optional[Union[str, CommandCategory]] = None,
    ) -> List[Command]:
        """Search for commands by fuzzy matching.

        Args:
            query: Search query
            limit: Maximum results to return
            category: Optional category filter

        Returns:
            List of matching commands, sorted by relevance
        """
        if not query:
            return self.get_recent()[:limit]

        query_lower = query.lower()
        scored_commands = []

        for cmd in self._commands.values():
            if not cmd.enabled or cmd.id in self._disabled_commands:
                continue

            if category and cmd.category != category:
                continue

            # Calculate relevance score
            score = self._calculate_match_score(query_lower, cmd)
            if score > 0:
                scored_commands.append((score, cmd))

        # Sort by score descending
        scored_commands.sort(key=lambda x: x[0], reverse=True)

        return [cmd for _, cmd in scored_commands[:limit]]

    def _calculate_match_score(self, query: str, command: Command) -> float:
        """Calculate fuzzy match score for a command."""
        score = 0.0

        name_lower = command.name.lower()
        id_lower = command.id.lower()
        desc_lower = command.description.lower()

        # Exact matches get highest score
        if query == name_lower:
            score += 100
        elif query == id_lower:
            score += 90
        # Prefix matches
        elif name_lower.startswith(query):
            score += 80
        elif id_lower.startswith(query):
            score += 70
        # Contains matches
        elif query in name_lower:
            score += 50 + (len(query) / len(name_lower)) * 20
        elif query in id_lower:
            score += 40 + (len(query) / len(id_lower)) * 20
        elif query in desc_lower:
            score += 20 + (len(query) / len(desc_lower)) * 10

        # Check aliases
        for alias in command.aliases:
            alias_lower = alias.lower()
            if query == alias_lower:
                score = max(score, 85)
            elif alias_lower.startswith(query):
                score = max(score, 65)
            elif query in alias_lower:
                score = max(score, 35)

        # Boost recently used commands
        if command.id in self._recent:
            recent_index = list(self._recent).index(command.id)
            score += 10 - recent_index  # More recent = higher boost

        return score

    def execute(self, command_id: str, **kwargs: Any) -> CommandResult:
        """Execute a command by ID.

        Args:
            command_id: The command ID or alias
            **kwargs: Arguments to pass to the handler

        Returns:
            CommandResult with success status and output
        """
        import time

        command = self.get(command_id)
        if not command:
            return CommandResult(
                success=False,
                command_id=command_id,
                error=f"Command not found: {command_id}",
            )

        if not command.enabled or command.id in self._disabled_commands:
            return CommandResult(
                success=False,
                command_id=command.id,
                error=f"Command is disabled: {command.id}",
            )

        start = time.monotonic()
        try:
            if command.is_async:
                # For async commands, return a future
                coro = command.handler(**kwargs)
                if asyncio.iscoroutine(coro):
                    from typing import cast, Coroutine

                    output = asyncio.create_task(cast(Coroutine[Any, Any, Any], coro))
                else:
                    output = coro
            else:
                output = command.handler(**kwargs)

            duration_ms = int((time.monotonic() - start) * 1000)
            result = CommandResult(
                success=True,
                command_id=command.id,
                output=output,
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            result = CommandResult(
                success=False,
                command_id=command.id,
                error=str(e),
                duration_ms=duration_ms,
            )
            logger.error(f"CommandRegistry: Error executing '{command.id}': {e}")

        # Record in history
        entry = CommandHistoryEntry(
            command_id=command.id,
            timestamp=datetime.now(),
            args=kwargs,
            result=result,
        )
        self._history.append(entry)

        # Update recent commands
        if command.id in self._recent:
            self._recent.remove(command.id)
        self._recent.appendleft(command.id)

        return result

    async def execute_async(self, command_id: str, **kwargs: Any) -> CommandResult:
        """Execute an async command by ID.

        Args:
            command_id: The command ID or alias
            **kwargs: Arguments to pass to the handler

        Returns:
            CommandResult with success status and output
        """
        import time

        command = self.get(command_id)
        if not command:
            return CommandResult(
                success=False,
                command_id=command_id,
                error=f"Command not found: {command_id}",
            )

        if not command.enabled or command.id in self._disabled_commands:
            return CommandResult(
                success=False,
                command_id=command.id,
                error=f"Command is disabled: {command.id}",
            )

        start = time.monotonic()
        try:
            if command.is_async:
                output = await command.handler(**kwargs)
            else:
                output = command.handler(**kwargs)

            duration_ms = int((time.monotonic() - start) * 1000)
            result = CommandResult(
                success=True,
                command_id=command.id,
                output=output,
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            result = CommandResult(
                success=False,
                command_id=command.id,
                error=str(e),
                duration_ms=duration_ms,
            )
            logger.error(f"CommandRegistry: Error executing '{command.id}': {e}")

        # Record in history
        entry = CommandHistoryEntry(
            command_id=command.id,
            timestamp=datetime.now(),
            args=kwargs,
            result=result,
        )
        self._history.append(entry)

        # Update recent commands
        if command.id in self._recent:
            self._recent.remove(command.id)
        self._recent.appendleft(command.id)

        return result

    def disable(self, command_id: str) -> bool:
        """Temporarily disable a command.

        Args:
            command_id: The command to disable

        Returns:
            True if command exists
        """
        if command_id in self._commands:
            self._disabled_commands.add(command_id)
            return True
        return False

    def enable(self, command_id: str) -> bool:
        """Re-enable a disabled command.

        Args:
            command_id: The command to enable

        Returns:
            True if command was disabled
        """
        if command_id in self._disabled_commands:
            self._disabled_commands.discard(command_id)
            return True
        return False

    def get_recent(self) -> List[Command]:
        """Get recently used commands.

        Returns:
            List of recently used commands
        """
        return [self._commands[cmd_id] for cmd_id in self._recent if cmd_id in self._commands]

    def get_history(self, limit: int = 20) -> List[CommandHistoryEntry]:
        """Get command execution history.

        Args:
            limit: Maximum entries to return

        Returns:
            List of recent command executions
        """
        return list(self._history)[-limit:]

    def clear_history(self) -> None:
        """Clear command history."""
        self._history.clear()

    def get_categories(self) -> List[CommandCategory]:
        """Get all categories that have commands.

        Returns:
            List of categories with registered commands
        """
        categories = set()
        for cmd in self._commands.values():
            if isinstance(cmd.category, CommandCategory):
                categories.add(cmd.category)
        return sorted(categories, key=lambda c: c.value)

    def get_keybindings(self) -> Dict[str, str]:
        """Get all command keybindings.

        Returns:
            Dict mapping keybinding to command_id
        """
        bindings = {}
        for cmd in self._commands.values():
            if cmd.keybinding:
                bindings[cmd.keybinding] = cmd.id
        return bindings

    def export_cheatsheet(self) -> str:
        """Export all commands as a markdown cheatsheet.

        Returns:
            Markdown-formatted command reference
        """
        lines = ["# Command Reference\n"]

        for category in self.get_categories():
            icon = self.CATEGORY_ICONS.get(category, "")
            lines.append(f"\n## {icon} {category.value.title()}\n")
            lines.append("| Command | Keybinding | Description |")
            lines.append("|---------|------------|-------------|")

            for cmd in self.get_by_category(category):
                binding = cmd.keybinding or "-"
                lines.append(f"| {cmd.name} | `{binding}` | {cmd.description} |")

        return "\n".join(lines)


# Global command registry instance
_global_registry: Optional[CommandRegistry] = None


def get_command_registry() -> CommandRegistry:
    """Get the global command registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = CommandRegistry()
        _register_default_commands(_global_registry)
    return _global_registry


def reset_command_registry() -> None:
    """Reset the global command registry (for testing)."""
    global _global_registry
    _global_registry = None


def _register_default_commands(registry: CommandRegistry) -> None:
    """Register default built-in commands."""

    # Navigation commands
    registry.register(
        Command(
            id="nav.dashboard",
            name="Go to Dashboard",
            description="Switch to the main dashboard screen",
            handler=lambda: None,  # Placeholder - will be connected to app
            category=CommandCategory.NAVIGATION,
            keybinding="1",
            icon="",
        )
    )

    registry.register(
        Command(
            id="nav.chat",
            name="Go to Chat",
            description="Switch to the multi-agent chat screen",
            handler=lambda: None,
            category=CommandCategory.NAVIGATION,
            keybinding="4",
            icon="",
        )
    )

    registry.register(
        Command(
            id="nav.workspace",
            name="Go to Workspace",
            description="Switch to the high-performance workspace screen",
            handler=lambda: None,
            category=CommandCategory.NAVIGATION,
            keybinding="5",
            icon="",
        )
    )

    registry.register(
        Command(
            id="nav.logs",
            name="Go to Logs",
            description="Switch to the log browser screen",
            handler=lambda: None,
            category=CommandCategory.NAVIGATION,
            keybinding="6",
            icon="",
        )
    )

    registry.register(
        Command(
            id="nav.settings",
            name="Go to Settings",
            description="Switch to the settings screen",
            handler=lambda: None,
            category=CommandCategory.NAVIGATION,
            keybinding="3",
            icon="",
        )
    )

    registry.register(
        Command(
            id="nav.services",
            name="Go to Services",
            description="Switch to the services management screen",
            handler=lambda: None,
            category=CommandCategory.NAVIGATION,
            keybinding="7",
            icon="",
        )
    )

    # File commands
    registry.register(
        Command(
            id="file.save",
            name="Save File",
            description="Save the current file",
            handler=lambda: None,
            category=CommandCategory.FILE,
            keybinding="ctrl+s",
            icon="",
        )
    )

    registry.register(
        Command(
            id="file.open",
            name="Open File",
            description="Open a file in the editor",
            handler=lambda: None,
            category=CommandCategory.FILE,
            keybinding="ctrl+o",
            icon="",
        )
    )

    # View commands
    registry.register(
        Command(
            id="view.toggle_sidebar",
            name="Toggle Sidebar",
            description="Show or hide the sidebar",
            handler=lambda: None,
            category=CommandCategory.VIEW,
            keybinding="ctrl+b",
            icon="",
        )
    )

    registry.register(
        Command(
            id="view.refresh",
            name="Refresh",
            description="Refresh the current view",
            handler=lambda: None,
            category=CommandCategory.VIEW,
            keybinding="r",
            icon="",
        )
    )

    # Help commands
    registry.register(
        Command(
            id="help.show",
            name="Show Help",
            description="Display context-aware help",
            handler=lambda: None,
            category=CommandCategory.HELP,
            keybinding="?",
            icon="",
        )
    )

    registry.register(
        Command(
            id="help.keybindings",
            name="Show Keybindings",
            description="Display all keyboard shortcuts",
            handler=lambda: None,
            category=CommandCategory.HELP,
            icon="",
        )
    )

    # System commands
    registry.register(
        Command(
            id="system.quit",
            name="Quit",
            description="Exit the application",
            handler=lambda: None,
            category=CommandCategory.SYSTEM,
            keybinding="q",
            icon="",
            requires_confirmation=True,
            confirmation_message="Are you sure you want to quit?",
        )
    )

    registry.register(
        Command(
            id="system.command_palette",
            name="Command Palette",
            description="Open the command palette",
            handler=lambda: None,
            category=CommandCategory.SYSTEM,
            keybinding="ctrl+shift+p",
            aliases=["palette", "commands"],
            icon="",
        )
    )

    # Theme commands - Textual built-in themes + custom halext
    themes = [
        ("halext", "Halext (Default)"),
        ("halext-light", "Halext Light"),
        ("textual-dark", "Textual Dark"),
        ("textual-light", "Textual Light"),
        ("nord", "Nord"),
        ("dracula", "Dracula"),
        ("gruvbox", "Gruvbox"),
        ("tokyo-night", "Tokyo Night"),
        ("monokai", "Monokai"),
        ("solarized-light", "Solarized Light"),
    ]

    for theme_id, theme_name in themes:
        registry.register(
            Command(
                id=f"view.theme_{theme_id.replace('-', '_')}",
                name=f"Theme: {theme_name}",
                description=f"Switch to {theme_name} theme",
                handler=lambda: None,
                category=CommandCategory.VIEW,
                icon="",
            )
        )
