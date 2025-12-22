"""Keybinding Cheatsheet Generator.

This module generates keybinding cheatsheets from the BindingRegistry
in various formats (markdown, terminal display, HTML).

Usage:
    cheatsheet = KeybindingCheatsheet()
    markdown = cheatsheet.to_markdown()
    terminal = cheatsheet.to_terminal()
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from tui.core.binding_registry import BindingContext, get_binding_registry
from tui.core.command_registry import CommandCategory, get_command_registry


class KeybindingCheatsheet:
    """Generates keybinding cheatsheets from the registries.

    Provides multiple output formats for displaying available
    keybindings organized by context and category.
    """

    def __init__(self):
        self._bindings = get_binding_registry()
        self._commands = get_command_registry()

    def get_bindings_by_context(self) -> Dict[str, List[Tuple[str, str, str]]]:
        """Get all bindings organized by context.

        Returns:
            Dict mapping context to list of (key, command_id, description)
        """
        result: Dict[str, List[Tuple[str, str, str]]] = {}

        for binding in self._bindings.get_all():
            context = binding.context or "global"
            if context not in result:
                result[context] = []

            # Get command description
            cmd = self._commands.get(binding.command_id)
            description = cmd.description if cmd else binding.command_id

            result[context].append((binding.key, binding.command_id, description))

        # Sort bindings within each context
        for context in result:
            result[context].sort(key=lambda x: x[0])

        return result

    def get_bindings_by_category(self) -> Dict[str, List[Tuple[str, str, str]]]:
        """Get all bindings organized by command category.

        Returns:
            Dict mapping category to list of (key, command_name, description)
        """
        result: Dict[str, List[Tuple[str, str, str]]] = {}

        for cmd in self._commands.list_all():
            if not cmd.keybinding:
                continue

            category = cmd.category.value if isinstance(cmd.category, CommandCategory) else str(cmd.category)
            if category not in result:
                result[category] = []

            result[category].append((cmd.keybinding, cmd.name, cmd.description))

        # Sort by key within each category
        for category in result:
            result[category].sort(key=lambda x: x[0])

        return result

    def get_which_key_bindings(self) -> Dict[str, List[Tuple[str, str]]]:
        """Get which-key (leader) bindings organized by prefix.

        Returns:
            Dict mapping prefix to list of (key, label)
        """
        result: Dict[str, List[Tuple[str, str]]] = {}

        # Get all which-key groups
        for group in self._bindings.get_which_key_groups():
            prefix = group.prefix
            if prefix not in result:
                result[prefix] = []

            for key, label, _ in group.bindings:
                result[prefix].append((key, label))

        return result

    def to_markdown(self) -> str:
        """Generate a markdown cheatsheet.

        Returns:
            Markdown formatted cheatsheet string
        """
        lines = [
            "# HAFS TUI Keybinding Cheatsheet",
            "",
            "## Global Bindings",
            "",
            "| Key | Command | Description |",
            "|-----|---------|-------------|",
        ]

        # Add global bindings
        by_context = self.get_bindings_by_context()
        for key, cmd_id, desc in by_context.get("global", []):
            key_display = self._format_key_markdown(key)
            lines.append(f"| {key_display} | {cmd_id} | {desc} |")

        lines.append("")

        # Add which-key bindings
        lines.extend([
            "## Which-Key (Leader) Bindings",
            "",
            "Press `Space` to activate leader mode, then:",
            "",
        ])

        which_key = self.get_which_key_bindings()
        for prefix, bindings in sorted(which_key.items()):
            prefix_display = prefix.replace("space", "SPC")
            lines.append(f"### {prefix_display}")
            lines.append("")
            lines.append("| Key | Action |")
            lines.append("|-----|--------|")

            for key, label in bindings:
                lines.append(f"| {key} | {label} |")

            lines.append("")

        # Add vim bindings if present
        vim_bindings = by_context.get(BindingContext.VIM_NORMAL, [])
        if vim_bindings:
            lines.extend([
                "## Vim Mode (Normal)",
                "",
                "| Key | Command | Description |",
                "|-----|---------|-------------|",
            ])

            for key, cmd_id, desc in vim_bindings:
                key_display = self._format_key_markdown(key)
                lines.append(f"| {key_display} | {cmd_id} | {desc} |")

            lines.append("")

        # Add category summary
        lines.extend([
            "## Commands by Category",
            "",
        ])

        by_category = self.get_bindings_by_category()
        for category, bindings in sorted(by_category.items()):
            lines.append(f"### {category.title()}")
            lines.append("")

            for key, name, desc in bindings:
                key_display = self._format_key_markdown(key)
                lines.append(f"- `{key_display}` - **{name}**: {desc}")

            lines.append("")

        return "\n".join(lines)

    def to_terminal(self, width: int = 80) -> str:
        """Generate a terminal-friendly cheatsheet.

        Args:
            width: Terminal width for formatting

        Returns:
            ANSI formatted cheatsheet string
        """
        lines = [
            "\033[1;36m╔══════════════════════════════════════════════════════════════════════════════╗\033[0m",
            "\033[1;36m║\033[0m                     \033[1;33mHAFS TUI Keybinding Cheatsheet\033[0m                          \033[1;36m║\033[0m",
            "\033[1;36m╚══════════════════════════════════════════════════════════════════════════════╝\033[0m",
            "",
            "\033[1;32m▶ Global Bindings\033[0m",
            "",
        ]

        by_context = self.get_bindings_by_context()
        for key, cmd_id, desc in by_context.get("global", [])[:10]:
            key_display = self._format_key_terminal(key)
            lines.append(f"  {key_display:12} │ {desc[:50]}")

        lines.extend([
            "",
            "\033[1;32m▶ Which-Key (Space + ...)\033[0m",
            "",
        ])

        which_key = self.get_which_key_bindings()
        for prefix, bindings in list(which_key.items())[:5]:
            prefix_short = prefix.replace("space ", "").replace("space", "")
            if prefix_short:
                lines.append(f"  \033[1;33m{prefix_short}\033[0m:")
                for key, label in bindings[:5]:
                    lines.append(f"    {key:8} → {label}")
            else:
                for key, label in bindings[:5]:
                    lines.append(f"  {key:10} → {label}")

        lines.extend([
            "",
            "\033[1;32m▶ Vim Mode\033[0m",
            "",
            "  h/j/k/l    │ Navigate",
            "  i          │ Insert mode",
            "  v          │ Visual mode",
            "  :          │ Command mode",
            "  /          │ Search",
            "  gg/G       │ Top/Bottom",
            "",
            "\033[2mPress Ctrl+P for command palette, ? for help\033[0m",
        ])

        return "\n".join(lines)

    def to_widget_hints(self) -> List[Tuple[str, str]]:
        """Get bindings formatted for WhichKeyBar hints.

        Returns:
            List of (key, label) tuples
        """
        hints = []

        by_context = self.get_bindings_by_context()
        for key, cmd_id, desc in by_context.get("global", []):
            # Shorten description for hint display
            label = desc[:20] + "..." if len(desc) > 20 else desc
            hints.append((key, label))

        return hints

    def _format_key_markdown(self, key: str) -> str:
        """Format a key for markdown display.

        Args:
            key: Key string

        Returns:
            Markdown formatted key
        """
        # Replace common patterns
        key = key.replace("ctrl+", "Ctrl+")
        key = key.replace("shift+", "Shift+")
        key = key.replace("alt+", "Alt+")
        key = key.replace("space", "Space")
        return f"`{key}`"

    def _format_key_terminal(self, key: str) -> str:
        """Format a key for terminal display with colors.

        Args:
            key: Key string

        Returns:
            ANSI colored key string
        """
        key = key.replace("ctrl+", "\033[1;35mCtrl+\033[0m")
        key = key.replace("shift+", "\033[1;35mShift+\033[0m")
        key = key.replace("alt+", "\033[1;35mAlt+\033[0m")
        return f"\033[1;36m{key}\033[0m"

    def export_to_file(self, path: str, format: str = "markdown") -> None:
        """Export cheatsheet to a file.

        Args:
            path: Output file path
            format: Output format (markdown, terminal)
        """
        if format == "markdown":
            content = self.to_markdown()
        elif format == "terminal":
            content = self.to_terminal()
        else:
            raise ValueError(f"Unknown format: {format}")

        with open(path, "w") as f:
            f.write(content)


def generate_cheatsheet(format: str = "terminal") -> str:
    """Generate a keybinding cheatsheet.

    Args:
        format: Output format (terminal, markdown)

    Returns:
        Formatted cheatsheet string
    """
    cheatsheet = KeybindingCheatsheet()

    if format == "markdown":
        return cheatsheet.to_markdown()
    elif format == "terminal":
        return cheatsheet.to_terminal()
    else:
        return cheatsheet.to_terminal()
