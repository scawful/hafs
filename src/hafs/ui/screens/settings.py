"""Settings screen for HAFS TUI."""

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, Static

from hafs.config.loader import load_config
from hafs.config.schema import PolicyType
from hafs.ui.mixins.which_key import WhichKeyMixin
from hafs.ui.screens.permissions_modal import PermissionsModal
from hafs.ui.widgets.which_key_bar import WhichKeyBar


class SettingsScreen(Screen, WhichKeyMixin):
    """Settings and rules editor screen."""

    BINDINGS = [
        ("q", "back", "Back"),
        ("r", "reload", "Reload Config"),
        ("p", "edit_policies", "Edit Policies"),
    ]

    DEFAULT_CSS = """
    SettingsScreen #footer-area {
        height: auto;
        background: $surface;
    }

    SettingsScreen #footer-grid {
        height: auto;
        width: 100%;
        layout: horizontal;
        align: center middle;
        padding: 0 1;
    }

    SettingsScreen #which-key-bar {
        width: 2fr;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the screen."""
        yield Header()

        config = load_config()

        with VerticalScroll(id="settings-container"):
            # General settings
            yield Static("[bold purple]GENERAL SETTINGS[/bold purple]")
            with Container(classes="settings-section"):
                yield Label(f"Refresh Interval: {config.general.refresh_interval}s")
                yield Label(f"Default Editor: {config.general.default_editor}")
                yield Label(f"Show Hidden Files: {config.general.show_hidden_files}")

            yield Static("")  # Spacer

            # Theme settings
            yield Static("[bold purple]THEME[/bold purple]")
            with Container(classes="settings-section"):
                yield Label(f"Primary: [{config.theme.primary}]█████[/]")
                yield Label(f"Secondary: [{config.theme.secondary}]█████[/]")
                yield Label(f"Accent: [{config.theme.accent}]█████[/]")

            yield Static("")  # Spacer

            # AFS Directory Policies
            yield Static(
                "[bold purple]AFS DIRECTORY POLICIES[/bold purple] "
                "[dim](Press [bold]p[/bold] to edit)[/dim]"
            )
            with Container(classes="settings-section"):
                for dir_config in config.afs_directories:
                    color = self._get_policy_color(dir_config.policy)
                    yield Label(
                        f"[{color}]●[/{color}] {dir_config.name}: "
                        f"[{color}]{dir_config.policy.value}[/{color}] "
                        f"[dim]- {dir_config.description}[/dim]"
                    )

            yield Static("")  # Spacer

            # Parser settings
            yield Static("[bold purple]PARSER SETTINGS[/bold purple]")
            with Container(classes="settings-section"):
                yield Label(
                    f"Gemini: {'✓' if config.parsers.gemini.enabled else '✗'} "
                    f"[dim]{config.parsers.gemini.base_path or '~/.gemini/tmp'}[/dim]"
                )
                yield Label(
                    f"Claude: {'✓' if config.parsers.claude.enabled else '✗'} "
                    f"[dim]{config.parsers.claude.base_path or '~/.claude/plans'}[/dim]"
                )
                antigravity_path = (
                    config.parsers.antigravity.base_path or "~/.gemini/antigravity/brain"
                )
                yield Label(
                    f"Antigravity: {'✓' if config.parsers.antigravity.enabled else '✗'} "
                    f"[dim]{antigravity_path}[/dim]"
                )

            yield Static("")  # Spacer

            # Tracked projects
            yield Static("[bold purple]TRACKED PROJECTS[/bold purple]")
            with Container(classes="settings-section"):
                if config.tracked_projects:
                    for project in config.tracked_projects:
                        yield Label(f"  • {project}")
                else:
                    yield Label("[dim]No tracked projects configured[/dim]")

            yield Static("")  # Spacer

            # Config file location
            yield Static("[bold purple]CONFIGURATION[/bold purple]")
            with Container(classes="settings-section"):
                yield Label("[dim]Config files are loaded from:[/dim]")
                yield Label("  • ./hafs.toml (project-local)")
                yield Label("  • ~/.config/hafs/config.toml (user)")
                yield Static("")
                yield Label("[dim]Edit these files to customize HAFS behavior.[/dim]")

        # Footer area with outline
        with Container(id="footer-area"):
            with Horizontal(id="footer-grid"):
                yield WhichKeyBar(id="which-key-bar")
                yield Footer()

    def on_mount(self) -> None:
        """Initialize screen on mount."""
        self.title = "HAFS - Settings"

    def action_back(self) -> None:
        """Go back to main screen."""
        self.app.pop_screen()

    def action_reload(self) -> None:
        """Reload configuration and refresh display."""
        self.refresh(recompose=True)
        self.notify("Configuration reloaded")

    def action_edit_policies(self) -> None:
        """Open policy editor modal."""
        config = load_config()
        # Check for .context in current directory
        context_path = Path.cwd() / ".context"
        self.app.push_screen(
            PermissionsModal(config.afs_directories, context_path),
        )

    def on_permissions_modal_permissions_updated(
        self, event: PermissionsModal.PermissionsUpdated
    ) -> None:
        """Handle permission updates from the modal."""
        # Update app config in-memory
        if hasattr(self.app, "config"):
            self.app.config.afs_directories = event.directories  # type: ignore[attr-defined]
        # Refresh display
        self.refresh(recompose=True)

    @staticmethod
    def _get_policy_color(policy: PolicyType) -> str:
        """Get color for policy type."""
        return {
            PolicyType.READ_ONLY: "blue",
            PolicyType.WRITABLE: "green",
            PolicyType.EXECUTABLE: "red",
        }.get(policy, "white")

    def get_which_key_map(self):  # type: ignore[override]
        return {
            "r": ("reload config", "reload"),
            "p": ("edit policies", "edit_policies"),
            "q": ("back", "back"),
        }
