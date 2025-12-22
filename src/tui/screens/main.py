"""Main dashboard screen for HAFS TUI."""

import os
import subprocess
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Static, TabbedContent

from config.loader import load_config
from core.tools import ToolFileSelected
from tui.mixins.vim_navigation import VimNavigationMixin
from tui.mixins.which_key import WhichKeyMixin
from tui.screens.ai_context_modal import AIContextModal
from tui.screens.context_selection_modal import ContextSelectionModal
from tui.screens.file_picker_modal import FilePickerModal
from tui.screens.permissions_modal import PermissionsModal
from tui.utils.file_ops import can_edit_file, rename_path
from tui.widgets.context_viewer import ContextViewer
from tui.widgets.dev_dashboard import DevDashboard
from tui.widgets.explorer import ExplorerWidget
from tui.widgets.header_bar import HeaderBar
from tui.widgets.policy_summary import PolicySummary
from tui.widgets.stats_panel import StatsPanel
from tui.widgets.which_key_bar import WhichKeyBar
from tui.widgets.protocol_widget import ProtocolWidget
from tui.widgets.context_summary import ContextSummaryWidget
from tui.widgets.agent_status import AgentStatusWidget


class MainScreen(Screen, VimNavigationMixin, WhichKeyMixin):
    """Main dashboard screen with project browser and context viewer."""

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("q", "quit", "Quit"),
        Binding("ctrl+p", "command_palette", "Commands"),
        Binding("ctrl+k", "command_palette", "Commands"),
        Binding("a", "add_item", "Add File/Dir"),
        Binding("p", "edit_policies", "Policies"),
        Binding("d", "delete_item", "Delete"),
        Binding("e", "edit_item", "Edit"),
        Binding("ctrl+s", "save_file", "Save"),
        Binding("m", "toggle_markdown_preview", "MD View"),
        Binding("f2", "rename_item", "Rename"),
        Binding("f5", "duplicate_item", "Copy"),
        Binding("ctrl+o", "open_external", "Open OS"),
        Binding("g", "ai_generate", "AI Generate"),
        Binding("c", "context_chat", "Chat w/ Context"),
        Binding("x", "add_to_context", "Add to Context"),
        Binding("w", "add_workspace", "Add Workspace"),
        Binding("F", "edit_fears", "Fears"),
        Binding("[", "shrink_sidebar", "Shrink"),
        Binding("]", "expand_sidebar", "Expand"),
        Binding("ctrl+b", "toggle_sidebar", "Sidebar"),
        # Vim navigation bindings
        *VimNavigationMixin.VIM_BINDINGS,
    ]

    LAYERS = ["base", "overlay"]

    DEFAULT_CSS = """
    MainScreen {
        layout: vertical;
    }

    MainScreen #main-container {
        height: 1fr;
    }

    MainScreen #sidebar {
        width: 32;
        min-width: 0;
        max-width: 60;
        background: $surface;
        border-right: solid $primary;
        height: 100%;
    }

    MainScreen #sidebar.sidebar-collapsed {
        width: 0;
        min-width: 0;
        max-width: 0;
        border-right: none;
        padding: 0;
    }

    MainScreen .sidebar-section {
        height: auto;
        min-height: 3;
        max-height: 50%;
        border-bottom: solid $primary;
    }

    MainScreen .sidebar-section:focus-within {
        border-left: solid $primary;
    }

    MainScreen .hidden {
        display: none;
    }

    MainScreen .section-header {
        height: 1;
        background: $primary;
        padding: 0 1;
    }

    MainScreen .section-header:hover {
        background: $primary;
    }

    MainScreen #workspace-panel {
        height: 1fr;
        min-height: 6;
        border-bottom: solid $primary;
    }

    MainScreen #context-summary {
        height: auto;
        min-height: 6;
        max-height: 12;
        border-bottom: solid $primary;
    }

    MainScreen #agent-status {
        height: auto;
        min-height: 8;
        max-height: 16;
    }

    MainScreen #filesystem-tree {
        height: 100%;
    }

    MainScreen #content {
        background: $background;
        height: 1fr;
    }

    MainScreen #stats-panel {
        height: auto;
    }

    MainScreen #footer-area {
        height: auto;
        background: $surface;
        border-top: solid $primary;
        padding-top: 0;
    }

    MainScreen #footer-grid {
        height: auto;
        width: 100%;
        layout: horizontal;
        align: center middle;
        padding: 0 1;
    }

    MainScreen #which-key-bar {
        width: 2fr;
    }

    MainScreen Footer {
        width: auto;
        min-width: 24;
    }
    """

    _sidebar_width: int = 32
    _sidebar_collapsed: bool = False
    _sidebar_last_width: int = 32

    def compose(self) -> ComposeResult:
        """Compose the screen with consolidated sidebar."""
        yield HeaderBar(id="header-bar")

        with Horizontal(id="main-container"):
            # Sidebar with ExplorerWidget, Context Summary, and Agents
            with Vertical(id="sidebar"):
                config = load_config()
                yield ExplorerWidget(
                    workspace_dirs=config.general.workspace_directories,
                    id="explorer"
                )

                # Context summary panel showing KB stats and recent items
                yield ContextSummaryWidget(id="context-summary")

                # Agent status panel with real-time updates and quick actions
                yield AgentStatusWidget(id="agent-status")

            # Main content area
            with Vertical(id="content"):
                yield DevDashboard(id="dev-dashboard")

        # Footer area with outline
        with Container(id="footer-area"):
            with Horizontal(id="footer-grid"):
                yield WhichKeyBar(id="which-key-bar")
                yield Footer(compact=True, show_command_palette=False)

    def on_mount(self) -> None:
        """Initialize screen on mount."""
        # Initialize vim navigation (loads setting from config)
        self.init_vim_navigation()
        self._consume_pending_open()
        try:
            policies = getattr(self.app, "config", load_config()).afs_directories  # type: ignore[attr-defined]
            self.query_one(PolicySummary).set_policies(policies)
        except Exception:
            pass

    def on_show(self) -> None:
        """Handle screen being shown again (e.g., returning from chat)."""
        self._consume_pending_open()

    def _consume_pending_open(self) -> None:
        """Open any file requested by another screen (best-effort)."""
        path = getattr(self.app, "_pending_open_path", None)
        if not isinstance(path, Path):
            return

        try:
            delattr(self.app, "_pending_open_path")
        except Exception:
            setattr(self.app, "_pending_open_path", None)

        if not path.exists():
            self.notify(f"File not found: {path}", severity="warning")
            return

        try:
            self.query_one("#dev-dashboard", DevDashboard).active = "tab-context"
        except Exception:
            pass

        try:
            viewer = self.query_one("#context-viewer", ContextViewer)
            viewer.set_file(path)
            if can_edit_file(path):
                viewer.enter_edit_mode()
        except Exception:
            self.notify(f"Open failed: {path}", severity="error")

    def on_header_bar_menu_selected(self, event: HeaderBar.MenuSelected) -> None:
        """Handle header bar menu selections."""
        if event.menu_id == "palette":
            self.action_command_palette()
        elif event.menu_id == "context":
            self.action_context_chat()

    def action_command_palette(self) -> None:
        """Open command palette (Ctrl+K)."""
        from tui.screens.command_palette import CommandPalette
        self.app.push_screen(CommandPalette())

    def on_explorer_widget_project_selected(self, event: ExplorerWidget.ProjectSelected) -> None:
        """Handle project selection from tree."""
        # Switch to Context tab
        try:
            self.query_one("#dev-dashboard", DevDashboard).active = "tab-context"
        except Exception:
            pass

        context_viewer = self.query_one("#context-viewer", ContextViewer)
        context_viewer.set_project(event.project)

        # Track project root for protocol helpers (ContextRoot.path is `.context`).
        try:
            setattr(self.app, "_active_protocol_root", event.project.path.parent)
            self.query_one("#protocol-widget", ProtocolWidget).set_target_root(
                event.project.path.parent
            )
        except Exception:
            pass

    def on_explorer_widget_file_selected(self, event: ExplorerWidget.FileSelected) -> None:
        """Handle file selection from tree."""
        # Switch to Context tab
        try:
            self.query_one("#dev-dashboard", DevDashboard).active = "tab-context"
        except Exception:
            pass

        context_viewer = self.query_one("#context-viewer", ContextViewer)
        context_viewer.set_file(event.path)

        # Best-effort: infer protocol root for helper widget.
        try:
            from core.afs.discovery import find_context_root

            context_root = find_context_root(event.path.parent)
            if context_root:
                setattr(self.app, "_active_protocol_root", context_root.parent)
                self.query_one("#protocol-widget", ProtocolWidget).set_target_root(
                    context_root.parent
                )
        except Exception:
            pass

    def on_protocol_widget_open_file_requested(
        self, event: ProtocolWidget.OpenFileRequested
    ) -> None:
        """Open a protocol file requested by the Protocol tab."""
        try:
            self.query_one("#dev-dashboard", DevDashboard).active = "tab-context"
        except Exception:
            pass

        viewer = self.query_one("#context-viewer", ContextViewer)
        viewer.set_file(event.path)

    def on_reports_widget_report_selected(self, event: "ReportsWidget.ReportSelected") -> None:
        """Open a report selected from the Reports tab."""
        try:
            self.query_one("#dev-dashboard", DevDashboard).active = "tab-context"
        except Exception:
            pass

        viewer = self.query_one("#context-viewer", ContextViewer)
        viewer.set_file(event.path)

    def on_tool_file_selected(self, event: ToolFileSelected) -> None:
        """Handle file open request from a tool."""
        # Switch to Context tab
        try:
            self.query_one("#dev-dashboard", DevDashboard).active = "tab-context"
        except Exception:
            pass

        viewer = self.query_one("#context-viewer", ContextViewer)
        viewer.set_file(event.path)

    def action_edit_item(self) -> None:
        """Edit current file (inline when possible, otherwise external editor)."""
        path = self._get_selected_path()
        if not path or not path.exists() or path.is_dir():
            self.notify("Select a file to edit", severity="warning")
            return

        viewer = self.query_one("#context-viewer", ContextViewer)
        viewer.set_file(path)

        if can_edit_file(path):
            if viewer.enter_edit_mode():
                self.notify(f"Editing {path.name} (Ctrl+S to save)", timeout=2)
        else:
            self._edit_file_external(path)

    def _edit_file_external(self, path: Path) -> None:
        """Open file in external editor."""
        editor = os.environ.get("EDITOR", "vim")
        self.app.suspend_process()  # type: ignore[attr-defined]
        subprocess.run([editor, str(path)])
        self.app.resume_process()  # type: ignore[attr-defined]
        self.query_one(ContextViewer).set_file(path)
        self._refresh_explorer()

    def action_save_file(self) -> None:
        """Save inline edits if active."""
        viewer = self.query_one("#context-viewer", ContextViewer)
        if viewer.is_editing:
            viewer.save_edits()
        else:
            self.notify("No inline edit in progress", severity="warning")

    def action_toggle_markdown_preview(self) -> None:
        """Toggle markdown preview/raw view."""
        self.query_one("#context-viewer", ContextViewer).toggle_markdown_preview()

    def action_add_item(self) -> None:
        """Add new file/folder with fuzzy file picker."""
        base_path = self._get_selected_directory()

        def on_file_selected(selected: Path | None) -> None:
            if not selected:
                return

            try:
                if selected.exists():
                    if selected.is_file():
                        self.query_one("#context-viewer", ContextViewer).set_file(selected)
                        self.notify(f"Opened: {selected.name}")
                    else:
                        self.notify(f"Selected directory: {selected.name}")
                else:
                    if selected.name.endswith("/"):
                        selected.mkdir(parents=True, exist_ok=True)
                        msg = f"Created directory: {selected.name.rstrip('/')}"
                    else:
                        selected.parent.mkdir(parents=True, exist_ok=True)
                        selected.touch()
                        msg = f"Created file: {selected.name}"
                    self.notify(msg)
                self._refresh_explorer()
            except Exception as exc:
                self.notify(f"Error: {exc}", severity="error")

        self.app.push_screen(
            FilePickerModal(
                base_path=base_path,
                prompt=f"Add file/directory in {base_path.name}",
                allow_new=True,
            ),
            on_file_selected,
        )

    def action_delete_item(self) -> None:
        """Delete selected file/directory (non-recursive)."""
        path = self._get_selected_path()
        if not path:
            self.notify("Select a file or directory to delete", severity="warning")
            return

        try:
            if path.is_dir():
                path.rmdir()
                msg = f"Deleted empty directory {path.name}"
            else:
                path.unlink()
                msg = f"Deleted {path.name}"
            self.notify(msg)
            self._refresh_explorer()
        except OSError:
            self.notify("Directory not empty. Recursive delete not supported.", severity="error")
        except Exception as exc:
            self.notify(f"Error deleting item: {exc}", severity="error")

    def action_rename_item(self) -> None:
        """Rename the currently selected file/directory."""
        path = self._get_selected_path()
        if not path:
            self.notify("Select a file to rename", severity="warning")
            return

        def on_new_path(selected: Path | None) -> None:
            if not selected:
                return

            if path.is_dir():
                try:
                    rename_path(path, selected)
                    self.notify(f"Renamed to {selected.name}")
                    self._refresh_explorer()
                except Exception as exc:
                    self.notify(f"Rename failed: {exc}", severity="error")
            else:
                viewer = self.query_one("#context-viewer", ContextViewer)
                viewer.set_file(path)
                viewer.rename_current(selected)

        self.app.push_screen(
            FilePickerModal(
                base_path=path.parent,
                prompt=f"Rename {path.name}",
                allow_new=True,
            ),
            on_new_path,
        )

    def action_duplicate_item(self) -> None:
        """Duplicate the selected file."""
        path = self._get_selected_path()
        if not path or not path.is_file():
            self.notify("Select a file to duplicate", severity="warning")
            return

        viewer = self.query_one("#context-viewer", ContextViewer)
        viewer.set_file(path)
        viewer.duplicate_current()

    def action_open_external(self) -> None:
        """Open the selected file externally."""
        path = self._get_selected_path()
        if not path:
            self.notify("Select a file first", severity="warning")
            return

        viewer = self.query_one("#context-viewer", ContextViewer)
        viewer.set_file(path)
        if viewer.open_external():
            self.notify(f"Opening {path.name} externally", timeout=2)

    def action_refresh(self) -> None:
        """Refresh all data."""
        self.query_one(StatsPanel).refresh_data()
        try:
            policies = getattr(self.app, "config", load_config()).afs_directories  # type: ignore[attr-defined]
            self.query_one(PolicySummary).set_policies(policies)
        except Exception:
            pass
        self._refresh_explorer()

        # Refresh sidebar widgets
        try:
            import asyncio
            context_summary = self.query_one("#context-summary", ContextSummaryWidget)
            asyncio.create_task(context_summary.refresh_stats())
        except Exception:
            pass

        try:
            import asyncio
            agent_status = self.query_one("#agent-status", AgentStatusWidget)
            asyncio.create_task(agent_status.refresh_status())
        except Exception:
            pass

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()

    def action_shrink_sidebar(self) -> None:
        """Shrink the sidebar width."""
        if self._sidebar_width <= 15:
            self._sidebar_width = 0
        else:
            self._sidebar_width = max(15, self._sidebar_width - 5)
        self._apply_sidebar_state()

    def action_expand_sidebar(self) -> None:
        """Expand the sidebar width."""
        if self._sidebar_width == 0:
            self._sidebar_width = max(15, self._sidebar_last_width or 32)
        else:
            self._sidebar_width = min(60, self._sidebar_width + 5)
        self._apply_sidebar_state()

    def action_toggle_sidebar(self) -> None:
        """Collapse/expand the entire sidebar."""
        if self._sidebar_width == 0:
            self._sidebar_width = max(15, self._sidebar_last_width or 32)
            self._apply_sidebar_state()
            self.notify("Sidebar expanded", timeout=1)
        else:
            self._sidebar_width = 0
            self._apply_sidebar_state()
            self.notify("Sidebar collapsed", timeout=1)

    def _apply_sidebar_state(self) -> None:
        """Apply sidebar width and collapsed styling."""
        try:
            sidebar = self.query_one("#sidebar", Vertical)
        except Exception:
            return

        if self._sidebar_width <= 0:
            self._sidebar_collapsed = True
            sidebar.add_class("sidebar-collapsed")
            sidebar.styles.width = 0
            return

        self._sidebar_collapsed = False
        self._sidebar_last_width = self._sidebar_width
        sidebar.remove_class("sidebar-collapsed")
        sidebar.styles.width = self._sidebar_width
        self.notify(f"Sidebar width: {self._sidebar_width}", timeout=1)

    def action_ai_generate(self) -> None:
        """Open AI context generation modal."""
        target_path = self._get_selected_directory()
        config = getattr(self.app, "config", None)

        self.app.push_screen(
            AIContextModal(
                target_path=target_path,
                config=config,
                backend="gemini",
            )
        )

    def action_context_chat(self) -> None:
        """Open context selection modal and switch to chat with selected context."""
        def on_context_selected(selected_paths: list[Path] | None) -> None:
            if not selected_paths:
                return

            # Import OrchestratorScreen here to avoid circular imports
            from tui.screens.orchestrator import OrchestratorScreen

            # Create orchestrator screen with coordinator from app
            coordinator = getattr(self.app, "_coordinator", None)
            orch_screen = OrchestratorScreen(
                coordinator=coordinator,
                context_paths=selected_paths,
            )

            # Switch to orchestrator screen
            self.app.push_screen(orch_screen)

            # Show notification about selected context
            count = len(selected_paths)
            self.notify(
                f"Switched to chat with {count} context item{'s' if count != 1 else ''}",
                timeout=3,
            )

            if coordinator:
                coordinator.set_context_items(selected_paths)

        self.app.push_screen(ContextSelectionModal(), on_context_selected)

    def action_add_to_context(self) -> None:
        """Add selected file/directory from filesystem tree to context."""
        path = self._get_selected_path()
        if not path:
            self.notify("Select a file or directory first", severity="warning")
            return

        viewer = self.query_one("#context-viewer", ContextViewer)
        viewer.set_file(path)
        self.notify(f"Added '{path.name}' to context")

    def action_add_workspace(self) -> None:
        """Add a new workspace directory via file picker."""
        def on_dir_selected(selected: Path | None) -> None:
            if not selected or not selected.is_dir():
                return

            # Load current config
            config = load_config()

            # Check if already exists
            existing_paths = [ws.path for ws in config.general.workspace_directories]
            if selected in existing_paths:
                self.notify(f"'{selected.name}' is already in workspace", severity="warning")
                return

            # Add new workspace directory
            from config.schema import WorkspaceDirectory
            config.general.workspace_directories.append(
                WorkspaceDirectory(path=selected, name=selected.name)
            )

            # Save config
            try:
                from config.saver import save_config
                save_config(config)
            except ImportError:
                self.notify("Install tomli-w to persist: pip install tomli-w", severity="warning")

            try:
                explorer = self.query_one(ExplorerWidget)
                explorer.workspace_dirs = config.general.workspace_directories
                if hasattr(explorer, "refresh_data"):
                    explorer.refresh_data()
            except Exception:
                pass

            self.notify(f"Added '{selected.name}' to workspace")

        # Use file picker starting from home
        self.app.push_screen(
            FilePickerModal(
                base_path=Path.home(),
                prompt=f"Select directory to add to workspace",
                allow_new=False,
            ),
            on_dir_selected,
        )

    def action_manage_global_context(self) -> None:
        """Open/edit the global user context (~/.context)."""
        global_root = Path.home() / ".context"
        if not global_root.exists():
            try:
                from core.afs.manager import AFSManager

                manager = AFSManager(load_config())
                manager.init(Path.home(), force=False)
                self.notify("Initialized global context", timeout=2)
            except Exception as exc:
                self.notify(f"Failed to init global context: {exc}", severity="error")
                return

        user_file = global_root / "memory" / "user.md"
        if not user_file.exists():
            user_file.parent.mkdir(parents=True, exist_ok=True)
            user_file.write_text(
                "# User Profile\n\n"
                "## Basics\n"
                "- Name:\n"
                "- Pronouns:\n"
                "- Location / Timezone:\n\n"
                "## Preferences\n"
                "- Coding style:\n"
                "- Tooling:\n"
                "- UI/UX likes/dislikes:\n\n"
                "## Projects & Goals\n"
                "- Current projects:\n"
                "- Long-term goals:\n",
                encoding="utf-8",
            )

        try:
            self.query_one("#dev-dashboard", DevDashboard).active = "tab-context"
        except Exception:
            pass

        viewer = self.query_one("#context-viewer", ContextViewer)
        viewer.set_file(user_file)
        if can_edit_file(user_file):
            viewer.enter_edit_mode()
        self.notify("Editing Global Context user profile", timeout=2)

    def action_edit_policies(self) -> None:
        """Open policy editor from the main dashboard."""
        config = load_config()
        context_path = Path.cwd() / ".context"
        self.app.push_screen(PermissionsModal(config.afs_directories, context_path))

    def action_edit_fears(self) -> None:
        """Open `.context/memory/fears.json` for editing."""
        try:
            from core.afs.manager import AFSManager

            manager = AFSManager(load_config())
            manager.ensure(Path.cwd())
        except Exception as exc:
            self.notify(f"Failed to ensure AFS: {exc}", severity="error")
            return

        fears_file = Path.cwd() / ".context" / "memory" / "fears.json"
        if not fears_file.exists():
            self.notify("fears.json not found in this project", severity="warning")
            return

        try:
            self.query_one("#dev-dashboard", DevDashboard).active = "tab-context"
        except Exception:
            pass

        viewer = self.query_one("#context-viewer", ContextViewer)
        viewer.set_file(fears_file)
        if can_edit_file(fears_file):
            viewer.enter_edit_mode()
        self.notify("Editing fears.json", timeout=2)

    def on_permissions_modal_permissions_updated(
        self, event: PermissionsModal.PermissionsUpdated
    ) -> None:
        """Handle policy updates from the modal."""
        if hasattr(self.app, "config"):
            self.app.config.afs_directories = event.directories  # type: ignore[attr-defined]
        try:
            self.query_one(PolicySummary).set_policies(event.directories)
        except Exception:
            pass
        try:
            self.query_one(StatsPanel).refresh_data()
        except Exception:
            pass

    def on_vim_mode_toggled(self, event: VimNavigationMixin.VimModeToggled) -> None:
        """Surface vim mode status so users discover the bindings."""
        status = "ON" if event.enabled else "OFF"
        self.notify(
            f"Vim mode {status} (j/k/h/l, gg/G, /, : available)",
            timeout=2,
        )

    def on_context_viewer_file_saved(self, event: ContextViewer.FileSaved) -> None:
        """React to inline save events."""
        self.notify(f"Saved {event.path.name}")
        self._refresh_explorer()

    def on_context_viewer_file_renamed(self, event: ContextViewer.FileRenamed) -> None:
        """Refresh explorer when files are renamed."""
        self.notify(f"Renamed {event.old_path.name} → {event.new_path.name}", timeout=2)
        self._refresh_explorer()

    def on_context_viewer_file_duplicated(self, event: ContextViewer.FileDuplicated) -> None:
        """Refresh explorer when files are duplicated."""
        self.notify(f"Copied to {event.new_path.name}", timeout=2)
        self._refresh_explorer()

    def on_context_viewer_file_error(self, event: ContextViewer.FileError) -> None:
        """Surface file operation errors."""
        path = f"{event.path}" if event.path else "file"
        self.notify(f"{path}: {event.error}", severity="error")

    def on_agent_status_widget_chat_requested(
        self, event: AgentStatusWidget.ChatRequested
    ) -> None:
        """Handle chat request from agent status widget."""
        self.action_context_chat()

    def on_agent_status_widget_agent_launch_requested(
        self, event: AgentStatusWidget.AgentLaunchRequested
    ) -> None:
        """Handle agent launch request from agent status widget."""
        if event.agent_type == "swarm":
            # Switch to swarm tab
            try:
                self.query_one("#dev-dashboard", DevDashboard).active = "tab-swarm-control"
                self.notify("Switched to Swarm control", timeout=2)
            except Exception:
                pass
        elif event.agent_type == "embed":
            # Launch embedding job
            self.notify("Starting embedding generation...", timeout=2)
            import subprocess
            subprocess.Popen(
                ["python3.11", "-m", "services.embedding_service", "--quick", "100"],
                start_new_session=True
            )

    def _get_selected_path(self) -> Path | None:
        """Return the current selection from viewer or explorer."""
        viewer = self.query_one("#context-viewer", ContextViewer)
        if viewer.current_path and viewer.current_path.exists():
            return viewer.current_path

        try:
            explorer = self.query_one(ExplorerWidget)
            tabbed = explorer.query_one(TabbedContent)
            active_tab = tabbed.active
            tree_id = "#project-tree" if active_tab == "tab-projects" else "#fs-tree"
            tree = explorer.query_one(tree_id)
            node = tree.cursor_node
        except Exception:
            return None

        if not node or not node.data:
            return None

        data = node.data
        if isinstance(data, Path):
            return data
        if isinstance(data, dict):
            path = data.get("path")
            if isinstance(path, Path):
                return path
        if hasattr(data, "root_path"):
            root_path = getattr(data, "root_path")
            if isinstance(root_path, Path):
                return root_path
        return None

    def _get_selected_directory(self) -> Path:
        """Pick a reasonable directory for creating files."""
        path = self._get_selected_path()
        if path:
            return path if path.is_dir() else path.parent

        config = load_config()
        if config.general.workspace_directories:
            default = config.general.workspace_directories[0].path
            if default.exists():
                return default
        return Path.cwd()

    def _refresh_explorer(self) -> None:
        """Refresh the explorer widget safely."""
        try:
            explorer = self.query_one(ExplorerWidget)
            if hasattr(explorer, "refresh_data"):
                explorer.refresh_data()
        except Exception:
            pass

    # Which-key map (Spacemacs-style leader bindings)
    def get_which_key_map(self):  # type: ignore[override]
        return {
            "f": (
                "+files",
                {
                    "a": ("add", "add_item"),
                    "e": ("edit", "edit_item"),
                    "s": ("save", "save_file"),
                    "r": ("rename", "rename_item"),
                    "d": ("delete", "delete_item"),
                    "c": ("copy", "duplicate_item"),
                    "o": ("open os", "open_external"),
                },
            ),
            "p": (
                "+projects",
                {
                    "w": ("add workspace", "add_workspace"),
                    "u": ("edit global user context", "manage_global_context"),
                    "t": ("toggle sidebar", "toggle_sidebar"),
                    "[": ("shrink sidebar", "shrink_sidebar"),
                    "]": ("expand sidebar", "expand_sidebar"),
                },
            ),
            "s": (
                "+search",
                {
                    "c": ("command palette", "command_palette"),
                },
            ),
            "c": (
                "+context/chat",
                {
                    "c": ("chat with context", "context_chat"),
                    "x": ("add selection to context", "add_to_context"),
                    "f": ("edit fears.json", "edit_fears"),
                },
            ),
            "g": ("ai generate", "ai_generate"),
            "l": ("logs", lambda: self.app.action_switch_logs()),
            "q": ("quit", "quit"),
        }

    async def on_header_bar_navigation_requested(self, event: HeaderBar.NavigationRequested) -> None:
        """Handle header bar navigation requests."""
        from tui.core.screen_router import get_screen_router

        route_map = {
            "dashboard": "/dashboard",
            "chat": "/chat",
            "logs": "/logs",
            "services": "/services",
            "analysis": "/analysis",
            "config": "/config",
        }
        route = route_map.get(event.screen)
        if route:
            router = get_screen_router()
            await router.navigate(route)
