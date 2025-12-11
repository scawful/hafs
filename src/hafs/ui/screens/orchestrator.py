"""Multi-agent orchestration screen."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Footer, LoadingIndicator, Static

from hafs.ui.mixins.vim_navigation import VimNavigationMixin
from hafs.ui.screens.permissions_modal import PermissionsModal
from hafs.ui.widgets.chat_input import ChatInput
from hafs.ui.widgets.context_panel import ContextPanel
from hafs.ui.widgets.header_bar import HeaderBar
from hafs.ui.widgets.keybinding_bar import (
    ORCHESTRATOR_SCREEN_BINDINGS_ROW1,
    ORCHESTRATOR_SCREEN_BINDINGS_ROW2,
    KeyBindingBar,
)
from hafs.ui.widgets.lane_container import LaneContainer
from hafs.ui.widgets.mode_toggle import ModeToggle
from hafs.ui.widgets.synergy_panel import SynergyPanel

if TYPE_CHECKING:
    from hafs.agents.coordinator import AgentCoordinator


class ViewMode(Enum):
    """View mode for agent lanes."""

    FOCUS = "focus"  # Single agent, full terminal
    MULTI = "multi"  # All agents, side-by-side (truncated when 3+)


class OrchestratorScreen(Screen, VimNavigationMixin):
    """Multi-agent orchestration screen with lanes.

    Layout:
    ┌─────────────────────────────────────────────────────────────┐
    │ Header                                                       │
    ├───────────────┬───────────────┬───────────────┬─────────────┤
    │ Agent 1       │ Agent 2       │ Agent 3       │ Context     │
    │ (Planner)     │ (Coder)       │ (Critic)      │ Panel       │
    │               │               │               │             │
    │ Terminal      │ Terminal      │ Terminal      │ Shared      │
    │ Output        │ Output        │ Output        │ Memory      │
    ├───────────────┴───────────────┴───────────────┴─────────────┤
    │ Synergy Panel (ToM metrics)                                  │
    ├─────────────────────────────────────────────────────────────┤
    │ Chat Input with @mention autocomplete                        │
    ├─────────────────────────────────────────────────────────────┤
    │ Footer                                                       │
    └─────────────────────────────────────────────────────────────┘

    Example:
        coordinator = AgentCoordinator(config)
        screen = OrchestratorScreen(coordinator)
        app.push_screen(screen)
    """

    LAYERS = ["base", "overlay", "_toastrack"]

    BINDINGS = [
        # Lane focus shortcuts (number keys for quick-switch)
        Binding("1", "focus_lane_1", "Lane 1", show=False, priority=False),
        Binding("2", "focus_lane_2", "Lane 2", show=False, priority=False),
        Binding("3", "focus_lane_3", "Lane 3", show=False, priority=False),
        Binding("4", "focus_lane_4", "Lane 4", show=False, priority=False),
        Binding("tab", "next_lane", "Next Lane", show=True),
        Binding("m", "toggle_view_mode", "View Mode", show=True, priority=False),
        # Agent management
        Binding("ctrl+n", "new_agent", "New Agent", show=True),
        Binding("ctrl+k", "kill_agent", "Kill Agent", show=False),
        # UI toggles
        Binding("ctrl+x", "toggle_context", "Context", show=True),
        Binding("ctrl+p", "manage_permissions", "Permissions", show=True),
        Binding("ctrl+s", "toggle_synergy", "Synergy", show=False),
        Binding("ctrl+l", "clear_current", "Clear", show=False),
        # Gemini-CLI special keys (forwarded to focused agent)
        Binding("ctrl+c", "send_interrupt", "Interrupt", show=False),
        Binding("ctrl+y", "send_yolo", "YOLO", show=False),
        Binding("shift+tab", "send_accept_edits", "Accept", show=False),
        # Navigation
        Binding("escape", "back", "Back", show=True),
        # Vim navigation bindings
        *VimNavigationMixin.VIM_BINDINGS,
    ]

    DEFAULT_CSS = """
    OrchestratorScreen {
        background: $surface;
    }

    OrchestratorScreen #main-area {
        width: 100%;
        height: 1fr;
    }

    OrchestratorScreen #lanes-area {
        width: 1fr;
        height: 100%;
    }

    OrchestratorScreen .hidden {
        display: none;
    }

    OrchestratorScreen #status-bar-container {
        height: 1;
        width: 100%;
        background: $primary-darken-2;
    }

    OrchestratorScreen #status-bar {
        width: 1fr;
        height: 1;
        background: $primary-darken-2;
        color: $text;
        padding: 0 1;
    }

    OrchestratorScreen #mode-toggle {
        width: auto;
        height: 1;
    }

    #loading-overlay {
        layer: overlay;
        height: 100%;
        width: 100%;
        align: center middle;
        background: $surface 50%;
    }

    OrchestratorScreen #footer-area {
        height: auto;
        background: $surface;
    }

    OrchestratorScreen #footer-grid {
        height: auto;
        width: 100%;
        layout: horizontal;
        align: center middle;
        padding: 0 1;
    }

    OrchestratorScreen #keybinding-bar {
        width: 2fr;
    }
    """

    def __init__(
        self,
        coordinator: "AgentCoordinator | None" = None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        context_paths: list[Path] | None = None,
    ):
        """Initialize orchestrator screen.

        Args:
            coordinator: AgentCoordinator for managing agents.
            name: Screen name.
            id: Screen ID.
            classes: CSS classes.
            context_paths: Optional list of context paths selected before chat.
        """
        super().__init__(name=name, id=id, classes=classes)
        self._coordinator = coordinator
        self._context_visible = True
        self._synergy_visible = True
        self._focused_lane_index = 0
        self._view_mode = ViewMode.FOCUS
        self._focused_lane_id: str | None = None
        self._pending_context_paths: list[Path] = list(context_paths or [])

    def compose(self) -> ComposeResult:
        """Compose the screen layout."""
        yield HeaderBar(id="header-bar")

        # Status bar with mode toggle
        with Horizontal(id="status-bar-container"):
            yield ModeToggle(id="mode-toggle")
            yield Static(
                "Press [bold]^N[/] add agent | [bold]@name[/] mention | [bold]/help[/] commands",
                id="status-bar",
            )

        # Main content area
        with Horizontal(id="main-area"):
            # Lanes container
            yield LaneContainer(id="lanes")

            # Context panel (right sidebar)
            yield ContextPanel(id="context-panel")

        # Synergy panel (bottom)
        yield SynergyPanel(id="synergy-panel")

        # Chat input
        yield ChatInput(
            agent_names=self._get_agent_names(),
            placeholder="Message agents... (@name to mention, /help for commands)",
            id="chat-input",
        )

        # Loading indicator (hidden by default if coordinator exists)
        if not self._coordinator:
            yield LoadingIndicator(id="loading-overlay")

        # Footer area with outline
        with Container(id="footer-area"):
            with Horizontal(id="footer-grid"):
                yield KeyBindingBar(
                    row1=ORCHESTRATOR_SCREEN_BINDINGS_ROW1,
                    row2=ORCHESTRATOR_SCREEN_BINDINGS_ROW2,
                    id="keybinding-bar",
                )
                yield Footer()

    async def on_mount(self) -> None:
        """Handle screen mount."""
        # Initialize vim navigation (loads setting from config)
        self.init_vim_navigation()

        # Focus the chat input
        self.query_one("#chat-input", ChatInput).focus_input()

        # Initialize mode toggle with current coordinator mode
        if self._coordinator:
            mode_toggle = self.query_one("#mode-toggle", ModeToggle)
            mode_toggle.set_mode(self._coordinator.mode)

        # Show configured policies in the context panel
        try:
            context_panel = self.query_one("#context-panel", ContextPanel)
            if hasattr(self.app, "config"):
                context_panel.update_policies(getattr(self.app, "config").afs_directories)  # type: ignore[attr-defined]
        except Exception:
            pass

        # Apply any pre-selected context paths to shared context/panel
        self._apply_context_paths()

        # Initialize coordinator if not provided
        if not self._coordinator:
            self._update_status("Initializing agents...")
            # Start initialization in background
            self.run_worker(self._init_coordinator())
        else:
            # Initialize with default agents if coordinator provided
            await self._setup_default_agents()

    def _update_status(self, message: str) -> None:
        """Update status bar with message."""
        try:
            status_bar = self.query_one("#status-bar", Static)
            status_bar.update(f"[bold]Chat[/] | {message}")
        except Exception:
            pass

    async def _init_coordinator(self) -> None:
        """Initialize the coordinator and agents in background."""
        import asyncio

        try:
            # Ensure backends are registered
            import hafs.backends  # noqa: F401
            from hafs.agents.coordinator import AgentCoordinator
            from hafs.models.agent import AgentRole

            # Get config from app
            config = getattr(self.app, "config", None)
            if not config:
                from hafs.config.loader import load_config
                config = load_config()

            self._update_status("Creating coordinator...")

            # Initialize coordinator with timeout
            try:
                self._coordinator = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, lambda: AgentCoordinator(config)
                    ),
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                self.notify("Agent coordinator initialization timed out", severity="error")
                self._update_status("Initialization timed out")
                return

            # Default agents to create
            agents_to_init = [
                {"name": "Planner", "role": "planner"},
                {"name": "Coder", "role": "coder"},
                {"name": "Critic", "role": "critic"},
            ]

            successful_agents = 0
            failed_agents = []
            default_backend = getattr(self.app, "_default_backend", "gemini")

            for i, agent_spec in enumerate(agents_to_init, 1):
                try:
                    self._update_status(
                        f"Registering agent {i}/{len(agents_to_init)}: "
                        f"{agent_spec['name']}..."
                    )
                    role = AgentRole(agent_spec.get("role", "general"))
                    if self._coordinator:
                        await self._coordinator.register_agent(
                            name=agent_spec["name"],
                            role=role,
                            backend_name=default_backend,
                        )
                        successful_agents += 1
                except Exception as e:
                    failed_agents.append(agent_spec["name"])
                    self.notify(
                        f"Failed to register agent {agent_spec['name']}: {e}",
                        severity="warning",
                    )

            # Remove loading indicator
            try:
                loading = self.query_one("#loading-overlay")
                loading.remove()
            except Exception:
                pass

            # Set up UI with agents
            if self._coordinator:
                await self._setup_default_agents()
                self._apply_context_paths()

                # Show summary
                if failed_agents:
                    self.notify(
                        f"Initialized {successful_agents} agents. "
                        f"Failed: {', '.join(failed_agents)}",
                        severity="warning",
                        timeout=5,
                    )
                else:
                    self.notify(f"All {successful_agents} agents ready", timeout=3)

                self._update_status(
                    "Press [bold]Ctrl+N[/] to add agent | [bold]@name[/] to mention"
                )

        except ImportError as e:
            self.notify(f"Failed to load agent modules: {e}", severity="error")
            self._update_status(f"Error: {e}")
        except Exception as e:
            self.notify(f"Chat initialization failed: {e}", severity="error")
            self._update_status(f"Error: {e}")

    async def set_coordinator(self, coordinator: "AgentCoordinator") -> None:
        """Set the coordinator and initialize agents.

        Args:
            coordinator: The initialized AgentCoordinator.
        """
        self._coordinator = coordinator

        # Update status to show we're setting up agents
        total_agents = len(coordinator.agents) if coordinator.agents else 0
        self._update_status(f"Setting up {total_agents} agents...")

        # Remove loading indicator
        try:
            loading = self.query_one("#loading-overlay")
            loading.remove()
        except Exception:
            pass

        await self._setup_default_agents()
        self._apply_context_paths()

        # Update status to ready state
        self._update_status(
            "Press [bold]Ctrl+N[/] to add agent | [bold]@name[/] to mention"
        )

    async def _setup_default_agents(self) -> None:
        """Set up default agents from coordinator."""
        if not self._coordinator:
            return

        lanes = self.query_one("#lanes", LaneContainer)

        # Add lanes first (UI update)
        for name, lane in self._coordinator.agents.items():
            await lanes.add_lane(lane, f"lane-{name.lower()}")

        # Update chat input with agent names
        self._update_agent_names()

        # Update context panel
        context_panel = self.query_one("#context-panel", ContextPanel)
        context_panel.update_context(self._coordinator.shared_context)

        # Set initial focus to first lane (focus view mode is default)
        lane_ids = lanes.lane_ids
        if lane_ids:
            self._focused_lane_id = lane_ids[0]
            self._focused_lane_index = 0
            lanes.set_focused_lane(self._focused_lane_id)
            self._update_focused_status()

        # Start agents in background to avoid blocking UI
        self.run_worker(self._start_agents_background())

    async def _start_agents_background(self) -> None:
        """Start all agents in background."""
        if self._coordinator:
            # Parallel startup
            await self._coordinator.start_all_agents()
            self.notify("All agents started")

    def _get_agent_names(self) -> list[str]:
        """Get list of agent names for autocomplete."""
        if self._coordinator:
            return list(self._coordinator.agents.keys())
        return []

    def _update_agent_names(self) -> None:
        """Update chat input with current agent names."""
        chat_input = self.query_one("#chat-input", ChatInput)
        chat_input.set_agent_names(self._get_agent_names())

    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        """Handle chat input submission.

        Args:
            event: The submission event.
        """
        message = event.value

        # Handle commands
        if message.startswith("/"):
            await self._handle_command(message)
            return

        # Route message through coordinator
        if self._coordinator:
            target = await self._coordinator.route_message(message, sender="user")

            # Focus the target lane if found
            if target:
                lanes = self.query_one("#lanes", LaneContainer)
                lane_widget = lanes.get_lane_by_agent_name(target)

                # IMPORTANT: Set up streaming BEFORE processing message
                # so we capture all output including the echoed input
                if lane_widget:
                    await lane_widget.start_streaming()
                    lane_widget.focus()
                    # Focus the terminal inside the lane widget for keyboard input
                    try:
                        from hafs.ui.widgets.terminal_emulator import TerminalDisplay
                        terminal = lane_widget.query_one("#terminal", TerminalDisplay)
                        terminal.focus()
                    except Exception:
                        pass

                # Now process the message (sends to backend)
                lane = self._coordinator.get_lane(target)
                if lane and not lane.is_busy:
                    await lane.process_next_message()

    async def _handle_command(self, command: str) -> None:
        """Handle slash commands.

        Args:
            command: The command string.
        """
        parts = command[1:].split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "help":
            self.notify(
                "Commands:\n"
                "  /add <name> <role> - Add new agent\n"
                "  /remove <name> - Remove agent\n"
                "  /list - List agents\n"
                "  /task <description> - Set active task\n"
                "  /clear - Clear current lane\n"
                "  /broadcast <msg> - Message all agents\n"
                "  /mode <planning|execution> - Set coordinator mode",
                title="Help",
                timeout=10,
            )
        elif cmd == "add" and args:
            await self._add_agent_command(args)
        elif cmd == "remove" and args:
            await self._remove_agent_command(args)
        elif cmd == "list":
            await self._list_agents_command()
        elif cmd == "task" and args:
            await self._set_task_command(args)
        elif cmd == "clear":
            self._clear_current_lane()
        elif cmd == "broadcast" and args:
            await self._broadcast_command(args)
        elif cmd == "mode":
            await self._mode_command(args)
        else:
            self.notify(f"Unknown command: {cmd}", severity="error")

    async def _add_agent_command(self, args: str) -> None:
        """Handle /add command."""
        parts = args.split(maxsplit=1)
        if len(parts) < 2:
            self.notify("Usage: /add <name> <role>", severity="error")
            return

        name, role_str = parts
        role_str = role_str.upper()

        # Validate role
        try:
            from hafs.models.agent import AgentRole

            role = AgentRole(role_str.lower())
        except ValueError:
            valid_roles = ", ".join(r.value for r in AgentRole)
            self.notify(f"Invalid role. Valid: {valid_roles}", severity="error")
            return

        if self._coordinator:
            try:
                lane = await self._coordinator.register_agent(
                    name=name,
                    role=role,
                    backend_name="gemini",  # Default backend
                )
                lanes = self.query_one("#lanes", LaneContainer)
                await lanes.add_lane(lane, f"lane-{name.lower()}")
                self._update_agent_names()
                self.notify(f"Added agent: {name} ({role.value})")
            except Exception as e:
                self.notify(f"Failed to add agent: {e}", severity="error")

    async def _remove_agent_command(self, name: str) -> None:
        """Handle /remove command."""
        if self._coordinator:
            try:
                await self._coordinator.unregister_agent(name)
                lanes = self.query_one("#lanes", LaneContainer)
                await lanes.remove_lane(f"lane-{name.lower()}")
                self._update_agent_names()
                self.notify(f"Removed agent: {name}")
            except Exception as e:
                self.notify(f"Failed to remove agent: {e}", severity="error")

    async def _list_agents_command(self) -> None:
        """Handle /list command."""
        if self._coordinator:
            agents = self._coordinator.agents
            if agents:
                lines = [f"  {name}: {lane.agent.role.value}" for name, lane in agents.items()]
                self.notify("Agents:\n" + "\n".join(lines), title="Agent List", timeout=5)
            else:
                self.notify("No agents registered", severity="warning")

    async def _set_task_command(self, task: str) -> None:
        """Handle /task command."""
        if self._coordinator:
            self._coordinator.update_shared_context(task=task)
            context_panel = self.query_one("#context-panel", ContextPanel)
            context_panel.update_context(self._coordinator.shared_context)
            self.notify(f"Task set: {task}")

    async def _broadcast_command(self, message: str) -> None:
        """Handle /broadcast command."""
        if self._coordinator:
            await self._coordinator.broadcast(message, sender="user")
            self.notify("Broadcast sent to all agents")

    async def _mode_command(self, mode_arg: str) -> None:
        """Handle /mode command.

        Args:
            mode_arg: Mode argument ("planning" or "execution").
        """
        if not self._coordinator:
            self.notify("No coordinator available", severity="error")
            return

        from hafs.agents.coordinator import CoordinatorMode

        mode_str = mode_arg.strip().lower()

        # Validate mode
        if mode_str not in ["planning", "execution"]:
            self.notify(
                "Invalid mode. Use: /mode planning OR /mode execution",
                severity="error",
            )
            return

        # Convert to enum
        mode = CoordinatorMode(mode_str)

        # Update coordinator
        await self._coordinator.set_mode(mode)

        # Update the mode toggle widget
        try:
            mode_toggle = self.query_one("#mode-toggle", ModeToggle)
            mode_toggle.set_mode(mode)
        except Exception:
            pass

        # Notify user
        mode_name = mode.value.upper()
        self.notify(f"Mode changed to {mode_name}", title="Mode Changed")

    def _clear_current_lane(self) -> None:
        """Clear the currently focused lane."""
        lanes = self.query_one("#lanes", LaneContainer)
        lane_ids = lanes.lane_ids
        if lane_ids and self._focused_lane_index < len(lane_ids):
            lane = lanes.get_lane(lane_ids[self._focused_lane_index])
            if lane:
                lane.clear_terminal()

    def action_focus_lane_1(self) -> None:
        """Focus first lane."""
        if self._is_input_focused():
            return
        self._focus_lane(0)

    def action_focus_lane_2(self) -> None:
        """Focus second lane."""
        if self._is_input_focused():
            return
        self._focus_lane(1)

    def action_focus_lane_3(self) -> None:
        """Focus third lane."""
        if self._is_input_focused():
            return
        self._focus_lane(2)

    def action_focus_lane_4(self) -> None:
        """Focus fourth lane."""
        if self._is_input_focused():
            return
        self._focus_lane(3)

    def action_next_lane(self) -> None:
        """Cycle to next lane."""
        lanes = self.query_one("#lanes", LaneContainer)
        lane_ids = lanes.lane_ids
        if lane_ids:
            next_index = (self._focused_lane_index + 1) % len(lane_ids)
            self._focus_lane(next_index)

    def action_toggle_view_mode(self) -> None:
        """Toggle between focus and multi-view modes."""
        if self._is_input_focused():
            return
        if self._view_mode == ViewMode.FOCUS:
            self._view_mode = ViewMode.MULTI
            self._set_multi_view()
            self.notify("Multi-view mode", timeout=2)
        else:
            self._view_mode = ViewMode.FOCUS
            self._set_focus_view()
            self.notify("Focus mode", timeout=2)

    def _focus_lane(self, index: int) -> None:
        """Focus a lane by index."""
        lanes = self.query_one("#lanes", LaneContainer)
        lane_ids = lanes.lane_ids
        if index < len(lane_ids):
            self._focused_lane_index = index
            self._focused_lane_id = lane_ids[index]
            lane = lanes.get_lane(lane_ids[index])
            if lane:
                lane.focus()
            # In focus mode, show only the focused lane
            if self._view_mode == ViewMode.FOCUS:
                lanes.set_focused_lane(self._focused_lane_id)
            # Update status bar
            self._update_focused_status()

    def _set_focus_view(self) -> None:
        """Set the view to focus mode (single lane visible)."""
        lanes = self.query_one("#lanes", LaneContainer)
        lanes.set_view_mode("focus")
        if self._focused_lane_id:
            lanes.set_focused_lane(self._focused_lane_id)

    def _set_multi_view(self) -> None:
        """Set the view to multi mode (all lanes visible)."""
        lanes = self.query_one("#lanes", LaneContainer)
        lanes.set_view_mode("multi")

    def _update_focused_status(self) -> None:
        """Update status bar with focused agent info."""
        if self._focused_lane_id and self._coordinator:
            # Get agent name from lane ID
            agent_name = self._focused_lane_id.replace("lane-", "").title()
            mode_str = "Focus" if self._view_mode == ViewMode.FOCUS else "Multi"
            self._update_status(
                f"[{mode_str}] Agent: [bold]{agent_name}[/] | "
                f"[dim]Tab[/] next | [dim]m[/] toggle view | [dim]Ctrl+Y[/] YOLO"
            )

    def action_new_agent(self) -> None:
        """Prompt to add new agent."""
        self.notify(
            "Use /add <name> <role> to add an agent\n"
            "Roles: general, planner, coder, critic, researcher",
            title="Add Agent",
            timeout=5,
        )

    def action_kill_agent(self) -> None:
        """Prompt to kill agent."""
        self.notify(
            "Use /remove <name> to remove an agent",
            title="Remove Agent",
            timeout=3,
        )

    def action_manage_permissions(self) -> None:
        """Open permissions management modal."""
        directories = getattr(self.app, "config", None)
        directory_configs = directories.afs_directories if directories else []
        # Check for .context in current directory
        context_path = Path.cwd() / ".context"
        self.app.push_screen(PermissionsModal(directory_configs, context_path))

    def action_toggle_context(self) -> None:
        """Toggle context panel visibility."""
        context_panel = self.query_one("#context-panel", ContextPanel)
        self._context_visible = not self._context_visible
        if self._context_visible:
            context_panel.remove_class("hidden")
        else:
            context_panel.add_class("hidden")

    def action_toggle_synergy(self) -> None:
        """Toggle synergy panel visibility."""
        synergy_panel = self.query_one("#synergy-panel", SynergyPanel)
        self._synergy_visible = not self._synergy_visible
        if self._synergy_visible:
            synergy_panel.remove_class("hidden")
        else:
            synergy_panel.add_class("hidden")

    def action_clear_current(self) -> None:
        """Clear current lane terminal."""
        self._clear_current_lane()

    def action_back(self) -> None:
        """Go back to previous screen or main dashboard."""
        # Check if there's a screen to go back to
        if len(self.app.screen_stack) > 1:
            self.app.pop_screen()
        else:
            # No previous screen (e.g., started with hafs chat), go to main
            from hafs.ui.screens.main import MainScreen
            self.app.switch_screen(MainScreen())

    def update_synergy_score(self, score) -> None:
        """Update the synergy panel with new score.

        Args:
            score: SynergyScore to display.
        """
        synergy_panel = self.query_one("#synergy-panel", SynergyPanel)
        synergy_panel.update_score(score)

    def add_finding(self, finding: str) -> None:
        """Add a finding to the shared context.

        Args:
            finding: Finding text.
        """
        if self._coordinator:
            self._coordinator.update_shared_context(finding=finding)
            context_panel = self.query_one("#context-panel", ContextPanel)
            context_panel.update_context(self._coordinator.shared_context)

    def add_decision(self, decision: str) -> None:
        """Add a decision to the shared context.

        Args:
            decision: Decision text.
        """
        if self._coordinator:
            self._coordinator.update_shared_context(decision=decision)
            context_panel = self.query_one("#context-panel", ContextPanel)
            context_panel.update_context(self._coordinator.shared_context)

    def on_permissions_modal_permissions_updated(  # type: ignore[override]
        self, event: PermissionsModal.PermissionsUpdated
    ) -> None:
        """Handle permission updates from the modal."""
        # Update app config in-memory
        if hasattr(self.app, "config"):
            self.app.config.afs_directories = event.directories  # type: ignore[attr-defined]

        # Refresh context panel display
        try:
            context_panel = self.query_one("#context-panel", ContextPanel)
            context_panel.update_policies(event.directories)
        except Exception:
            pass

    async def on_mode_toggle_mode_changed(self, event: ModeToggle.ModeChanged) -> None:
        """Handle mode changes from the ModeToggle widget.

        Args:
            event: The mode change event.
        """
        if self._coordinator:
            from hafs.agents.coordinator import CoordinatorMode

            # Convert string to enum
            mode = CoordinatorMode(event.mode)

            # Update coordinator mode
            await self._coordinator.set_mode(mode)

            # Notify user
            mode_name = mode.value.upper()
            self.notify(f"Mode changed to {mode_name}", title="Mode Changed")

    # Gemini-CLI special key forwarding actions

    async def action_send_interrupt(self) -> None:
        """Send Ctrl+C (interrupt) to focused agent."""
        await self._send_key_to_focused_lane("ctrl+c")

    async def action_send_yolo(self) -> None:
        """Send Ctrl+Y (YOLO mode) to focused agent for Gemini-CLI."""
        await self._send_key_to_focused_lane("ctrl+y")
        self.notify("YOLO mode sent", timeout=1)

    async def action_send_accept_edits(self) -> None:
        """Send Shift+Tab (accept edits) to focused agent for Gemini-CLI."""
        await self._send_key_to_focused_lane("shift+tab")
        self.notify("Accept edits sent", timeout=1)

    async def _send_key_to_focused_lane(self, key: str) -> None:
        """Send a special key to the currently focused agent lane.

        Args:
            key: Key name to send (e.g., "ctrl+c", "ctrl+y", "shift+tab").
        """
        if not self._focused_lane_id or not self._coordinator:
            return

        # Get the AgentLane from the coordinator
        agent_name = self._focused_lane_id.replace("lane-", "")
        lane = self._coordinator.get_lane(agent_name)
        if lane:
            await lane.send_key(key)

    def _apply_context_paths(self) -> None:
        """Apply any pending context paths to shared context and UI."""
        if not self._pending_context_paths:
            return

        if self._coordinator:
            self._coordinator.set_context_items(self._pending_context_paths)
            self._pending_context_paths = []
            try:
                context_panel = self.query_one("#context-panel", ContextPanel)
                context_panel.update_context(self._coordinator.shared_context)
            except Exception:
                pass
        # If coordinator isn't ready yet, keep paths queued
