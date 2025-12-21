"""Multi-agent orchestration screen."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, LoadingIndicator, Static

from hafs.ui.core.standard_keymaps import get_standard_keymap
from hafs.ui.mixins.vim_navigation import VimNavigationMixin
from hafs.ui.mixins.which_key import WhichKeyMixin
from hafs.ui.screens.permissions_modal import PermissionsModal
from hafs.ui.widgets.chat_input import ChatInput
from hafs.ui.widgets.context_panel import ContextPanel
from hafs.ui.widgets.headless_chat import HeadlessChatView
from hafs.ui.widgets.header_bar import HeaderBar
from hafs.ui.widgets.lane_container import LaneContainer
from hafs.ui.widgets.mode_toggle import ModeToggle
from hafs.ui.widgets.synergy_panel import SynergyPanel
from hafs.ui.widgets.which_key_bar import WhichKeyBar

if TYPE_CHECKING:
    from hafs.agents.coordinator import AgentCoordinator


class ViewMode(Enum):
    """View mode for agent lanes."""

    FOCUS = "focus"  # Single agent, full terminal
    MULTI = "multi"  # All agents, side-by-side (truncated when 3+)


class ChatUIMode(Enum):
    """Primary UI mode for chat output."""

    HEADLESS = "headless"  # Stream parsed output into transcript
    TERMINAL = "terminal"  # Full terminal emulation per agent


class OrchestratorScreen(WhichKeyMixin, VimNavigationMixin, Screen):
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

    # Track flow state for change detection
    _previous_flow_state: bool = False

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
        layout: vertical;
    }

    OrchestratorScreen .hidden {
        display: none;
    }

    OrchestratorScreen #status-bar-container {
        height: 1;
        width: 100%;
        background: $primary;
    }

    OrchestratorScreen #status-bar {
        width: 1fr;
        height: 1;
        background: $primary;
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

    OrchestratorScreen #start-overlay {
        layer: overlay;
        height: 100%;
        width: 100%;
        align: center middle;
        background: $surface 70%;
    }

    OrchestratorScreen #start-dialog {
        width: 72;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }

    OrchestratorScreen #start-title {
        width: 100%;
        content-align: center middle;
        padding-bottom: 1;
        color: $primary;
    }

    OrchestratorScreen #start-subtitle {
        width: 100%;
        content-align: center middle;
        color: $text-disabled;
        padding-bottom: 1;
    }

    OrchestratorScreen #start-buttons {
        width: 100%;
        height: auto;
        content-align: center middle;
    }

    OrchestratorScreen #start-hint {
        width: 100%;
        height: auto;
        content-align: center middle;
        color: $text-disabled;
        padding-top: 1;
    }

    OrchestratorScreen #footer-area {
        height: auto;
        background: $surface;
        border-top: solid $primary;
    }

    OrchestratorScreen #which-key-bar {
        width: 100%;
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
        self._chat_ui_mode: ChatUIMode | None = ChatUIMode.TERMINAL if coordinator else None
        self._context_visible = True
        self._synergy_visible = True
        self._focused_lane_index = 0
        self._view_mode = ViewMode.FOCUS
        self._focused_lane_id: str | None = None
        self._pending_context_paths: list[Path] = list(context_paths or [])
        self._headless_busy: bool = False
        self._metacognition_monitor = None
        self._fears_repo = None

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
            # Left pane (either terminal lanes or headless transcript)
            with Container(id="lanes-area"):
                yield LaneContainer(id="lanes")
                yield HeadlessChatView(id="headless-chat", classes="hidden")

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

        # Overlay: spinner used during background initialization
        yield LoadingIndicator(id="loading-overlay", classes="hidden")

        # Overlay: start mode chooser (only when no coordinator yet)
        with Container(
            id="start-overlay",
            classes="hidden" if self._coordinator else None,
        ):
            with Vertical(id="start-dialog"):
                yield Static("[bold]Start Chat[/bold]", id="start-title")
                yield Static(
                    "Choose how you want to run agents",
                    id="start-subtitle",
                )
                with Horizontal(id="start-buttons"):
                    yield Button(
                        "Quick Answer (Headless)",
                        id="start-headless",
                        variant="primary",
                    )
                    yield Button(
                        "Interactive Terminal",
                        id="start-terminal",
                    )
                yield Static(
                    "Tip: Use /add to add specialist agents later",
                    id="start-hint",
                )

        # Footer area
        with Container(id="footer-area"):
            yield WhichKeyBar(id="which-key-bar")

    def get_which_key_map(self):
        """Return which-key bindings for this screen."""
        keymap = get_standard_keymap(self)
        keymap["a"] = ("+agent", {
            "n": ("new", self.action_new_agent),
            "k": ("kill", self.action_kill_agent),
            "1": ("lane1", self.action_focus_lane_1),
            "2": ("lane2", self.action_focus_lane_2),
            "3": ("lane3", self.action_focus_lane_3),
            "4": ("lane4", self.action_focus_lane_4),
        })
        keymap["v"] = ("+view", {
            "c": ("context", self.action_toggle_context),
            "s": ("synergy", self.action_toggle_synergy),
            "m": ("mode", self.action_toggle_view_mode),
        })
        keymap["p"] = ("permissions", self.action_manage_permissions)
        keymap["y"] = ("yolo", self.action_send_yolo)
        return keymap

    async def on_mount(self) -> None:
        """Handle screen mount."""
        # Initialize vim navigation (loads setting from config)
        self.init_vim_navigation()
        # Initialize which-key hints
        self.init_which_key_hints()

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

        if not self._coordinator:
            # No coordinator yet: prompt the user to choose a start mode.
            initial_agents = getattr(self.app, "_initial_agents", None)
            if initial_agents:
                # If the user explicitly supplied agents via CLI, preserve the
                # previous behavior and start in terminal mode automatically.
                self._begin_start(ChatUIMode.TERMINAL)
            else:
                self._update_status("Choose a chat mode to begin")
                try:
                    self.query_one("#start-headless", Button).focus()
                except Exception:
                    pass
        else:
            # Coordinator provided: assume terminal mode and set up lanes.
            self._chat_ui_mode = self._chat_ui_mode or ChatUIMode.TERMINAL
            await self._setup_default_agents()
            self._apply_context_paths()

            # Focus the chat input
            try:
                self.query_one("#chat-input", ChatInput).focus_input()
            except Exception:
                pass

        # Start flow state monitoring
        self.set_interval(1.0, self._check_flow_state)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses (start mode chooser)."""
        button_id = event.button.id or ""
        if button_id == "start-headless":
            self._begin_start(ChatUIMode.HEADLESS)
        elif button_id == "start-terminal":
            self._begin_start(ChatUIMode.TERMINAL)

    def _begin_start(self, mode: ChatUIMode) -> None:
        """Begin initializing chat in the selected UI mode."""
        if self._coordinator:
            return

        self._chat_ui_mode = mode
        self._headless_busy = False

        self._set_start_overlay_visible(False)
        self._set_loading_visible(True)
        self._set_chat_view_mode(mode)

        label = "Headless" if mode == ChatUIMode.HEADLESS else "Terminal"
        self._update_status(f"Starting {label} mode…")
        self.run_worker(self._init_coordinator_for_mode(mode))

    def _set_loading_visible(self, visible: bool) -> None:
        try:
            loading = self.query_one("#loading-overlay", LoadingIndicator)
            if visible:
                loading.remove_class("hidden")
            else:
                loading.add_class("hidden")
        except Exception:
            pass

    def _set_start_overlay_visible(self, visible: bool) -> None:
        try:
            overlay = self.query_one("#start-overlay")
            if visible:
                overlay.remove_class("hidden")
            else:
                overlay.add_class("hidden")
        except Exception:
            pass

    def _set_chat_view_mode(self, mode: ChatUIMode) -> None:
        """Toggle between terminal lanes and headless transcript."""
        try:
            lanes = self.query_one("#lanes", LaneContainer)
            headless = self.query_one("#headless-chat", HeadlessChatView)
            if mode == ChatUIMode.HEADLESS:
                lanes.add_class("hidden")
                headless.remove_class("hidden")
            else:
                headless.add_class("hidden")
                lanes.remove_class("hidden")
        except Exception:
            pass

    async def _init_coordinator_for_mode(self, mode: ChatUIMode) -> None:
        """Create coordinator + initial agent(s) for the chosen UI mode."""
        import asyncio

        try:
            import hafs.backends  # noqa: F401
            from hafs.agents.coordinator import AgentCoordinator
            from hafs.agents.roles import get_role_system_prompt
            from hafs.models.agent import AgentRole

            # Get config from app
            config = getattr(self.app, "config", None)
            if not config:
                from hafs.config.loader import load_config

                config = load_config()

            # Ensure cognitive protocol scaffold exists (non-destructive).
            try:
                from hafs.core.afs.manager import AFSManager

                AFSManager(config).ensure(Path.cwd())
            except Exception:
                pass

            # Create coordinator
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
                self._set_loading_visible(False)
                self._set_start_overlay_visible(True)
                return

            # Persist coordinator on the app so other screens can reuse it
            try:
                setattr(self.app, "_coordinator", self._coordinator)
            except Exception:
                pass

            default_backend = getattr(self.app, "_default_backend", "gemini")
            if mode == ChatUIMode.HEADLESS:
                default_backend = self._map_backend_for_headless(default_backend)
            agents_to_init = [{"name": "Assistant", "role": "general"}]
            initial_agents = getattr(self.app, "_initial_agents", None)
            if mode == ChatUIMode.TERMINAL and initial_agents:
                agents_to_init = list(initial_agents)

            for i, agent_spec in enumerate(agents_to_init, 1):
                if not self._coordinator:
                    break
                name = agent_spec.get("name", f"Agent{i}")
                role = AgentRole(agent_spec.get("role", "general"))
                persona = agent_spec.get("persona")
                self._update_status(f"Registering {i}/{len(agents_to_init)}: {name}…")
                await self._coordinator.register_agent(
                    name=name,
                    role=role,
                    backend_name=default_backend,
                    system_prompt=get_role_system_prompt(role, persona=persona),
                    persona=persona,
                )

            # Update UI pieces common to both modes
            self._update_agent_names()
            try:
                context_panel = self.query_one("#context-panel", ContextPanel)
                if self._coordinator:
                    context_panel.update_context(self._coordinator.shared_context)
            except Exception:
                pass

            self._apply_context_paths()

            # Terminal mode mounts lanes and starts agent(s) so the prompt appears.
            if mode == ChatUIMode.TERMINAL and self._coordinator:
                await self._setup_default_agents()

            # Hide overlays and focus input
            self._set_loading_visible(False)
            self._set_start_overlay_visible(False)
            try:
                self.query_one("#chat-input", ChatInput).focus_input()
            except Exception:
                pass

            ready = "Headless" if mode == ChatUIMode.HEADLESS else "Terminal"
            self._update_status(f"{ready} mode ready | /add for more agents | /help for commands")

        except ImportError as exc:
            self.notify(f"Failed to load agent modules: {exc}", severity="error")
            self._update_status(f"Error: {exc}")
            self._coordinator = None
            self._chat_ui_mode = None
            self._set_start_overlay_visible(True)
        except Exception as exc:
            self.notify(f"Chat initialization failed: {exc}", severity="error")
            self._update_status(f"Error: {exc}")
            self._coordinator = None
            self._chat_ui_mode = None
            self._set_start_overlay_visible(True)
        finally:
            self._set_loading_visible(False)

    def _update_state_last_user_input(self, message: str) -> None:
        """Update .context/scratchpad/state.md with the latest user input."""
        sanitized = " ".join(message.strip().splitlines()).strip()
        if not sanitized:
            return

        try:
            from hafs.core.afs.state_contextualizer import update_state_md

            update_state_md(Path.cwd(), last_user_input=sanitized)
        except Exception:
            return

    def _apply_fears_to_state(self, message: str) -> None:
        """Consult memory/fears.json and update state.md risk section when relevant."""
        fears_file = Path.cwd() / ".context" / "memory" / "fears.json"
        state_file = Path.cwd() / ".context" / "scratchpad" / "state.md"
        if not (fears_file.exists() and state_file.exists()):
            return

        try:
            from hafs.core.fears.repository import FearsRepository

            if self._fears_repo is None:
                self._fears_repo = FearsRepository(fears_file)
            matches = self._fears_repo.match(message)
        except Exception:
            return

        if not matches:
            return

        try:
            from hafs.core.fears.scoring import compute_confidence, strongest_match

            match = strongest_match(matches) or matches[0]
            confidence = compute_confidence(matches)
        except Exception:
            match = matches[0]
            confidence = 0.4

        concern = match.concern.strip()
        mitigation = match.mitigation.strip()

        try:
            from hafs.core.protocol.state_md import update_risk_block

            update_risk_block(
                state_file,
                concern=concern,
                confidence=confidence,
                mitigation=mitigation,
            )
        except Exception:
            return

    def _record_metacognition_action(self, action_description: str) -> None:
        """Record an action into .context/scratchpad/metacognition.json."""
        try:
            if self._metacognition_monitor is None:
                from hafs.core.metacognition.monitor import MetacognitionMonitor

                self._metacognition_monitor = MetacognitionMonitor()
                self._metacognition_monitor.load_state()

            monitor = self._metacognition_monitor
            monitor.record_action(action_description)
            monitor.check_flow_state()
            monitor.save_state()
        except Exception:
            pass

    def _record_metacognition_success(self) -> None:
        """Record a successful step and persist metacognition state."""
        try:
            if self._metacognition_monitor is None:
                return
            monitor = self._metacognition_monitor
            monitor.record_success()
            monitor.check_flow_state()
            monitor.save_state()
        except Exception:
            pass

    def _update_status(self, message: str) -> None:
        """Update status bar with message."""
        try:
            status_bar = self.query_one("#status-bar", Static)
            status_bar.update(f"[bold]Chat[/] | {message}")
        except Exception:
            pass

    def _check_flow_state(self) -> None:
        """Check for flow state changes and notify user."""
        try:
            synergy_panel = self.query_one("#synergy-panel", SynergyPanel)
            current_flow = synergy_panel.flow_state

            if current_flow != self._previous_flow_state:
                self._previous_flow_state = current_flow
                if current_flow:
                    # Entering flow state
                    self.notify(
                        "Agent is in [bold green]FLOW STATE[/]\n"
                        "Autonomous operation enabled - reduced confirmations",
                        title="⚡ Flow State Active",
                        timeout=5,
                    )
                    self._update_flow_status_indicator(True)
                else:
                    # Exiting flow state
                    self.notify(
                        "Flow state ended - returning to normal operation",
                        title="Flow State Ended",
                        timeout=3,
                    )
                    self._update_flow_status_indicator(False)
        except Exception:
            pass

    def _update_flow_status_indicator(self, in_flow: bool) -> None:
        """Update status bar to indicate flow state."""
        try:
            status_bar = self.query_one("#status-bar", Static)
            if in_flow:
                status_bar.update(
                    "[bold green]⚡ FLOW[/] | Autonomous mode active - agent operating optimally"
                )
            else:
                # Restore default status
                self._update_focused_status()
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

            if self._coordinator:
                self._attach_history(self._coordinator, config)

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
                        f"Registering agent {i}/{len(agents_to_init)}: {agent_spec['name']}..."
                    )
                    role = AgentRole(agent_spec.get("role", "general"))
                    if self._coordinator:
                        await self._coordinator.register_agent(
                            name=agent_spec["name"],
                            role=role,
                            backend_name=default_backend,
                            persona=agent_spec.get("persona"),
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

    def _attach_history(self, coordinator: "AgentCoordinator", config) -> None:
        """Attach history logger and session manager to coordinator."""
        try:
            from hafs.core.history import HistoryLogger, SessionManager
        except Exception:
            return

        context_root = getattr(config.general, "context_root", None)
        if context_root is None:
            return

        history_dir = context_root / "history"
        sessions_dir = history_dir / "sessions"
        project_id = Path.cwd().name

        session_manager = SessionManager(sessions_dir, project_id=project_id)
        history_logger = HistoryLogger(
            history_dir=history_dir,
            session_manager=session_manager,
            project_id=project_id,
        )
        session_manager.set_history_logger(history_logger)
        session_manager.create()

        coordinator.set_session_manager(session_manager)
        coordinator.set_history_logger(history_logger)

    async def set_coordinator(self, coordinator: "AgentCoordinator") -> None:
        """Set the coordinator and initialize agents.

        Args:
            coordinator: The initialized AgentCoordinator.
        """
        self._coordinator = coordinator
        self._chat_ui_mode = self._chat_ui_mode or ChatUIMode.TERMINAL
        try:
            setattr(self.app, "_coordinator", coordinator)
        except Exception:
            pass

        config = getattr(self.app, "config", None)
        if config:
            self._attach_history(coordinator, config)

        # Update status to show we're setting up agents
        total_agents = len(coordinator.agents) if coordinator.agents else 0
        self._update_status(f"Setting up {total_agents} agents...")

        self._set_loading_visible(False)
        self._set_start_overlay_visible(False)
        self._set_chat_view_mode(ChatUIMode.TERMINAL)

        await self._setup_default_agents()
        self._apply_context_paths()

        # Update status to ready state
        self._update_status("Press [bold]Ctrl+N[/] to add agent | [bold]@name[/] to mention")

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

        if not self._coordinator or not self._chat_ui_mode:
            self.notify("Choose a chat mode to begin", severity="warning")
            self._set_start_overlay_visible(True)
            return

        if self._chat_ui_mode == ChatUIMode.HEADLESS:
            await self._handle_headless_message(message)
            return

        self._update_state_last_user_input(message)
        self._apply_fears_to_state(message)
        self._record_metacognition_action(f"chat:{message[:120]}")

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

    async def _handle_headless_message(self, message: str) -> None:
        """Handle a user message in headless mode."""
        if self._headless_busy:
            self.notify("Wait for the current response to finish", severity="warning")
            return

        if not self._coordinator:
            return

        self._headless_busy = True
        try:
            view = self.query_one("#headless-chat", HeadlessChatView)
            view.write_user(message)

            self._update_state_last_user_input(message)
            self._apply_fears_to_state(message)
            self._record_metacognition_action(f"chat:{message[:120]}")

            try:
                target = await self._coordinator.route_message(message, sender="user")
            except Exception as exc:
                view.write_system(f"Route error: {exc}")
                return

            lane = self._coordinator.get_lane(target)
            if lane and not lane.is_running:
                await lane.start()

            view.start_assistant(target)
            async for chunk in self._coordinator.stream_agent_response(target):
                view.write_assistant_chunk(chunk)

            self._record_metacognition_success()
        finally:
            self._headless_busy = False

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
                "  /mode <planning|execution> - Set coordinator mode\n"
                "  /ui <headless|terminal> - Switch UI mode\n"
                "  /orchestrate [topic] - Run plan→execute→verify→summarize\n"
                "\nProtocol:\n"
                "  /open <state|goals|deferred|metacognition|fears>\n"
                "  /goal <text> - Set primary goal\n"
                "  /defer <text> - Append to deferred\n"
                "  /snapshot [reason] - Save state.md to history",
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
        elif cmd == "ui" and args:
            await self._ui_command(args)
        elif cmd == "open" and args:
            await self._open_protocol_command(args)
        elif cmd == "goal" and args:
            await self._goal_command(args)
        elif cmd == "defer" and args:
            await self._defer_command(args)
        elif cmd == "snapshot":
            await self._snapshot_command(args)
        elif cmd == "orchestrate":
            await self._orchestrate_command(args)
        else:
            self.notify(f"Unknown command: {cmd}", severity="error")

    async def _open_protocol_command(self, args: str) -> None:
        """Handle /open (open protocol file in the main viewer)."""
        kind = args.strip().lower()
        allowed = {"state", "goals", "deferred", "metacognition", "fears"}
        if kind not in allowed:
            self.notify("Usage: /open state|goals|deferred|metacognition|fears", severity="error")
            return

        try:
            from hafs.core.protocol.actions import open_protocol_file

            # kind is validated above
            path = open_protocol_file(Path.cwd(), kind)  # type: ignore[arg-type]
        except Exception as exc:
            self.notify(f"Failed to resolve protocol file: {exc}", severity="error")
            return

        try:
            setattr(self.app, "_pending_open_path", path)
            self.app.action_switch_main()
        except Exception:
            pass

        self.notify(f"Opening: {path}", timeout=2)

    async def _goal_command(self, args: str) -> None:
        """Handle /goal (set primary goal)."""
        text = args.strip()
        if not text:
            self.notify("Usage: /goal <text>", severity="error")
            return
        try:
            from hafs.core.protocol.actions import set_primary_goal

            path = set_primary_goal(Path.cwd(), text)
            self.notify("Primary goal updated", timeout=2)
        except Exception as exc:
            self.notify(f"Failed to set goal: {exc}", severity="error")
            return

    async def _defer_command(self, args: str) -> None:
        """Handle /defer (append to deferred.md)."""
        text = args.strip()
        if not text:
            self.notify("Usage: /defer <text>", severity="error")
            return
        try:
            from hafs.core.protocol.actions import append_deferred

            path = append_deferred(Path.cwd(), text)
            self.notify("Deferred item added", timeout=2)
        except Exception as exc:
            self.notify(f"Failed to defer: {exc}", severity="error")
            return

    async def _snapshot_command(self, args: str) -> None:
        """Handle /snapshot (copy state.md to history)."""
        reason = args.strip() or None
        try:
            from hafs.core.protocol.actions import snapshot_state

            dest = snapshot_state(Path.cwd(), reason=reason)
            if dest:
                self.notify(f"Snapshot saved: {dest.name}", timeout=2)
            else:
                self.notify("Snapshot failed", severity="error")
        except Exception as exc:
            self.notify(f"Snapshot failed: {exc}", severity="error")

    async def _orchestrate_command(self, args: str) -> None:
        """Handle /orchestrate (run the pipeline)."""
        topic = args.strip()
        if not topic and self._coordinator:
            topic = self._coordinator.shared_context.active_task or ""
        if not topic:
            self.notify("Usage: /orchestrate <topic>", severity="error")
            return

        self._update_status(f"Orchestrating: {topic}")
        self.run_worker(
            self._run_orchestration(topic),
            exclusive=True,
            group="orchestration",
        )

    async def _run_orchestration(self, topic: str) -> None:
        """Execute the unified orchestration pipeline and surface results."""
        from hafs.core.orchestration_entrypoint import run_orchestration

        default_backend = getattr(self.app, "_default_backend", "gemini")
        if self._chat_ui_mode == ChatUIMode.HEADLESS:
            default_backend = self._map_backend_for_headless(default_backend)

        config = getattr(self.app, "config", None)
        agent_specs = None
        coordinator = self._coordinator
        auto_add_lanes = self._chat_ui_mode == ChatUIMode.TERMINAL and coordinator is not None

        async def _attach_lane(lane) -> None:
            if not auto_add_lanes:
                return
            try:
                lanes = self.query_one("#lanes", LaneContainer)
                await lanes.add_lane(lane, f"lane-{lane.agent.name.lower()}")
                self._update_agent_names()
            except Exception:
                return

        try:
            result = await run_orchestration(
                mode="coordinator",
                topic=topic,
                agents=agent_specs,
                default_backend=default_backend,
                config=config,
                coordinator=coordinator,
                on_agent_registered=_attach_lane,
            )
        except Exception as exc:
            self.notify(f"Orchestration failed: {exc}", severity="error")
            self._update_status("Orchestration failed")
            return

        context_root = (
            getattr(config.general, "context_root", None)
            if config
            else Path.home() / ".context"
        ) or (Path.home() / ".context")
        output_dir = context_root / "scratchpad" / "orchestration"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"orchestration_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        filename.write_text(result or "", encoding="utf-8")

        if self._chat_ui_mode == ChatUIMode.HEADLESS:
            try:
                view = self.query_one("#headless-chat", HeadlessChatView)
                view.write_system("Orchestration complete.")
                view.start_assistant("Orchestrator")
                view.write_assistant_chunk(result or "")
            except Exception:
                pass

        if self._coordinator:
            self._update_agent_names()

        self.notify(f"Orchestration saved: {filename}", timeout=4)
        self._update_status("Orchestration complete")

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
                from hafs.agents.roles import get_role_system_prompt

                default_backend = getattr(self.app, "_default_backend", "gemini")
                if self._chat_ui_mode == ChatUIMode.HEADLESS:
                    default_backend = self._map_backend_for_headless(default_backend)
                lane = await self._coordinator.register_agent(
                    name=name,
                    role=role,
                    backend_name=default_backend,
                    system_prompt=get_role_system_prompt(role),
                )
                if self._chat_ui_mode == ChatUIMode.TERMINAL:
                    lanes = self.query_one("#lanes", LaneContainer)
                    await lanes.add_lane(lane, f"lane-{name.lower()}")
                self._update_agent_names()
                self.notify(f"Added agent: {name} ({role.value})")
            except Exception as e:
                self.notify(f"Failed to add agent: {e}", severity="error")

    @staticmethod
    def _map_backend_for_headless(default_backend: str) -> str:
        """Map interactive backends to one-shot versions for headless mode."""
        mapping = {
            "gemini": "gemini_oneshot",
            "claude": "claude_oneshot",
        }
        return mapping.get(default_backend, default_backend)

    async def _remove_agent_command(self, name: str) -> None:
        """Handle /remove command."""
        if self._coordinator:
            try:
                await self._coordinator.unregister_agent(name)
                if self._chat_ui_mode == ChatUIMode.TERMINAL:
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
            try:
                from hafs.core.goals.manager import GoalManager

                goals = GoalManager()
                goals.load_state()
                goals.set_primary_goal(description=task, user_stated=task)
                goals.save_state()
            except Exception:
                pass
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

    async def _ui_command(self, mode_arg: str) -> None:
        """Handle /ui command (switch between headless and terminal output)."""
        mode_str = mode_arg.strip().lower()
        if mode_str not in ("headless", "terminal"):
            self.notify("Usage: /ui headless OR /ui terminal", severity="error")
            return

        desired = ChatUIMode.HEADLESS if mode_str == "headless" else ChatUIMode.TERMINAL
        if self._chat_ui_mode == desired:
            self.notify(f"Already in {mode_str} mode", timeout=2)
            return

        self._chat_ui_mode = desired
        self._set_chat_view_mode(desired)

        if desired == ChatUIMode.TERMINAL and self._coordinator:
            # Ensure lanes exist for all current agents.
            await self._setup_default_agents()

        try:
            view = self.query_one("#headless-chat", HeadlessChatView)
            view.write_system(f"Switched UI mode to {mode_str}")
        except Exception:
            pass

        self.notify(f"UI mode: {mode_str}", timeout=2)

    def _clear_current_lane(self) -> None:
        """Clear the currently focused lane."""
        if self._chat_ui_mode == ChatUIMode.HEADLESS:
            try:
                self.query_one("#headless-chat", HeadlessChatView).clear()
            except Exception:
                pass
            return

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
        if self._chat_ui_mode != ChatUIMode.TERMINAL:
            return
        lanes = self.query_one("#lanes", LaneContainer)
        lane_ids = lanes.lane_ids
        if lane_ids:
            next_index = (self._focused_lane_index + 1) % len(lane_ids)
            self._focus_lane(next_index)

    def action_toggle_view_mode(self) -> None:
        """Toggle between focus and multi-view modes."""
        if self._is_input_focused():
            return
        if self._chat_ui_mode != ChatUIMode.TERMINAL:
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
        if self._chat_ui_mode != ChatUIMode.TERMINAL:
            return
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

    def on_context_panel_open_file_requested(self, event: ContextPanel.OpenFileRequested) -> None:
        """Handle context item click to open file in main viewer."""
        path = event.path
        if not path.exists():
            self.notify(f"File not found: {path}", severity="warning")
            return

        # Set the path for the main screen to open when we switch
        try:
            setattr(self.app, "_pending_open_path", path)
            self.app.action_switch_main()
            self.notify(f"Opening: {path.name}", timeout=2)
        except Exception as exc:
            self.notify(f"Failed to open file: {exc}", severity="error")

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

    async def on_header_bar_navigation_requested(self, event: HeaderBar.NavigationRequested) -> None:
        """Handle header bar navigation requests."""
        from hafs.ui.core.screen_router import get_screen_router

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
