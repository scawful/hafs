"""Modular Chat Screen for HAFS TUI.

This is the new modular chat screen that replaces the monolithic OrchestratorScreen.
It uses the core infrastructure (EventBus, StateStore, CommandRegistry,
BindingRegistry, NavigationController) for clean separation of concerns.

Architecture:
- ChatScreen: Thin layout coordinator (<400 lines)
- ChatInputPanel: Input with @mentions (existing widget)
- LaneContainer: Agent lane management (existing widget)
- ContextPanel: Shared context display (existing widget)
- SynergyPanel: ToM metrics (existing widget)
- HeadlessChatView: Headless mode output (existing widget)

The screen delegates most logic to the core infrastructure and widgets.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, LoadingIndicator, Static

from hafs.ui.core.command_registry import Command, CommandCategory, get_command_registry
from hafs.ui.core.event_bus import (
    AgentStatusEvent,
    ChatEvent,
    ContextEvent,
    PhaseEvent,
    get_event_bus,
)
from hafs.ui.core.navigation_controller import get_navigation_controller
from hafs.ui.core.state_store import get_state_store
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
    FOCUS = "focus"
    MULTI = "multi"


class ChatUIMode(Enum):
    """Primary UI mode for chat output."""
    HEADLESS = "headless"
    TERMINAL = "terminal"


class ChatScreen(Screen):
    """Modular chat screen with agent orchestration.

    Uses the core infrastructure for:
    - Input handling via NavigationController
    - State management via StateStore
    - Commands via CommandRegistry
    - Events via EventBus
    """

    BINDINGS = [
        Binding("1", "focus_lane_1", "Lane 1", show=False),
        Binding("2", "focus_lane_2", "Lane 2", show=False),
        Binding("3", "focus_lane_3", "Lane 3", show=False),
        Binding("4", "focus_lane_4", "Lane 4", show=False),
        Binding("tab", "next_lane", "Next Lane"),
        Binding("m", "toggle_view_mode", "View Mode", show=False),
        Binding("ctrl+n", "new_agent", "New Agent"),
        Binding("ctrl+x", "toggle_context", "Context"),
        Binding("ctrl+s", "toggle_synergy", "Synergy", show=False),
        Binding("ctrl+p", "command_palette", "Commands"),
        Binding("escape", "back", "Back"),
    ]

    DEFAULT_CSS = """
    ChatScreen {
        background: $surface;
    }

    ChatScreen #main-area {
        width: 100%;
        height: 1fr;
    }

    ChatScreen #lanes-area {
        width: 1fr;
        height: 100%;
        layout: vertical;
    }

    ChatScreen .hidden {
        display: none;
    }

    ChatScreen #status-bar-container {
        height: 1;
        width: 100%;
        background: $primary-darken-2;
    }

    ChatScreen #status-bar {
        width: 1fr;
        height: 1;
        background: $primary-darken-2;
        color: $text;
        padding: 0 1;
    }

    ChatScreen #mode-toggle {
        width: auto;
        height: 1;
    }

    ChatScreen #loading-overlay {
        layer: overlay;
        height: 100%;
        width: 100%;
        align: center middle;
        background: $surface 50%;
    }

    ChatScreen #start-overlay {
        layer: overlay;
        height: 100%;
        width: 100%;
        align: center middle;
        background: $surface 70%;
    }

    ChatScreen #start-dialog {
        width: 72;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }

    ChatScreen #start-title {
        width: 100%;
        content-align: center middle;
        padding-bottom: 1;
        color: $primary;
    }

    ChatScreen #start-subtitle {
        width: 100%;
        content-align: center middle;
        color: $text-muted;
        padding-bottom: 1;
    }

    ChatScreen #start-buttons {
        width: 100%;
        height: auto;
        content-align: center middle;
    }

    ChatScreen #start-hint {
        width: 100%;
        height: auto;
        content-align: center middle;
        color: $text-muted;
        padding-top: 1;
    }

    ChatScreen #footer-area {
        height: auto;
        background: $surface;
        border-top: solid $primary-darken-2;
    }

    ChatScreen #footer-grid {
        height: auto;
        width: 100%;
        layout: horizontal;
        padding: 0 1;
    }

    ChatScreen #which-key-bar {
        width: 2fr;
    }
    """

    LAYERS = ["base", "overlay", "_toastrack"]

    def __init__(
        self,
        coordinator: "AgentCoordinator | None" = None,
        context_paths: list[Path] | None = None,
    ) -> None:
        super().__init__()
        self._state = get_state_store()
        self._bus = get_event_bus()
        self._nav = get_navigation_controller()
        self._commands = get_command_registry()

        self._coordinator = coordinator
        self._chat_ui_mode: ChatUIMode | None = ChatUIMode.TERMINAL if coordinator else None
        self._view_mode = ViewMode.FOCUS
        self._focused_lane_index = 0
        self._focused_lane_id: str | None = None
        self._pending_context_paths: list[Path] = list(context_paths or [])
        self._context_visible = True
        self._synergy_visible = True
        self._headless_busy = False
        self._previous_flow_state = False

        self._register_commands()

    def _register_commands(self) -> None:
        """Register chat-specific commands."""
        commands = [
            Command(
                id="chat.add_agent",
                name="Add Agent",
                description="Add a new agent to the chat",
                handler=self.action_new_agent,
                category=CommandCategory.AGENT,
                keybinding="ctrl+n",
            ),
            Command(
                id="chat.toggle_context",
                name="Toggle Context Panel",
                description="Show or hide the context panel",
                handler=self.action_toggle_context,
                category=CommandCategory.VIEW,
                keybinding="ctrl+x",
            ),
            Command(
                id="chat.toggle_synergy",
                name="Toggle Synergy Panel",
                description="Show or hide the synergy metrics panel",
                handler=self.action_toggle_synergy,
                category=CommandCategory.VIEW,
                keybinding="ctrl+s",
            ),
            Command(
                id="chat.next_lane",
                name="Next Agent Lane",
                description="Cycle to the next agent lane",
                handler=self.action_next_lane,
                category=CommandCategory.NAVIGATION,
                keybinding="tab",
            ),
        ]

        for cmd in commands:
            try:
                self._commands.register(cmd)
            except ValueError:
                pass  # Already registered

    def compose(self) -> ComposeResult:
        """Compose the chat screen layout."""
        yield HeaderBar(id="header-bar")

        with Horizontal(id="status-bar-container"):
            yield ModeToggle(id="mode-toggle")
            yield Static(
                "Press [bold]^N[/] add agent | [bold]@name[/] mention | [bold]/help[/] commands",
                id="status-bar",
            )

        with Horizontal(id="main-area"):
            with Container(id="lanes-area"):
                yield LaneContainer(id="lanes")
                yield HeadlessChatView(id="headless-chat", classes="hidden")
            yield ContextPanel(id="context-panel")

        yield SynergyPanel(id="synergy-panel")

        yield ChatInput(
            agent_names=self._get_agent_names(),
            placeholder="Message agents... (@name to mention, /help for commands)",
            id="chat-input",
        )

        yield LoadingIndicator(id="loading-overlay", classes="hidden")

        with Container(
            id="start-overlay",
            classes="hidden" if self._coordinator else None,
        ):
            with Vertical(id="start-dialog"):
                yield Static("[bold]Start Chat[/bold]", id="start-title")
                yield Static("Choose how you want to run agents", id="start-subtitle")
                with Horizontal(id="start-buttons"):
                    yield Button("Quick Answer (Headless)", id="start-headless", variant="primary")
                    yield Button("Interactive Terminal", id="start-terminal")
                yield Static("Tip: Use /add to add specialist agents later", id="start-hint")

        with Container(id="footer-area"):
            with Horizontal(id="footer-grid"):
                yield WhichKeyBar(id="which-key-bar")
                yield Footer(compact=True, show_command_palette=False)

    async def on_mount(self) -> None:
        """Initialize screen on mount."""
        self._nav.set_screen_context("chat")

        # Load visibility states from store
        self._context_visible = self._state.get("settings.chat_context_visible", True)
        self._synergy_visible = self._state.get("settings.chat_synergy_visible", True)

        # Subscribe to events
        self._bus.subscribe("agent.*", self._on_agent_event)
        self._bus.subscribe("chat.*", self._on_chat_event)

        if self._coordinator:
            try:
                mode_toggle = self.query_one("#mode-toggle", ModeToggle)
                mode_toggle.set_mode(self._coordinator.mode)
            except Exception:
                pass
            await self._setup_agents()
            self._apply_context_paths()
            try:
                self.query_one("#chat-input", ChatInput).focus_input()
            except Exception:
                pass
        else:
            self._update_status("Choose a chat mode to begin")
            try:
                self.query_one("#start-headless", Button).focus()
            except Exception:
                pass

        self.set_interval(1.0, self._check_flow_state)

    def _on_agent_event(self, event: AgentStatusEvent) -> None:
        """Handle agent status events."""
        self._state.set(f"agents.{event.agent_id}.status", event.status)
        self._state.set(f"agents.{event.agent_id}.health", event.health)

    def _on_chat_event(self, event: ChatEvent) -> None:
        """Handle chat events."""
        pass  # Placeholder for chat event handling

    # Button handlers

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle start mode button presses."""
        button_id = event.button.id or ""
        if button_id == "start-headless":
            await self._begin_start(ChatUIMode.HEADLESS)
        elif button_id == "start-terminal":
            await self._begin_start(ChatUIMode.TERMINAL)

    async def _begin_start(self, mode: ChatUIMode) -> None:
        """Begin initializing chat in the selected UI mode."""
        if self._coordinator:
            return

        self._chat_ui_mode = mode
        self._set_start_overlay_visible(False)
        self._set_loading_visible(True)
        self._set_chat_view_mode(mode)

        label = "Headless" if mode == ChatUIMode.HEADLESS else "Terminal"
        self._update_status(f"Starting {label} mode...")

        self.run_worker(self._init_coordinator(mode))

    async def _init_coordinator(self, mode: ChatUIMode) -> None:
        """Initialize coordinator for the selected mode."""
        import asyncio

        try:
            import hafs.backends  # noqa: F401
            from hafs.agents.coordinator import AgentCoordinator
            from hafs.agents.roles import get_role_system_prompt
            from hafs.models.agent import AgentRole

            config = getattr(self.app, "config", None)
            if not config:
                from hafs.config.loader import load_config
                config = load_config()

            try:
                self._coordinator = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, lambda: AgentCoordinator(config)
                    ),
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                self.notify("Coordinator initialization timed out", severity="error")
                self._set_loading_visible(False)
                self._set_start_overlay_visible(True)
                return

            try:
                setattr(self.app, "_coordinator", self._coordinator)
            except Exception:
                pass

            default_backend = getattr(self.app, "_default_backend", "gemini")
            if mode == ChatUIMode.HEADLESS:
                default_backend = self._map_backend_for_headless(default_backend)

            # Create default agent
            await self._coordinator.register_agent(
                name="Assistant",
                role=AgentRole.GENERAL,
                backend_name=default_backend,
                system_prompt=get_role_system_prompt(AgentRole.GENERAL),
            )

            self._update_agent_names()
            self._apply_context_paths()

            if mode == ChatUIMode.TERMINAL:
                await self._setup_agents()

            self._set_loading_visible(False)
            self._set_start_overlay_visible(False)

            try:
                self.query_one("#chat-input", ChatInput).focus_input()
            except Exception:
                pass

            ready = "Headless" if mode == ChatUIMode.HEADLESS else "Terminal"
            self._update_status(f"{ready} mode ready | /add for more agents")

            # Publish event
            self._bus.publish(PhaseEvent(phase="plan", progress=0.0, message="Chat ready"))

        except Exception as exc:
            self.notify(f"Initialization failed: {exc}", severity="error")
            self._coordinator = None
            self._chat_ui_mode = None
            self._set_loading_visible(False)
            self._set_start_overlay_visible(True)

    async def _setup_agents(self) -> None:
        """Set up agent lanes from coordinator."""
        if not self._coordinator:
            return

        lanes = self.query_one("#lanes", LaneContainer)

        for name, lane in self._coordinator.agents.items():
            await lanes.add_lane(lane, f"lane-{name.lower()}")

        self._update_agent_names()

        try:
            context_panel = self.query_one("#context-panel", ContextPanel)
            context_panel.update_context(self._coordinator.shared_context)
        except Exception:
            pass

        lane_ids = lanes.lane_ids
        if lane_ids:
            self._focused_lane_id = lane_ids[0]
            self._focused_lane_index = 0
            lanes.set_focused_lane(self._focused_lane_id)
            self._update_focused_status()

        self.run_worker(self._start_agents_background())

    async def _start_agents_background(self) -> None:
        """Start all agents in background."""
        if self._coordinator:
            await self._coordinator.start_all_agents()
            self.notify("All agents started")

    # Chat input handling

    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        """Handle chat input submission."""
        message = event.value

        if message.startswith("/"):
            await self._handle_command(message)
            return

        if not self._coordinator or not self._chat_ui_mode:
            self.notify("Choose a chat mode to begin", severity="warning")
            self._set_start_overlay_visible(True)
            return

        # Publish chat event
        self._bus.publish(ChatEvent(
            content=message,
            role="user",
            agent_id="user",
        ))

        if self._chat_ui_mode == ChatUIMode.HEADLESS:
            await self._handle_headless_message(message)
        else:
            await self._handle_terminal_message(message)

    async def _handle_terminal_message(self, message: str) -> None:
        """Handle message in terminal mode."""
        if not self._coordinator:
            return

        target = await self._coordinator.route_message(message, sender="user")

        if target:
            lanes = self.query_one("#lanes", LaneContainer)
            lane_widget = lanes.get_lane_by_agent_name(target)

            if lane_widget:
                await lane_widget.start_streaming()
                lane_widget.focus()

            lane = self._coordinator.get_lane(target)
            if lane and not lane.is_busy:
                await lane.process_next_message()

    async def _handle_headless_message(self, message: str) -> None:
        """Handle message in headless mode."""
        if self._headless_busy or not self._coordinator:
            return

        self._headless_busy = True
        try:
            view = self.query_one("#headless-chat", HeadlessChatView)
            view.write_user(message)

            target = await self._coordinator.route_message(message, sender="user")
            lane = self._coordinator.get_lane(target)

            if lane and not lane.is_running:
                await lane.start()

            view.start_assistant(target)
            async for chunk in self._coordinator.stream_agent_response(target):
                view.write_assistant_chunk(chunk)
        finally:
            self._headless_busy = False

    async def _handle_command(self, command: str) -> None:
        """Handle slash commands."""
        parts = command[1:].split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # Delegate to command registry where possible
        command_mapping = {
            "help": self._show_help,
            "add": lambda: self._add_agent(args),
            "remove": lambda: self._remove_agent(args),
            "list": self._list_agents,
            "clear": self._clear_current_lane,
            "ui": lambda: self._switch_ui_mode(args),
        }

        handler = command_mapping.get(cmd)
        if handler:
            if asyncio.iscoroutinefunction(handler):
                await handler()
            else:
                handler()
        else:
            self.notify(f"Unknown command: {cmd}", severity="error")

    def _show_help(self) -> None:
        """Show help message."""
        self.notify(
            "Commands:\n"
            "  /add <name> <role> - Add new agent\n"
            "  /remove <name> - Remove agent\n"
            "  /list - List agents\n"
            "  /clear - Clear current lane\n"
            "  /ui <headless|terminal> - Switch UI mode",
            title="Help",
            timeout=8,
        )

    async def _add_agent(self, args: str) -> None:
        """Handle /add command."""
        parts = args.split(maxsplit=1)
        if len(parts) < 2:
            self.notify("Usage: /add <name> <role>", severity="error")
            return

        name, role_str = parts

        try:
            from hafs.agents.roles import get_role_system_prompt
            from hafs.models.agent import AgentRole

            role = AgentRole(role_str.lower())
        except ValueError:
            self.notify(f"Invalid role: {role_str}", severity="error")
            return

        if self._coordinator:
            try:
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

    async def _remove_agent(self, name: str) -> None:
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

    def _list_agents(self) -> None:
        """Handle /list command."""
        if self._coordinator and self._coordinator.agents:
            lines = [f"  {name}: {lane.agent.role.value}"
                     for name, lane in self._coordinator.agents.items()]
            self.notify("Agents:\n" + "\n".join(lines), title="Agent List", timeout=5)
        else:
            self.notify("No agents registered", severity="warning")

    def _clear_current_lane(self) -> None:
        """Clear the current lane."""
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

    async def _switch_ui_mode(self, mode_arg: str) -> None:
        """Switch UI mode."""
        mode_str = mode_arg.strip().lower()
        if mode_str not in ("headless", "terminal"):
            self.notify("Usage: /ui headless OR /ui terminal", severity="error")
            return

        desired = ChatUIMode.HEADLESS if mode_str == "headless" else ChatUIMode.TERMINAL
        if self._chat_ui_mode == desired:
            return

        self._chat_ui_mode = desired
        self._set_chat_view_mode(desired)

        if desired == ChatUIMode.TERMINAL and self._coordinator:
            await self._setup_agents()

        self.notify(f"UI mode: {mode_str}", timeout=2)

    # UI helpers

    def _set_loading_visible(self, visible: bool) -> None:
        """Set loading overlay visibility."""
        try:
            loading = self.query_one("#loading-overlay", LoadingIndicator)
            loading.set_class(not visible, "hidden")
        except Exception:
            pass

    def _set_start_overlay_visible(self, visible: bool) -> None:
        """Set start overlay visibility."""
        try:
            overlay = self.query_one("#start-overlay")
            overlay.set_class(not visible, "hidden")
        except Exception:
            pass

    def _set_chat_view_mode(self, mode: ChatUIMode) -> None:
        """Toggle between terminal and headless views."""
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

    def _update_status(self, message: str) -> None:
        """Update status bar."""
        try:
            status = self.query_one("#status-bar", Static)
            status.update(f"[bold]Chat[/] | {message}")
        except Exception:
            pass

    def _update_focused_status(self) -> None:
        """Update status bar with focused agent info."""
        if self._focused_lane_id:
            agent_name = self._focused_lane_id.replace("lane-", "").title()
            mode_str = "Focus" if self._view_mode == ViewMode.FOCUS else "Multi"
            self._update_status(
                f"[{mode_str}] Agent: [bold]{agent_name}[/] | Tab next | m toggle view"
            )

    def _get_agent_names(self) -> list[str]:
        """Get list of agent names."""
        if self._coordinator:
            return list(self._coordinator.agents.keys())
        return []

    def _update_agent_names(self) -> None:
        """Update chat input with agent names."""
        try:
            chat_input = self.query_one("#chat-input", ChatInput)
            chat_input.set_agent_names(self._get_agent_names())
        except Exception:
            pass

    def _apply_context_paths(self) -> None:
        """Apply pending context paths."""
        if not self._pending_context_paths or not self._coordinator:
            return

        self._coordinator.set_context_items(self._pending_context_paths)
        self._pending_context_paths = []

        try:
            context_panel = self.query_one("#context-panel", ContextPanel)
            context_panel.update_context(self._coordinator.shared_context)
        except Exception:
            pass

    def _check_flow_state(self) -> None:
        """Check for flow state changes."""
        try:
            synergy_panel = self.query_one("#synergy-panel", SynergyPanel)
            current_flow = synergy_panel.flow_state

            if current_flow != self._previous_flow_state:
                self._previous_flow_state = current_flow
                if current_flow:
                    self.notify(
                        "Agent in FLOW STATE - autonomous operation enabled",
                        title="Flow State Active",
                        timeout=5,
                    )
        except Exception:
            pass

    @staticmethod
    def _map_backend_for_headless(default_backend: str) -> str:
        """Map backends for headless mode."""
        mapping = {"gemini": "gemini_oneshot", "claude": "claude_oneshot"}
        return mapping.get(default_backend, default_backend)

    # Actions

    def action_focus_lane_1(self) -> None:
        self._focus_lane(0)

    def action_focus_lane_2(self) -> None:
        self._focus_lane(1)

    def action_focus_lane_3(self) -> None:
        self._focus_lane(2)

    def action_focus_lane_4(self) -> None:
        self._focus_lane(3)

    def action_next_lane(self) -> None:
        """Cycle to next lane."""
        if self._chat_ui_mode != ChatUIMode.TERMINAL:
            return
        lanes = self.query_one("#lanes", LaneContainer)
        lane_ids = lanes.lane_ids
        if lane_ids:
            next_idx = (self._focused_lane_index + 1) % len(lane_ids)
            self._focus_lane(next_idx)

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
            if self._view_mode == ViewMode.FOCUS:
                lanes.set_focused_lane(self._focused_lane_id)
            self._update_focused_status()

    def action_toggle_view_mode(self) -> None:
        """Toggle between focus and multi-view modes."""
        if self._chat_ui_mode != ChatUIMode.TERMINAL:
            return

        lanes = self.query_one("#lanes", LaneContainer)
        if self._view_mode == ViewMode.FOCUS:
            self._view_mode = ViewMode.MULTI
            lanes.set_view_mode("multi")
            self.notify("Multi-view mode", timeout=2)
        else:
            self._view_mode = ViewMode.FOCUS
            lanes.set_view_mode("focus")
            if self._focused_lane_id:
                lanes.set_focused_lane(self._focused_lane_id)
            self.notify("Focus mode", timeout=2)

    def action_toggle_context(self) -> None:
        """Toggle context panel visibility."""
        try:
            context_panel = self.query_one("#context-panel", ContextPanel)
            self._context_visible = not self._context_visible
            context_panel.set_class(not self._context_visible, "hidden")
            self._state.set("settings.chat_context_visible", self._context_visible)
        except Exception:
            pass

    def action_toggle_synergy(self) -> None:
        """Toggle synergy panel visibility."""
        try:
            synergy_panel = self.query_one("#synergy-panel", SynergyPanel)
            self._synergy_visible = not self._synergy_visible
            synergy_panel.set_class(not self._synergy_visible, "hidden")
            self._state.set("settings.chat_synergy_visible", self._synergy_visible)
        except Exception:
            pass

    def action_new_agent(self) -> None:
        """Prompt to add new agent."""
        self.notify(
            "Use /add <name> <role> to add an agent\n"
            "Roles: general, planner, coder, critic, researcher",
            title="Add Agent",
            timeout=5,
        )

    def action_command_palette(self) -> None:
        """Open command palette."""
        from hafs.ui.screens.command_palette import CommandPalette
        self.app.push_screen(CommandPalette())

    def action_back(self) -> None:
        """Go back to previous screen."""
        if len(self.app.screen_stack) > 1:
            self.app.pop_screen()
        else:
            from hafs.ui.screens.dashboard import DashboardScreen
            self.app.switch_screen(DashboardScreen())


# Import for async check
import asyncio
