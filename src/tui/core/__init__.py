"""Core UI infrastructure for HAFS TUI.

This module provides the foundational components for the TUI:
- EventBus: Pub/sub system for cross-widget communication
- StateStore: Centralized reactive state container
- CommandRegistry: Extensible command registration
- BindingRegistry: Centralized keybinding management
- ScreenRouter: Navigation management
- NavigationController: Unified input handling
- ChatProtocol: Chat event protocol for streaming
- ChatAdapter: Bridge between orchestration and UI
- Accessibility: Theme and accessibility settings
- Cheatsheet: Keybinding cheatsheet generator
"""

from tui.core.event_bus import EventBus, Event, Subscription
from tui.core.state_store import StateStore, StateSlice
from tui.core.command_registry import CommandRegistry, Command
from tui.core.binding_registry import BindingRegistry, Binding, BindingConflict
from tui.core.screen_router import ScreenRouter, Route
from tui.core.navigation_controller import (
    NavigationController,
    VimMode,
    InputMode,
    get_navigation_controller,
)
from tui.core.accessibility import (
    AccessibilityManager,
    ThemeMode,
    get_accessibility,
)
from tui.core.cheatsheet import KeybindingCheatsheet, generate_cheatsheet
from tui.core.chat_protocol import (
    ChatMessage,
    MessageRole,
    MessageStatus,
    StreamingContext,
)
from tui.core.chat_adapter import ChatAdapter

__all__ = [
    # Event Bus
    "EventBus",
    "Event",
    "Subscription",
    # State Store
    "StateStore",
    "StateSlice",
    # Command Registry
    "CommandRegistry",
    "Command",
    # Binding Registry
    "BindingRegistry",
    "Binding",
    "BindingConflict",
    # Screen Router
    "ScreenRouter",
    "Route",
    # Navigation Controller
    "NavigationController",
    "VimMode",
    "InputMode",
    "get_navigation_controller",
    # Accessibility
    "AccessibilityManager",
    "ThemeMode",
    "get_accessibility",
    # Cheatsheet
    "KeybindingCheatsheet",
    "generate_cheatsheet",
    # Chat Protocol
    "ChatMessage",
    "MessageRole",
    "MessageStatus",
    "StreamingContext",
    # Chat Adapter
    "ChatAdapter",
]
