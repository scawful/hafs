"""Screen Router - Navigation management for TUI screens.

This module provides URL-style routing for the TUI, enabling:
- Path-based screen navigation (/dashboard, /chat, /logs)
- Parameter passing between screens
- Navigation history with back/forward
- Modal/overlay screen stacking
- Route guards (authentication, permissions)
- Deep linking support

Routes:
- /dashboard - Main dashboard
- /chat - Multi-agent chat
- /chat/:agent_id - Chat with specific agent
- /logs - Log browser
- /logs/:date - Logs for specific date
- /settings - Settings screen
- /services - Services management
- /knowledge - Knowledge graph explorer
- /analysis - Research analysis dashboard

Usage:
    router = ScreenRouter(app)

    # Register routes
    router.register("/dashboard", DashboardScreen)
    router.register("/chat/:agent_id", ChatScreen)

    # Navigate
    await router.navigate("/chat/planner")

    # Go back
    await router.back()
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Pattern, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from textual.app import App
    from textual.screen import Screen

logger = logging.getLogger(__name__)


@dataclass
class Route:
    """A registered route in the router.

    Routes map URL-like paths to screen classes.
    Supports path parameters like /chat/:agent_id.
    """
    path: str
    screen_class: Type["Screen"]
    name: Optional[str] = None
    title: Optional[str] = None
    icon: Optional[str] = None
    guard: Optional[Callable[["RouteContext"], bool]] = None
    is_modal: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Compiled regex for matching
    _pattern: Optional[Pattern] = field(default=None, repr=False)
    _param_names: List[str] = field(default_factory=list, repr=False)

    def __post_init__(self):
        self._compile_pattern()

    def _compile_pattern(self) -> None:
        """Compile path into regex pattern for matching."""
        # Extract parameter names
        param_pattern = r":([a-zA-Z_][a-zA-Z0-9_]*)"
        self._param_names = re.findall(param_pattern, self.path)

        # Convert path to regex
        regex = self.path
        regex = re.sub(param_pattern, r"([^/]+)", regex)
        regex = f"^{regex}$"
        self._pattern = re.compile(regex)

    def match(self, path: str) -> Optional[Dict[str, str]]:
        """Check if a path matches this route.

        Args:
            path: The path to match

        Returns:
            Dict of path parameters if matched, None otherwise
        """
        if not self._pattern:
            return None

        match = self._pattern.match(path)
        if not match:
            return None

        # Extract parameters
        params = {}
        for i, name in enumerate(self._param_names):
            params[name] = match.group(i + 1)

        return params


@dataclass
class RouteContext:
    """Context passed to route guards and screen factories."""
    path: str
    params: Dict[str, str]
    query: Dict[str, str]
    previous_path: Optional[str]
    is_modal: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NavigationEntry:
    """An entry in the navigation history."""
    path: str
    params: Dict[str, str]
    query: Dict[str, str]
    timestamp: datetime = field(default_factory=datetime.now)
    title: Optional[str] = None


class ScreenRouter:
    """URL-style router for TUI screen navigation.

    Provides path-based navigation with:
    - Route registration with path parameters
    - Navigation history
    - Modal/overlay screen stacking
    - Route guards
    - Event emission for navigation tracking
    """

    def __init__(self, app: Optional["App"] = None):
        self._app = app
        self._routes: Dict[str, Route] = {}
        self._route_list: List[Route] = []  # For ordered matching
        self._history: List[NavigationEntry] = []
        self._forward_stack: List[NavigationEntry] = []
        self._current: Optional[NavigationEntry] = None
        self._modal_stack: List[str] = []
        self._max_history: int = 50
        self._default_route: str = "/dashboard"

    def set_app(self, app: "App") -> None:
        """Set the Textual app reference.

        Args:
            app: The Textual App instance
        """
        self._app = app

    def register(
        self,
        path: str,
        screen_class: Type["Screen"],
        name: Optional[str] = None,
        title: Optional[str] = None,
        icon: Optional[str] = None,
        guard: Optional[Callable[[RouteContext], bool]] = None,
        is_modal: bool = False,
        **metadata: Any,
    ) -> Route:
        """Register a route.

        Args:
            path: URL-like path pattern (e.g., "/chat/:agent_id")
            screen_class: The Screen class to instantiate
            name: Optional route name for reference
            title: Display title for the route
            icon: Icon for navigation UI
            guard: Optional function to check if navigation is allowed
            is_modal: If True, screen is pushed as a modal overlay
            **metadata: Additional metadata for the route

        Returns:
            The created Route object
        """
        route = Route(
            path=path,
            screen_class=screen_class,
            name=name or path,
            title=title,
            icon=icon,
            guard=guard,
            is_modal=is_modal,
            metadata=metadata,
        )

        self._routes[path] = route
        self._route_list.append(route)

        logger.debug(f"ScreenRouter: Registered route '{path}' -> {screen_class.__name__}")
        return route

    def unregister(self, path: str) -> bool:
        """Unregister a route.

        Args:
            path: The path to unregister

        Returns:
            True if route was found and removed
        """
        if path in self._routes:
            route = self._routes[path]
            del self._routes[path]
            self._route_list.remove(route)
            return True
        return False

    def get_route(self, path: str) -> Optional[Route]:
        """Get the route for a path.

        Args:
            path: The path to look up

        Returns:
            The matching Route or None
        """
        # Try exact match first
        if path in self._routes:
            return self._routes[path]

        # Try pattern matching
        for route in self._route_list:
            if route.match(path):
                return route

        return None

    async def navigate(
        self,
        path: str,
        query: Optional[Dict[str, str]] = None,
        replace: bool = False,
    ) -> bool:
        """Navigate to a path.

        Args:
            path: The path to navigate to
            query: Optional query parameters
            replace: If True, replace current history entry instead of pushing

        Returns:
            True if navigation succeeded
        """
        if not self._app:
            logger.error("ScreenRouter: No app set")
            return False

        query = query or {}

        # Find matching route
        route = None
        params = {}
        for r in self._route_list:
            match_params = r.match(path)
            if match_params is not None:
                route = r
                params = match_params
                break

        if not route:
            logger.warning(f"ScreenRouter: No route for '{path}'")
            return False

        # Create context
        context = RouteContext(
            path=path,
            params=params,
            query=query,
            previous_path=self._current.path if self._current else None,
            is_modal=route.is_modal,
            metadata=route.metadata,
        )

        # Check guard
        if route.guard and not route.guard(context):
            logger.info(f"ScreenRouter: Guard blocked navigation to '{path}'")
            return False

        # Create navigation entry
        entry = NavigationEntry(
            path=path,
            params=params,
            query=query,
            title=route.title,
        )

        # Update history
        if replace and self._history:
            self._history[-1] = entry
        else:
            if self._current:
                self._history.append(self._current)
                if len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history:]

        self._current = entry
        self._forward_stack.clear()  # Clear forward stack on new navigation

        # Create and show screen
        try:
            # Create screen with params
            screen = self._create_screen(route, context)

            if route.is_modal:
                self._modal_stack.append(path)
                self._app.push_screen(screen)
            else:
                # Clear modal stack when navigating to non-modal
                self._modal_stack.clear()
                # Use push_screen if no screens on stack, otherwise switch_screen
                if len(self._app.screen_stack) == 0:
                    self._app.push_screen(screen)
                else:
                    self._app.switch_screen(screen)

            # Emit navigation event
            self._emit_navigation_event(path, params, query)

            logger.info(f"ScreenRouter: Navigated to '{path}'")
            return True

        except Exception as e:
            logger.error(f"ScreenRouter: Failed to navigate to '{path}': {e}")
            return False

    def _create_screen(self, route: Route, context: RouteContext) -> "Screen":
        """Create a screen instance from a route.

        Override this method to customize screen creation.
        """
        # Try to pass params and query to screen constructor
        try:
            return route.screen_class(**context.params, **context.query)
        except TypeError:
            # Fall back to no-argument constructor
            return route.screen_class()

    def _emit_navigation_event(
        self,
        path: str,
        params: Dict[str, str],
        query: Dict[str, str],
    ) -> None:
        """Emit a navigation event to the event bus."""
        from hafs.ui.core.event_bus import NavigationEvent, get_event_bus

        bus = get_event_bus()
        bus.publish(NavigationEvent(
            action="navigate",
            path=path,
            params={"path_params": params, "query": query},
            previous_path=self._history[-1].path if self._history else None,
        ))

    async def push(
        self,
        path: str,
        query: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Push a modal/overlay screen.

        Args:
            path: The path to push
            query: Optional query parameters

        Returns:
            True if push succeeded
        """
        route = self.get_route(path)
        if route:
            # Force modal mode
            route.is_modal = True
        return await self.navigate(path, query)

    async def pop(self) -> bool:
        """Pop the current modal screen.

        Returns:
            True if there was a modal to pop
        """
        if not self._app:
            return False

        if self._modal_stack:
            self._modal_stack.pop()
            self._app.pop_screen()
            return True

        return False

    async def back(self) -> bool:
        """Navigate back in history.

        Returns:
            True if navigation succeeded
        """
        if not self._history:
            return False

        # Save current for forward stack
        if self._current:
            self._forward_stack.append(self._current)

        # Pop from history
        entry = self._history.pop()
        self._current = entry

        # Navigate without adding to history
        return await self._navigate_to_entry(entry)

    async def forward(self) -> bool:
        """Navigate forward in history.

        Returns:
            True if navigation succeeded
        """
        if not self._forward_stack:
            return False

        # Save current for history
        if self._current:
            self._history.append(self._current)

        # Pop from forward stack
        entry = self._forward_stack.pop()
        self._current = entry

        return await self._navigate_to_entry(entry)

    async def _navigate_to_entry(self, entry: NavigationEntry) -> bool:
        """Navigate to a history entry without modifying history."""
        route = self.get_route(entry.path)
        if not route:
            return False

        context = RouteContext(
            path=entry.path,
            params=entry.params,
            query=entry.query,
            previous_path=None,
            is_modal=route.is_modal,
        )

        try:
            screen = self._create_screen(route, context)
            if route.is_modal:
                self._app.push_screen(screen)
            else:
                self._app.switch_screen(screen)
            return True
        except Exception as e:
            logger.error(f"ScreenRouter: Failed to navigate: {e}")
            return False

    def get_current_path(self) -> str:
        """Get the current path.

        Returns:
            Current path or default route
        """
        return self._current.path if self._current else self._default_route

    def get_current_params(self) -> Dict[str, str]:
        """Get current path parameters.

        Returns:
            Dict of path parameters
        """
        return self._current.params if self._current else {}

    def get_current_query(self) -> Dict[str, str]:
        """Get current query parameters.

        Returns:
            Dict of query parameters
        """
        return self._current.query if self._current else {}

    def get_history(self, limit: int = 10) -> List[NavigationEntry]:
        """Get navigation history.

        Args:
            limit: Maximum entries to return

        Returns:
            List of history entries (newest first)
        """
        return list(reversed(self._history[-limit:]))

    def can_go_back(self) -> bool:
        """Check if back navigation is possible."""
        return len(self._history) > 0

    def can_go_forward(self) -> bool:
        """Check if forward navigation is possible."""
        return len(self._forward_stack) > 0

    def get_all_routes(self) -> List[Route]:
        """Get all registered routes.

        Returns:
            List of all routes
        """
        return list(self._route_list)

    def get_navigation_items(self) -> List[Dict[str, Any]]:
        """Get navigation items for menu/sidebar.

        Returns:
            List of dicts with path, title, icon, active status
        """
        items = []
        current_path = self.get_current_path()

        for route in self._route_list:
            if route.is_modal:
                continue  # Don't show modals in nav

            # Only show non-parameterized routes
            if ":" not in route.path:
                items.append({
                    "path": route.path,
                    "title": route.title or route.name or route.path,
                    "icon": route.icon,
                    "active": current_path == route.path,
                })

        return items

    def clear_history(self) -> None:
        """Clear navigation history."""
        self._history.clear()
        self._forward_stack.clear()

    def set_default_route(self, path: str) -> None:
        """Set the default route.

        Args:
            path: The default path to use
        """
        self._default_route = path


# Global screen router instance
_global_router: Optional[ScreenRouter] = None


def get_screen_router() -> ScreenRouter:
    """Get the global screen router instance."""
    global _global_router
    if _global_router is None:
        _global_router = ScreenRouter()
    return _global_router


def reset_screen_router() -> None:
    """Reset the global screen router (for testing)."""
    global _global_router
    _global_router = None


def register_default_routes(router: ScreenRouter, use_modular: bool = True) -> None:
    """Register default routes for HAFS TUI.

    This should be called after screen classes are imported.

    Args:
        router: The ScreenRouter instance
        use_modular: If True, use new modular screens (DashboardScreen, ChatScreen).
                     If False, use legacy screens (MainScreen, OrchestratorScreen).
    """
    # Import screens here to avoid circular imports
    from hafs.ui.screens.logs import LogsScreen
    from hafs.ui.screens.settings import SettingsScreen
    from hafs.ui.screens.services import ServicesScreen

    if use_modular:
        # Use new modular screens
        from hafs.ui.screens.dashboard import DashboardScreen
        from hafs.ui.screens.chat import ChatScreen

        router.register(
            "/dashboard",
            DashboardScreen,
            name="dashboard",
            title="Dashboard",
            icon="",
        )

        router.register(
            "/chat",
            ChatScreen,
            name="chat",
            title="Chat",
            icon="",
        )
    else:
        # Use legacy screens
        from hafs.ui.screens.main import MainScreen
        from hafs.ui.screens.orchestrator import OrchestratorScreen

        router.register(
            "/dashboard",
            MainScreen,
            name="dashboard",
            title="Dashboard",
            icon="",
        )

        router.register(
            "/chat",
            OrchestratorScreen,
            name="chat",
            title="Chat",
            icon="",
        )

    router.register(
        "/logs",
        LogsScreen,
        name="logs",
        title="Logs",
        icon="",
    )

    router.register(
        "/settings",
        SettingsScreen,
        name="settings",
        title="Settings",
        icon="",
    )

    router.register(
        "/services",
        ServicesScreen,
        name="services",
        title="Services",
        icon="",
    )

    # Register new Sprint 4/6 screens
    if use_modular:
        from hafs.ui.screens.analysis_screen import AnalysisDashboardScreen
        from hafs.ui.screens.config_screen import ConfigScreen

        router.register(
            "/analysis",
            AnalysisDashboardScreen,
            name="analysis",
            title="Analysis",
            icon="",
        )

        router.register(
            "/config",
            ConfigScreen,
            name="config",
            title="Configuration",
            icon="",
        )

    logger.info(f"ScreenRouter: Registered {len(router.get_all_routes())} default routes (modular={use_modular})")
