"""Spacemacs-style leader key / which-key mixin.

Features:
- No timeout: stays open until dismissed with Escape
- Persistent abbreviated hints when not active
- Full hints when activated via Space
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, cast, TYPE_CHECKING

from textual.events import Key

if TYPE_CHECKING:
    from textual.app import App
    from textual.screen import Screen
    from textual.widget import Widget

from tui.widgets.which_key_bar import WhichKeyBar

WhichKeyAction = Callable[[], Any] | str
WhichKeyNode = dict[str, tuple[str, Mapping[str, Any] | WhichKeyAction] | WhichKeyAction]


class WhichKeyMixin:
    """Mixin that implements a leader key with which-key hints.

    Screens using this mixin should override `get_which_key_map`.
    """

    # Set to None to disable timeout entirely (stays open until Escape)
    WHICH_KEY_TIMEOUT: Optional[float] = None

    # Show abbreviated hints even when not active
    SHOW_PERSISTENT_HINTS: bool = True

    def _ensure_which_key_state(self) -> None:
        """Initialize internal state lazily.

        Textual's Screen __init__ doesn't call mixin __init__,
        so we ensure state exists when first used.
        """
        if not hasattr(self, "_which_key_active"):
            self._which_key_active = False
            self._which_key_prefix: list[str] = []
            self._which_key_node: WhichKeyNode = {}
            self._which_key_initialized = False

    def get_which_key_map(self) -> WhichKeyNode:
        """Return the root which-key map. Override in screens."""
        return {}

    def on_key(self, event: Key) -> None:  # type: ignore[override]
        self._ensure_which_key_state()
        if self._is_input_focused():
            return

        if not self._which_key_active:
            if event.key == "space":
                self._start_which_key()
                event.stop()
            return

        # Active leader mode
        if event.key in {"escape", "backspace"}:
            self._cancel_which_key()
            event.stop()
            return

        self._handle_which_key_key(event.key)
        event.stop()

    def _is_input_focused(self) -> bool:
        """Avoid hijacking leader key while typing."""
        try:
            from textual.widgets import Input, TextArea

            focused = getattr(self, "focused", None)
            return isinstance(focused, (Input, TextArea))
        except Exception:
            return False

    def _start_which_key(self) -> None:
        """Activate which-key mode."""
        self._which_key_active = True
        self._which_key_prefix = []
        self._which_key_node = self.get_which_key_map()
        self._update_bar()
        self._update_mode_indicator("which-key")

    def _cancel_which_key(self) -> None:
        """Deactivate which-key mode and return to idle state."""
        self._which_key_active = False
        self._which_key_prefix = []
        self._which_key_node = {}
        self._update_bar()
        self._update_mode_indicator("normal")

    def _handle_which_key_key(self, key: str) -> None:
        """Handle a keypress in which-key mode."""
        node = self._which_key_node
        if not isinstance(node, Mapping) or key not in node:
            self.notify(  # type: ignore[attr-defined]
                f"No binding for SPC {' '.join(self._which_key_prefix + [key])}",
                severity="warning",
                timeout=1,
            )
            self._cancel_which_key()
            return

        entry = node[key]

        # Normalize entry
        label: str = ""
        target: Any = None
        children: Mapping[str, Any] | None = None

        if isinstance(entry, tuple) and len(entry) == 2:
            label = entry[0]
            target = entry[1]
            if isinstance(target, Mapping):
                children = target
                target = None
        else:
            target = entry

        self._which_key_prefix.append(key)

        if children is not None:
            self._which_key_node = cast("WhichKeyNode", children)
            self._update_bar()
            return

        self._invoke_target(target, label or key)
        self._cancel_which_key()

    def _invoke_target(self, target: WhichKeyAction | None, label: str) -> None:
        """Execute the target action or callable."""
        import asyncio
        import inspect

        if target is None:
            return

        if callable(target):
            result = target()
            # Handle async functions - schedule them as tasks
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)
            return

        # Treat as action name without "action_"
        action_name = str(target)
        if not action_name.startswith("action_"):
            action_name = f"action_{action_name}"

        action = None
        if hasattr(self, action_name):
            action = getattr(self, action_name)
        elif hasattr(self, "app") and hasattr(self.app, action_name):
            action = getattr(self.app, action_name)

        if action:
            result = action()
            # Handle async actions
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)
            return

        self.notify(  # type: ignore[attr-defined]
            f"Unbound action: {label}",
            severity="error",
            timeout=1,
        )

    def _update_bar(self) -> None:
        """Update the WhichKeyBar with current state."""
        try:
            bar = self.query_one(WhichKeyBar)  # type: ignore[attr-defined]
        except Exception:
            return

        if not self._which_key_active:
            # Show abbreviated hints when idle (if enabled)
            if self.SHOW_PERSISTENT_HINTS:
                root_map = self.get_which_key_map()
                if root_map:
                    bar.show_abbreviated_hints(root_map)
                else:
                    bar.hide_hints()
            else:
                bar.hide_hints()
            return

        # Full hints when active
        hints: list[tuple[str, str]] = []
        node = self._which_key_node or {}
        if isinstance(node, Mapping):
            for k, v in node.items():
                if isinstance(v, tuple) and len(v) == 2:
                    hints.append((k, v[0]))
                else:
                    hints.append((k, str(v)))

        prefix = " ".join(self._which_key_prefix)
        bar.show_hints(prefix=prefix, hints=hints)

    def init_which_key_hints(self) -> None:
        """Initialize abbreviated hints on screen mount.

        Call this in on_mount() to show hints immediately.
        """
        self._ensure_which_key_state()
        if self.SHOW_PERSISTENT_HINTS and not self._which_key_initialized:
            self._which_key_initialized = True
            # Delay slightly to ensure bar is mounted
            if hasattr(self, "call_after_refresh"):
                self.call_after_refresh(self._update_bar)  # type: ignore[attr-defined]

    def _update_mode_indicator(self, mode: str) -> None:
        """Update the mode indicator widget if present.

        Args:
            mode: One of 'normal', 'insert', 'which-key', 'visual', 'command'
        """
        try:
            from tui.widgets.mode_indicator import ModeIndicator, InputMode

            indicator = self.query_one(ModeIndicator)  # type: ignore[attr-defined]

            mode_map = {
                "normal": InputMode.NORMAL,
                "insert": InputMode.INSERT,
                "which-key": InputMode.WHICH_KEY,
                "visual": InputMode.VISUAL,
                "command": InputMode.COMMAND,
            }

            if mode.lower() in mode_map:
                indicator.mode = mode_map[mode.lower()]
        except Exception:
            # Mode indicator not present, that's fine
            pass

