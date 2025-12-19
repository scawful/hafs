"""Spacemacs-style leader key / which-key mixin."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from textual.events import Key

from hafs.ui.widgets.which_key_bar import WhichKeyBar

WhichKeyAction = Callable[[], Any] | str
WhichKeyNode = dict[str, tuple[str, Mapping[str, Any] | WhichKeyAction] | WhichKeyAction]


class WhichKeyMixin:
    """Mixin that implements a leader key with which-key hints.

    Screens using this mixin should override `get_which_key_map`.
    """

    WHICH_KEY_TIMEOUT = 2.0

    def _ensure_which_key_state(self) -> None:
        """Initialize internal state lazily.

        Textual's Screen __init__ doesn't call mixin __init__,
        so we ensure state exists when first used.
        """
        if not hasattr(self, "_which_key_active"):
            self._which_key_active = False
            self._which_key_prefix = []
            self._which_key_node = {}
            self._which_key_timer = None

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
        self._which_key_active = True
        self._which_key_prefix = []
        self._which_key_node = self.get_which_key_map()
        self._reset_timer()
        self._update_bar()

    def _cancel_which_key(self) -> None:
        self._which_key_active = False
        self._which_key_prefix = []
        self._which_key_node = {}
        self._stop_timer()
        self._update_bar()

    def _reset_timer(self) -> None:
        self._stop_timer()
        if hasattr(self, "set_timer"):
            self._which_key_timer = self.set_timer(self.WHICH_KEY_TIMEOUT, self._cancel_which_key)

    def _stop_timer(self) -> None:
        timer = getattr(self, "_which_key_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
        self._which_key_timer = None

    def _handle_which_key_key(self, key: str) -> None:
        node = self._which_key_node
        if not isinstance(node, Mapping) or key not in node:
            self.notify(f"No binding for SPC {' '.join(self._which_key_prefix + [key])}", severity="warning", timeout=1)  # type: ignore[attr-defined]
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
            self._which_key_node = children
            self._reset_timer()
            self._update_bar()
            return

        self._invoke_target(target, label or key)
        self._cancel_which_key()

    def _invoke_target(self, target: WhichKeyAction | None, label: str) -> None:
        if target is None:
            return
        if callable(target):
            target()
            return

        # Treat as action name without "action_"
        action_name = str(target)
        if not action_name.startswith("action_"):
            action_name = f"action_{action_name}"

        if hasattr(self, action_name):
            getattr(self, action_name)()
            return

        if hasattr(self, "app") and hasattr(self.app, action_name):
            getattr(self.app, action_name)()
            return

        self.notify(f"Unbound action: {label}", severity="error", timeout=1)  # type: ignore[attr-defined]

    def _update_bar(self) -> None:
        try:
            bar = self.query_one(WhichKeyBar)
        except Exception:
            return

        if not self._which_key_active:
            bar.hide_hints()
            return

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
