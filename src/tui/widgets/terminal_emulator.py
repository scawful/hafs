"""Terminal emulator widget using pyte for proper terminal emulation."""

from __future__ import annotations

import asyncio
import os
import pty
import signal
from typing import Any, Callable

import pyte
from rich.console import RenderableType
from rich.style import Style
from rich.text import Text
from textual.reactive import reactive
from textual.widget import Widget


class TerminalDisplay(Widget, can_focus=True):
    """A terminal display widget that renders PTY output using pyte.

    This widget does NOT spawn its own process. Instead, it receives raw
    terminal data via the `feed()` method and renders it properly using
    pyte for terminal emulation.

    Use this widget when you have an existing PTY process managed elsewhere
    (e.g., by a backend) and need to display its output.

    Example:
        display = TerminalDisplay(rows=24, cols=80)
        display.set_write_callback(pty_write_func)
        # In your PTY output callback:
        display.feed(raw_output)
    """

    DEFAULT_CSS = """
    TerminalDisplay {
        width: 100%;
        height: 100%;
        background: $surface;
        padding: 0;
    }

    TerminalDisplay:focus {
        border: solid $accent;
    }
    """

    # Color mapping from pyte to Rich
    PYTE_COLORS = {
        "black": "black",
        "red": "red",
        "green": "green",
        "brown": "yellow",
        "blue": "blue",
        "magenta": "magenta",
        "cyan": "cyan",
        "white": "white",
        "default": "default",
    }

    # Key mapping for special keys
    KEY_MAP = {
        "enter": "\r",
        "tab": "\t",
        "backspace": "\x7f",
        "delete": "\x1b[3~",
        "escape": "\x1b",
        "up": "\x1b[A",
        "down": "\x1b[B",
        "right": "\x1b[C",
        "left": "\x1b[D",
        "home": "\x1b[H",
        "end": "\x1b[F",
        "pageup": "\x1b[5~",
        "pagedown": "\x1b[6~",
        "insert": "\x1b[2~",
        "f1": "\x1bOP",
        "f2": "\x1bOQ",
        "f3": "\x1bOR",
        "f4": "\x1bOS",
        "f5": "\x1b[15~",
        "f6": "\x1b[17~",
        "f7": "\x1b[18~",
        "f8": "\x1b[19~",
        "f9": "\x1b[20~",
        "f10": "\x1b[21~",
        "f11": "\x1b[23~",
        "f12": "\x1b[24~",
    }

    def __init__(
        self,
        rows: int = 24,
        cols: int = 80,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize terminal display.

        Args:
            rows: Initial terminal rows.
            cols: Initial terminal columns.
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._rows = rows
        self._cols = cols

        # Pyte terminal emulation - use DiffScreen for dirty tracking
        self._screen = pyte.Screen(cols, rows)
        self._stream = pyte.Stream(self._screen)

        # Input callback for sending keystrokes to PTY
        self._write_callback: Callable[[str], None] | None = None

        # Refresh throttling to reduce flicker
        self._pending_refresh = False
        self._refresh_timer: asyncio.TimerHandle | None = None
        self._min_refresh_interval = 0.05  # 50ms = 20 FPS max (more stable)

        # Track last rendered content to avoid unnecessary redraws
        self._last_buffer_hash: int = 0

    def set_write_callback(self, callback: Callable[[str], None] | None) -> None:
        """Set callback for writing input to the PTY.

        Args:
            callback: Function that writes string data to the PTY.
        """
        self._write_callback = callback

    def write(self, data: str) -> None:
        """Write data to the PTY (send input).

        Args:
            data: String to write to PTY stdin.
        """
        if self._write_callback:
            self._write_callback(data)

    # Mouse reporting and other sequences to filter out
    # These cause flicker without adding visual content
    FILTER_PATTERNS = [
        "\x1b[?1000h",  # Enable mouse tracking
        "\x1b[?1000l",  # Disable mouse tracking
        "\x1b[?1002h",  # Enable mouse button tracking
        "\x1b[?1002l",  # Disable mouse button tracking
        "\x1b[?1003h",  # Enable all mouse tracking
        "\x1b[?1003l",  # Disable all mouse tracking
        "\x1b[?1006h",  # Enable SGR mouse mode
        "\x1b[?1006l",  # Disable SGR mouse mode
        "\x1b[?25h",    # Show cursor
        "\x1b[?25l",    # Hide cursor
        "\x1b[?1049h",  # Enable alternate screen buffer
        "\x1b[?1049l",  # Disable alternate screen buffer
        "\x1b[?2004h",  # Enable bracketed paste
        "\x1b[?2004l",  # Disable bracketed paste
    ]

    def feed(self, data: str) -> None:
        """Feed raw terminal data to the display.

        This method accepts raw PTY output (including ANSI escape sequences)
        and processes it through pyte for proper terminal emulation.

        Args:
            data: Raw terminal output string.
        """
        # Filter out mouse reporting and other mode-switching sequences
        # that cause unnecessary redraws
        filtered_data = data
        for pattern in self.FILTER_PATTERNS:
            filtered_data = filtered_data.replace(pattern, "")

        if filtered_data:
            self._stream.feed(filtered_data)
            # Schedule a throttled refresh to reduce flicker
            self._schedule_refresh()

    def _schedule_refresh(self) -> None:
        """Schedule a throttled refresh."""
        if not self._pending_refresh:
            self._pending_refresh = True
            # Use set_timer for better timing control
            self.set_timer(self._min_refresh_interval, self._do_refresh)

    def _do_refresh(self) -> None:
        """Perform the actual refresh after throttle delay."""
        self._pending_refresh = False

        # Calculate a hash of the visible content to skip redundant redraws
        content_hash = self._compute_buffer_hash()
        if content_hash != self._last_buffer_hash:
            self._last_buffer_hash = content_hash
            self.refresh()

    def _compute_buffer_hash(self) -> int:
        """Compute a hash of the current screen buffer content."""
        # Build a simple hash from visible characters
        content_parts = []
        for y in range(self._screen.lines):
            line_chars = []
            for x in range(self._screen.columns):
                char = self._screen.buffer[y][x]
                line_chars.append(char.data or " ")
            content_parts.append("".join(line_chars).rstrip())
        return hash(tuple(content_parts))

    def resize(self, rows: int, cols: int) -> None:
        """Resize the terminal screen.

        Args:
            rows: New number of rows.
            cols: New number of columns.
        """
        if rows != self._rows or cols != self._cols:
            self._rows = rows
            self._cols = cols
            self._screen.resize(rows, cols)

    def clear(self) -> None:
        """Clear the terminal screen."""
        self._screen.reset()
        self.refresh()

    def render(self) -> RenderableType:
        """Render the terminal screen."""
        lines = []

        for y in range(self._screen.lines):
            line = Text()
            for x in range(self._screen.columns):
                char = self._screen.buffer[y][x]

                # Build style from character attributes
                style_parts = []

                # Foreground color
                if char.fg != "default":
                    if char.fg in self.PYTE_COLORS:
                        style_parts.append(self.PYTE_COLORS[char.fg])
                    elif isinstance(char.fg, str) and len(char.fg) == 6:
                        style_parts.append(f"#{char.fg}")

                # Background color
                if char.bg != "default":
                    if char.bg in self.PYTE_COLORS:
                        style_parts.append(f"on {self.PYTE_COLORS[char.bg]}")
                    elif isinstance(char.bg, str) and len(char.bg) == 6:
                        style_parts.append(f"on #{char.bg}")

                # Text styles
                if char.bold:
                    style_parts.append("bold")
                if char.italics:
                    style_parts.append("italic")
                if char.underscore:
                    style_parts.append("underline")
                if char.reverse:
                    style_parts.append("reverse")

                style = Style.parse(" ".join(style_parts)) if style_parts else None
                line.append(char.data or " ", style=style)

            lines.append(line)

        result = Text("\n").join(lines)
        return result

    def on_resize(self, event) -> None:
        """Handle widget resize."""
        new_cols = max(20, event.size.width)
        new_rows = max(5, event.size.height)
        self.resize(new_rows, new_cols)

    def on_key(self, event) -> None:
        """Handle key presses and forward to PTY."""
        if not self._write_callback:
            return

        key = event.key

        # Handle ctrl+ combinations
        if key.startswith("ctrl+"):
            char = key[5:]
            if len(char) == 1 and char.isalpha():
                # Convert to control character (ctrl+a = \x01, ctrl+b = \x02, etc.)
                ctrl_char = chr(ord(char.upper()) - 64)
                self.write(ctrl_char)
                event.stop()
                return

        # Handle special keys
        if key in self.KEY_MAP:
            self.write(self.KEY_MAP[key])
            event.stop()
            return

        # Handle regular printable characters
        if event.character and len(event.character) == 1:
            self.write(event.character)
            event.stop()
            return

    def on_paste(self, event) -> None:
        """Handle paste events."""
        if self._write_callback and event.text:
            self.write(event.text)
            event.stop()


class TerminalEmulator(Widget, can_focus=True):
    """A proper terminal emulator widget using pyte.

    This widget spawns a PTY subprocess and renders its output using pyte
    for accurate terminal emulation including cursor positioning, colors,
    and screen clearing.

    Example:
        terminal = TerminalEmulator(command="gemini")
        await terminal.start()
    """

    DEFAULT_CSS = """
    TerminalEmulator {
        width: 100%;
        height: 100%;
        background: $surface;
        padding: 0;
    }

    TerminalEmulator:focus {
        border: solid $accent;
    }
    """

    is_running: reactive[bool] = reactive(False)

    # Color mapping from pyte to Rich
    PYTE_COLORS = {
        "black": "black",
        "red": "red",
        "green": "green",
        "brown": "yellow",
        "blue": "blue",
        "magenta": "magenta",
        "cyan": "cyan",
        "white": "white",
        "default": "default",
    }

    def __init__(
        self,
        command: str | list[str] | None = None,
        rows: int = 24,
        cols: int = 80,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize terminal emulator.

        Args:
            command: Command to run (string or list of args).
            rows: Initial terminal rows.
            cols: Initial terminal columns.
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._command = (
            command if isinstance(command, list) else ([command] if command else ["bash"])
        )
        self._rows = rows
        self._cols = cols

        # Pyte terminal emulation
        self._screen = pyte.Screen(cols, rows)
        self._stream = pyte.Stream(self._screen)

        # PTY process
        self._master_fd: int | None = None
        self._pid: int | None = None
        self._read_task: asyncio.Task[Any] | None = None

        # Callbacks
        self._on_exit: Callable[[int], None] | None = None

    async def start(self) -> bool:
        """Start the terminal subprocess.

        Returns:
            True if started successfully.
        """
        if self.is_running:
            return True

        try:
            # Fork PTY
            pid, master_fd = pty.fork()

            if pid == 0:
                # Child process - exec the command
                os.execvp(self._command[0], self._command)
            else:
                # Parent process
                self._pid = pid
                self._master_fd = master_fd
                self.is_running = True

                # Set non-blocking
                os.set_blocking(master_fd, False)

                # Set terminal size
                self._set_pty_size()

                # Start reading output
                self._read_task = asyncio.create_task(self._read_loop())

                return True

        except OSError as e:
            self.log.error(f"Failed to start terminal: {e}")
            return False

        return False

    def _set_pty_size(self) -> None:
        """Set the PTY window size."""
        if self._master_fd is not None:
            import fcntl
            import struct
            import termios

            winsize = struct.pack("HHHH", self._rows, self._cols, 0, 0)
            fcntl.ioctl(self._master_fd, termios.TIOCSWINSZ, winsize)

    async def _read_loop(self) -> None:
        """Background task to read PTY output and feed to pyte."""
        import errno

        while self.is_running and self._master_fd is not None:
            try:
                await asyncio.sleep(0.01)

                try:
                    data = os.read(self._master_fd, 4096)
                    if data:
                        # Feed to pyte stream for terminal emulation
                        self._stream.feed(data.decode("utf-8", errors="replace"))
                        # Trigger re-render
                        self.refresh()
                    else:
                        # EOF - process exited
                        break
                except OSError as e:
                    if e.errno == errno.EAGAIN:
                        continue
                    elif e.errno == errno.EIO:
                        break
                    else:
                        raise

            except asyncio.CancelledError:
                break

        self.is_running = False
        self._wait_for_exit()

    def _wait_for_exit(self) -> None:
        """Wait for subprocess to exit."""
        if self._pid is not None:
            try:
                _, status = os.waitpid(self._pid, os.WNOHANG)
                exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
                if self._on_exit:
                    self._on_exit(exit_code)
            except ChildProcessError:
                pass

    async def stop(self) -> None:
        """Stop the terminal subprocess."""
        if not self.is_running:
            return

        if self._pid is not None:
            try:
                os.kill(self._pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass

        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError:
                pass
            self._master_fd = None

        self.is_running = False

    def write(self, data: str) -> None:
        """Write data to the terminal (send input).

        Args:
            data: String to write to PTY stdin.
        """
        if self._master_fd is not None and self.is_running:
            try:
                os.write(self._master_fd, data.encode("utf-8"))
            except OSError:
                pass

    def send_key(self, key: str) -> None:
        """Send a special key to the terminal.

        Args:
            key: Key name (e.g., "ctrl+c", "ctrl+y", "shift+tab").
        """
        key_map = {
            "ctrl+c": "\x03",
            "ctrl+d": "\x04",
            "ctrl+y": "\x19",
            "ctrl+z": "\x1a",
            "shift+tab": "\x1b[Z",
            "tab": "\t",
            "enter": "\n",
            "escape": "\x1b",
            "up": "\x1b[A",
            "down": "\x1b[B",
            "left": "\x1b[C",
            "right": "\x1b[D",
            "backspace": "\x7f",
            "delete": "\x1b[3~",
        }
        sequence = key_map.get(key.lower())
        if sequence:
            self.write(sequence)

    def render(self) -> RenderableType:
        """Render the terminal screen."""
        lines = []

        for y in range(self._screen.lines):
            line = Text()
            for x in range(self._screen.columns):
                char = self._screen.buffer[y][x]

                # Build style from character attributes
                style_parts = []

                # Foreground color
                if char.fg != "default":
                    if char.fg in self.PYTE_COLORS:
                        style_parts.append(self.PYTE_COLORS[char.fg])
                    elif isinstance(char.fg, str) and len(char.fg) == 6:
                        # Hex color
                        style_parts.append(f"#{char.fg}")

                # Background color
                if char.bg != "default":
                    if char.bg in self.PYTE_COLORS:
                        style_parts.append(f"on {self.PYTE_COLORS[char.bg]}")
                    elif isinstance(char.bg, str) and len(char.bg) == 6:
                        style_parts.append(f"on #{char.bg}")

                # Text styles
                if char.bold:
                    style_parts.append("bold")
                if char.italics:
                    style_parts.append("italic")
                if char.underscore:
                    style_parts.append("underline")
                if char.reverse:
                    style_parts.append("reverse")

                style = Style.parse(" ".join(style_parts)) if style_parts else None
                line.append(char.data or " ", style=style)

            lines.append(line)

        # Join lines with newlines
        result = Text("\n").join(lines)
        return result

    def on_resize(self, event) -> None:
        """Handle widget resize."""
        # Update terminal size based on widget size
        # This is approximate - actual calculation depends on font metrics
        new_cols = max(20, event.size.width)
        new_rows = max(5, event.size.height)

        if new_cols != self._cols or new_rows != self._rows:
            self._cols = new_cols
            self._rows = new_rows
            self._screen.resize(new_rows, new_cols)
            self._set_pty_size()

    def on_key(self, event) -> None:
        """Handle key presses."""
        if not self.is_running:
            return

        # Map Textual keys to terminal input
        key = event.key

        if len(key) == 1:
            # Regular character
            self.write(key)
        elif key == "enter":
            self.write("\n")
        elif key == "tab":
            self.write("\t")
        elif key == "backspace":
            self.write("\x7f")
        elif key == "delete":
            self.write("\x1b[3~")
        elif key == "escape":
            self.write("\x1b")
        elif key == "up":
            self.write("\x1b[A")
        elif key == "down":
            self.write("\x1b[B")
        elif key == "left":
            self.write("\x1b[C")
        elif key == "right":
            self.write("\x1b[D")
        elif key.startswith("ctrl+"):
            # Handle ctrl combinations
            char = key[5:]
            if len(char) == 1:
                # Convert to control character
                ctrl_char = chr(ord(char.upper()) - 64)
                self.write(ctrl_char)

        event.stop()

    def set_exit_callback(self, callback: Callable[[int], None]) -> None:
        """Set callback for process exit.

        Args:
            callback: Function called with exit code.
        """
        self._on_exit = callback

    def clear(self) -> None:
        """Clear the terminal screen."""
        self._screen.reset()
        self.refresh()
