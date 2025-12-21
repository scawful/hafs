"""PTY subprocess wrapper for CLI tool backends."""

from __future__ import annotations

import asyncio
import errno
import os
import pty
import signal
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class PtyOptions:
    """Configuration options for PTY subprocess."""

    rows: int = 24
    cols: int = 80
    env: dict[str, str] = field(default_factory=dict)
    working_dir: str | None = None
    buffer_size: int = 4096
    output_queue_maxsize: int = 1000


class PtyWrapper:
    """PTY subprocess wrapper for CLI tool backends.

    Provides async PTY creation and management for spawning interactive
    CLI tools like gemini, claude, codex.

    Example:
        pty = PtyWrapper(["gemini"])
        await pty.start()
        await pty.write("Hello, world!")
        async for chunk in pty.read_output():
            print(chunk, end="")
        await pty.terminate()
    """

    def __init__(
        self,
        command: list[str],
        options: PtyOptions | None = None,
    ):
        """Initialize PTY wrapper.

        Args:
            command: Command and arguments to spawn.
            options: PTY configuration options.
        """
        self.command = command
        self.options = options or PtyOptions()

        self._master_fd: int | None = None
        self._pid: int | None = None
        self._output_queue: asyncio.Queue[str] = asyncio.Queue(
            maxsize=self.options.output_queue_maxsize
        )
        self._running = False
        self._reader_task: asyncio.Task[None] | None = None
        self._exit_code: int | None = None
        self._on_output: Callable[[str], None] | None = None
        self._on_exit: Callable[[int], None] | None = None

    async def start(self) -> bool:
        """Fork PTY and start subprocess.

        Returns:
            True if subprocess started successfully.
        """
        if self._running:
            return True

        try:
            # Fork PTY
            pid, master_fd = pty.fork()

            if pid == 0:
                # Child process
                if self.options.working_dir:
                    os.chdir(self.options.working_dir)

                # Set environment
                env = os.environ.copy()
                env.update(self.options.env)

                # Use a simpler terminal type to reduce escape sequence complexity
                # This helps reduce flickering from mouse reporting, etc.
                env["TERM"] = "xterm"
                # Disable mouse reporting in some applications
                env["DISABLE_MOUSE"] = "1"

                # Execute command
                os.execvpe(self.command[0], self.command, env)
            else:
                # Parent process
                self._pid = pid
                self._master_fd = master_fd
                self._running = True

                # Set non-blocking
                os.set_blocking(master_fd, False)

                # Start reader task
                self._reader_task = asyncio.create_task(self._read_loop())

                return True

        except OSError as e:
            self._running = False
            raise RuntimeError(f"Failed to spawn PTY: {e}") from e

        return False

    async def _read_loop(self) -> None:
        """Background task to read PTY output."""

        while self._running and self._master_fd is not None:
            try:
                # Wait for data to be available
                await asyncio.sleep(0.01)  # Small delay to prevent busy-waiting

                try:
                    data = os.read(self._master_fd, self.options.buffer_size)
                    if data:
                        text = data.decode("utf-8", errors="replace")

                        # Queue for async iteration
                        try:
                            self._output_queue.put_nowait(text)
                        except asyncio.QueueFull:
                            # Drop oldest if queue is full
                            try:
                                self._output_queue.get_nowait()
                                self._output_queue.put_nowait(text)
                            except asyncio.QueueEmpty:
                                pass

                        # Call callback if set
                        if self._on_output:
                            self._on_output(text)
                    else:
                        # EOF - process exited
                        break
                except OSError as e:
                    if e.errno == errno.EAGAIN:
                        # No data available, continue
                        continue
                    elif e.errno == errno.EIO:
                        # PTY closed
                        break
                    else:
                        raise

            except asyncio.CancelledError:
                break

        # Clean up
        self._running = False
        await self._wait_for_exit()

    async def _wait_for_exit(self) -> None:
        """Wait for subprocess to exit and capture exit code."""
        if self._pid is not None:
            try:
                _, status = os.waitpid(self._pid, os.WNOHANG)
                if os.WIFEXITED(status):
                    self._exit_code = os.WEXITSTATUS(status)
                elif os.WIFSIGNALED(status):
                    self._exit_code = -os.WTERMSIG(status)
                else:
                    self._exit_code = -1

                if self._on_exit:
                    self._on_exit(self._exit_code)
            except ChildProcessError:
                self._exit_code = -1

    async def write(self, data: str) -> None:
        """Write to PTY stdin.

        Args:
            data: String data to write.
        """
        if not self._running or self._master_fd is None:
            raise RuntimeError("PTY not running")

        try:
            os.write(self._master_fd, data.encode("utf-8"))
        except OSError as e:
            raise RuntimeError(f"Failed to write to PTY: {e}") from e

    async def read_output(self) -> AsyncIterator[str]:
        """Stream output from PTY.

        Yields:
            Output chunks as they become available.
        """
        while self._running or not self._output_queue.empty():
            try:
                chunk = await asyncio.wait_for(
                    self._output_queue.get(),
                    timeout=0.1,
                )
                yield chunk
            except asyncio.TimeoutError:
                if not self._running:
                    break
                continue

    async def terminate(self, timeout: float = 5.0) -> int:
        """Gracefully terminate subprocess.

        Args:
            timeout: Seconds to wait before SIGKILL.

        Returns:
            Process exit code.
        """
        if not self._running:
            return self._exit_code or 0

        if self._pid is not None:
            # Try SIGTERM first
            try:
                os.kill(self._pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

            # Wait for exit
            try:
                await asyncio.wait_for(
                    self._wait_for_termination(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                # Force kill
                try:
                    os.kill(self._pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

        # Cancel reader task
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        # Close master fd
        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError:
                pass
            self._master_fd = None

        self._running = False
        return self._exit_code or 0

    async def _wait_for_termination(self) -> None:
        """Wait for process to terminate."""
        while self._running:
            await asyncio.sleep(0.1)

    def resize(self, rows: int, cols: int) -> None:
        """Resize PTY window.

        Args:
            rows: New number of rows.
            cols: New number of columns.
        """
        if self._master_fd is not None:
            import fcntl
            import struct
            import termios

            winsize = struct.pack("HHHH", rows, cols, 0, 0)
            fcntl.ioctl(self._master_fd, termios.TIOCSWINSZ, winsize)

    def set_output_callback(self, callback: Callable[[str], None] | None) -> None:
        """Set callback for output data.

        Args:
            callback: Function called with each output chunk.
        """
        self._on_output = callback

    def set_exit_callback(self, callback: Callable[[int], None] | None) -> None:
        """Set callback for process exit.

        Args:
            callback: Function called with exit code.
        """
        self._on_exit = callback

    @property
    def is_running(self) -> bool:
        """Check if subprocess is alive."""
        return self._running

    @property
    def pid(self) -> int | None:
        """Get subprocess PID."""
        return self._pid

    @property
    def exit_code(self) -> int | None:
        """Get exit code (None if still running)."""
        return self._exit_code

    # Special key sequences for terminal control

    KEY_MAP = {
        "ctrl+c": "\x03",  # ETX - Interrupt
        "ctrl+d": "\x04",  # EOT - End of transmission
        "ctrl+y": "\x19",  # EM - YOLO mode for Gemini-CLI
        "ctrl+z": "\x1a",  # SUB - Suspend
        "shift+tab": "\x1b[Z",  # Reverse tab
        "tab": "\t",
        "enter": "\n",
        "escape": "\x1b",
        "up": "\x1b[A",
        "down": "\x1b[B",
        "left": "\x1b[C",
        "right": "\x1b[D",
    }

    def send_key(self, key: str) -> None:
        """Send a special key sequence to the PTY.

        Args:
            key: Key name (e.g., "ctrl+c", "ctrl+y", "shift+tab").
        """
        if not self._running or self._master_fd is None:
            return

        sequence = self.KEY_MAP.get(key.lower())
        if sequence:
            try:
                os.write(self._master_fd, sequence.encode("utf-8"))
            except OSError:
                pass

    def send_interrupt(self) -> None:
        """Send Ctrl+C (SIGINT) to the PTY."""
        self.send_key("ctrl+c")

    def send_suspend(self) -> None:
        """Send Ctrl+Z (SIGTSTP) to the PTY."""
        self.send_key("ctrl+z")
