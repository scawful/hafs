"""Claude CLI backend using PTY subprocess."""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from pathlib import Path

from hafs.backends.base import BackendCapabilities, BaseChatBackend
from hafs.backends.pty import PtyOptions, PtyWrapper


class ClaudeResponseParser:
    """Parser for Claude CLI (claude-code) output format."""

    ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
    SPINNER_PATTERN = re.compile(r"[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏⣾⣽⣻⢿⡿⣟⣯⣷]")
    PROMPT_PATTERN = re.compile(r"^(❯|>|\$|claude>)\s*", re.MULTILINE)
    TOOL_CALL_PATTERN = re.compile(r"^(Reading|Writing|Running|Searching)", re.MULTILINE)

    def __init__(self) -> None:
        self._buffer = ""
        self._in_response = False

    def parse_chunk(self, chunk: str) -> str | None:
        """Parse a chunk of Claude CLI output.

        Args:
            chunk: Raw output chunk from PTY.

        Returns:
            Parsed response text, or None if not part of response.
        """
        # Strip ANSI escape codes
        text = self.ESCAPE_PATTERN.sub("", chunk)

        # Skip spinner characters
        text = self.SPINNER_PATTERN.sub("", text)

        # Skip prompt
        text = self.PROMPT_PATTERN.sub("", text)

        # Clean up whitespace
        text = text.strip()

        if text:
            return text
        return None

    def reset(self) -> None:
        """Reset parser state."""
        self._buffer = ""
        self._in_response = False


class ClaudeCliBackend(BaseChatBackend):
    """Claude CLI (claude-code) backend using PTY subprocess.

    Spawns the `claude` CLI tool and manages interaction via PTY.

    Example:
        backend = ClaudeCliBackend(project_dir=Path.cwd())
        await backend.start()
        await backend.send_message("Explain this code")
        async for chunk in backend.stream_response():
            print(chunk, end="")
        await backend.stop()
    """

    def __init__(
        self,
        project_dir: Path | None = None,
        command: str = "claude",
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ):
        """Initialize Claude CLI backend.

        Args:
            project_dir: Working directory for claude CLI.
            command: Command to run (default: "claude").
            extra_args: Additional CLI arguments.
            env: Environment variables to set.
        """
        self._project_dir = project_dir
        self._command = command
        self._extra_args = extra_args or []
        self._env = env or {}
        self._pty: PtyWrapper | None = None
        self._parser = ClaudeResponseParser()
        self._busy = False
        self._pending_context: str | None = None

    @property
    def name(self) -> str:
        return "claude"

    @property
    def display_name(self) -> str:
        return "Claude CLI"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_streaming=True,
            supports_tool_use=True,
            supports_images=True,
            supports_files=True,
            max_context_tokens=200000,
        )

    async def start(self) -> bool:
        """Start the Claude CLI process.

        Returns:
            True if started successfully.
        """
        if self._pty and self._pty.is_running:
            return True

        cmd = [self._command] + self._extra_args

        options = PtyOptions(
            env=self._env,
            working_dir=str(self._project_dir) if self._project_dir else None,
        )

        self._pty = PtyWrapper(cmd, options)

        try:
            success = await self._pty.start()
            if success:
                self._parser.reset()
            return success
        except RuntimeError:
            return False

    async def stop(self) -> None:
        """Stop the Claude CLI process."""
        if self._pty:
            await self._pty.terminate()
            self._pty = None
            self._parser.reset()

    async def send_message(self, message: str) -> None:
        """Send a message to Claude CLI.

        Args:
            message: The message to send.

        Raises:
            RuntimeError: If backend is not running.
        """
        if not self._pty or not self._pty.is_running:
            raise RuntimeError("Claude backend not running")

        self._busy = True
        self._parser.reset()

        # Prepend context if pending
        if self._pending_context:
            full_message = f"{self._pending_context}\n\n{message}"
            self._pending_context = None
        else:
            full_message = message

        # Send message with newline
        await self._pty.write(full_message + "\n")

    async def stream_response(self) -> AsyncGenerator[str, None]:
        """Stream response chunks from Claude CLI.

        Yields:
            Parsed response text chunks.
        """
        if not self._pty:
            return

        try:
            async for chunk in self._pty.read_output():
                parsed = self._parser.parse_chunk(chunk)
                if parsed:
                    yield parsed
        finally:
            self._busy = False

    async def inject_context(self, context: str) -> None:
        """Inject context to prepend to next message.

        Args:
            context: Context text to inject.
        """
        self._pending_context = context

    @property
    def is_running(self) -> bool:
        return self._pty is not None and self._pty.is_running

    @property
    def is_busy(self) -> bool:
        return self._busy

